"""Library file for executing the training and evaluation of models on QM9."""

import functools
import os
from typing import Dict, Iterable, Tuple, Union
import time

import chex
import flax
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import optax
import yaml
from absl import logging

from clu import (
    metric_writers,
    metrics,
    parameter_overview,
    periodic_actions,
)

from src.qm9 import datatypes, hooks, models, train_state, input_pipeline, loss


@flax.struct.dataclass
class Metrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_and_atom_type_loss: metrics.Average.from_output("focus_and_atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


def device_batch(
    fragment_iterator: Iterable[datatypes.Fragments],
) -> Iterable[datatypes.Fragments]:
    """Batches a set of graphs to the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, fragment in enumerate(fragment_iterator):
        batch.append(fragment)

        if idx % num_devices == num_devices - 1:
            batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *batch)
            batch = datatypes.Fragments.from_graphstuple(batch)
            yield batch
            batch = []


def create_optimizer(config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Create an optimizer as specified by the config."""
    if config.optimizer == "adam":
        tx = optax.adam(learning_rate=config.learning_rate)
    elif config.optimizer == "sgd":
        tx = optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)

    if not config.get("gradient_clip_norm"):
        return tx

    logging.info(
        "Applying gradient clipping with norm %0.2f.", config.gradient_clip_norm
    )
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        tx,
    )


@functools.partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(3, 4, 5))
@chex.assert_max_traces(n=2)
def train_step(
    graphs: datatypes.Fragments,
    state: train_state.TrainState,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params: optax.Params, graphs: datatypes.Fragments) -> float:
        preds = state.apply_fn(params, None, graphs)
        total_loss, (
            focus_and_atom_type_loss,
            position_loss,
        ) = loss.regression_loss(preds=preds, graphs=graphs, **loss_kwargs)
        mask = jraph.get_graph_padding_mask(graphs)
        mean_loss = jnp.sum(jnp.where(mask, total_loss, 0.0)) / jnp.sum(mask)
        return mean_loss, (
            total_loss,
            focus_and_atom_type_loss,
            position_loss,
            mask,
        )

    # Compute gradients.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (
        _,
        (total_loss, focus_and_atom_type_loss, position_loss, mask),
    ), grads = grad_fn(state.params, graphs)

    # Average gradients across devices.
    grads = jax.lax.pmean(grads, axis_name="device")
    state = state.apply_gradients(grads=grads)

    # Compute metrics.
    batch_metrics = Metrics.gather_from_model_output(
        axis_name="device",
        total_loss=total_loss,
        focus_and_atom_type_loss=focus_and_atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )
    return state, batch_metrics


# @functools.partial(jax.jit, static_argnums=(2,))
@functools.partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(2,))
@chex.assert_max_traces(n=2)
def evaluate_step(
    graphs: datatypes.Fragments,
    state: train_state.TrainState,
    loss_kwargs: Dict[str, Union[float, int]],
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # Compute predictions and resulting loss.
    preds = state.apply_fn(state.params, None, graphs)
    total_loss = loss.regression_loss(preds=preds, graphs=graphs, **loss_kwargs)

    # Consider only valid graphs.
    mask = jraph.get_graph_padding_mask(graphs)
    return Metrics.gather_from_model_output(
        axis_name="device",
        total_loss=total_loss,
        focus_and_atom_type_loss=focus_and_atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )


def evaluate_model(
    state: train_state.TrainState,
    datasets: Dict[str, Iterable[datatypes.Fragments]],
    loss_kwargs: Dict[str, Union[float, int]],
    num_eval_steps: int,
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split, fragment_iterator in datasets.items():
        split_metrics = Metrics.empty()
        split_metrics = flax.jax_utils.replicate(split_metrics)

        # Loop over graphs.
        for eval_step, graphs in enumerate(device_batch(fragment_iterator)):
            if eval_step >= num_eval_steps:
                break

            # Compute metrics for this batch.
            graphs = jax.tree_util.tree_map(jnp.asarray, graphs)
            batch_metrics = evaluate_step(graphs, state, loss_kwargs)
            split_metrics = split_metrics.merge(batch_metrics)

        split_metrics = flax.jax_utils.unreplicate(split_metrics)
        eval_metrics[split + "_eval"] = split_metrics

    return eval_metrics


def train_and_evaluate(
    config: ml_collections.FrozenConfigDict, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the TensorBoard summaries are written to.

    Returns:
      The final train state
    """
    # We only support single-host training.
    assert jax.process_count() == 1

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)
    datasets = input_pipeline.get_datasets(dataset_rng, config)

    # Create and initialize the network.
    logging.info("Initializing network.")
    init_graphs = next(datasets["train"])
    net = models.create_model(config, run_in_evaluation_mode=False)

    rng, init_rng = jax.random.split(rng)
    params = jax.jit(net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(config)

    # Create a corresponding evaluation function for generation.
    eval_net = models.create_model(config, run_in_evaluation_mode=True)

    # Create the training state.
    state = train_state.TrainState.create(
        apply_fn=jax.jit(net.apply),
        eval_apply_fn=jax.jit(eval_net.apply),
        params=params,
        tx=tx,
        best_params=params,
        step_for_best_params=0,
        metrics_for_best_params={},
        train_metrics=Metrics.empty(),
    )

    # Set up checkpointing of the model.
    # We will record the best model seen during training.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_hook = hooks.CheckpointHook(checkpoint_dir, max_to_keep=1)
    state = checkpoint_hook.restore_or_initialize(state)
    initial_step = state.step

    # Replicate the training and evaluation state across devices.
    state = flax.jax_utils.replicate(state)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    profiler = periodic_actions.Profile(
        logdir=workdir,
        every_secs=10800,
    )
    train_metrics_hook = hooks.LogTrainMetricsHook(writer)
    evaluate_model_hook = hooks.EvaluateModelHook(
        evaluate_model_fn=lambda state: evaluate_model(
            state,
            datasets,
            config.loss_kwargs,
            config.num_eval_steps,
        ),
        writer=writer,
    )

    # Begin training loop.
    logging.info("Starting training.")
    for step in range(initial_step, config.num_train_steps):
        # Log, if required.
        first_or_last_step = step in [initial_step, config.num_train_steps]
        if step % config.log_every_steps == 0 or first_or_last_step:
            state = train_metrics_hook(state)

        # Evaluate model, if required.
        if config.eval and (step % config.eval_every_steps == 0 or first_or_last_step):
            logging.info("Evaluating model.")
            state = evaluate_model_hook(state)
            checkpoint_hook(state)

        # Get a batch of graphs.
        try:
            start = time.perf_counter()
            graphs = next(device_batch(datasets["train"]))
            graphs = jax.tree_util.tree_map(jnp.asarray, graphs)
            logging.log_first_n(
                logging.INFO,
                "Time to get next batch of fragments: %0.2f ms.",
                10,
                (time.perf_counter() - start) * 1e3,
            )

        except StopIteration:
            logging.info("No more training data. Continuing with final evaluation.")
            break

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            state, batch_metrics = train_step(
                graphs,
                state,
                config.loss_kwargs,
            )

            # Update metrics.
            state = state.replace(
                train_metrics=state.train_metrics.merge(batch_metrics)
            )

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
        report_progress(step)
        profiler(step)

    return state