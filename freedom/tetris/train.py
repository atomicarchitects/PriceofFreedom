"""Main training script for Tetris dataset."""

from typing import Tuple, Callable
import time
import os

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax
from tqdm.auto import tqdm
import wandb
import e3nn_jax as e3nn
import ml_collections
import yaml

try:
    from ctypes import cdll
except ImportError:
    cdll = None

from src.models import utils as model_utils


def get_tetris_dataset() -> jraph.GraphsTuple:
    """Get the Tetris dataset."""

    all_positions = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    all_positions = jnp.array(all_positions, dtype=jnp.float32)
    all_labels = jnp.arange(8)

    graphs = []
    for positions, labels in zip(all_positions, all_labels):
        senders, receivers = e3nn.radius_graph(positions, 1.1)

        graphs += [
            jraph.GraphsTuple(
                nodes={
                    "numbers": jnp.ones((len(positions),), dtype=jnp.int32),  # [num_nodes, 1]
                    "positions": positions,  # [num_nodes, 3]
                },
                edges=None,
                globals=labels[None],  # [num_graphs]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([len(positions)]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)


@jax.jit
def apply_random_rotation(graphs: jraph.GraphsTuple, key: jnp.ndarray) -> jraph.GraphsTuple:
    """Apply a random rotation to the nodes of the graph."""
    alpha, beta, gamma = jax.random.uniform(key, (3,), minval=-jnp.pi, maxval=jnp.pi)

    rotated_nodes = e3nn.IrrepsArray("1o", graphs.nodes["positions"])
    rotated_nodes = rotated_nodes.transform_by_angles(alpha, beta, gamma)
    rotated_nodes = rotated_nodes.array
    rotated_graphs = graphs._replace(nodes={"positions": rotated_nodes, "numbers": graphs.nodes["numbers"]})
    return rotated_graphs


def check_equivariance(
    fn: Callable[[jraph.GraphsTuple], jnp.ndarray], graphs: jraph.GraphsTuple, num_seeds: int = 50
) -> None:
    """Check if a function is equivariant."""
    failed_seeds = []
    for key in range(num_seeds):
        key = jax.random.PRNGKey(key)
        rotated_graphs = apply_random_rotation(graphs, key)

        logits = fn(graphs)
        rotated_logits = fn(rotated_graphs)
        if not jnp.allclose(logits, rotated_logits, atol=1e-4):
            failed_seeds.append(key)
            if len(failed_seeds) == 0:
                logging.info(
                    "Model is not equivariant: error = %f",
                    jnp.max(jnp.abs(logits - rotated_logits)),
                )

    if failed_seeds:
        logging.info(
            "Model is not equivariant: failed for %0.1f%% of random seeds.",
            len(failed_seeds) / num_seeds * 100,
        )
    logging.info("Model is equivariant.")


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Train and evaluate a model based on the configuration."""

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Create the model.
    model = model_utils.create_model(config)

    # Create the optimizer.
    tx = model_utils.create_optimizer(config)

    # Create the dataset.
    graphs = get_tetris_dataset()

    # Initialize the model and optimizer.
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = jax.jit(model.init)(init_rng, graphs)
    opt_state = tx.init(params)

    # Tabulate the model.
    logging.info(f"Parameter shapes: {jax.tree_map(lambda x: x.shape, params)}")
    logging.info(f"Total parameters: {sum(jax.tree_leaves(jax.tree_map(lambda x: x.size, params)))}")

    # Wrapper around the loss function.
    def loss_fn(params: optax.Params, graphs_batch: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits = model.apply(params, graphs_batch)
        labels = graphs_batch.globals  # [num_graphs]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @jax.jit
    def update_fn(
        params: optax.Params, opt_state: optax.OptState, graphs_batch: jraph.GraphsTuple
    ) -> Tuple[optax.Params, optax.OptState, jnp.ndarray, jnp.ndarray]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, graphs_batch)
        labels = graphs_batch.globals
        preds = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(preds == labels)

        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy, preds

    # Compile the update function.
    wall = time.perf_counter()
    logging.info("Compiling...")
    _, _, accuracy, _ = update_fn(params, opt_state, graphs)
    logging.info(f"Compilation took {time.perf_counter() - wall:.1f}s")
    logging.info(f"Initial accuracy = {100 * accuracy:.2f}%")

    # Train.
    wall = time.perf_counter()
    logging.info("Training...")
    with tqdm(range(config.num_training_steps)) as bar:
        for step in bar:
            if config.profile and step == 20:
                libcudart = cdll.LoadLibrary("libcudart.so")
                libcudart.cudaProfilerStart()

            # Apply random rotations to the dataset.
            rng, step_rng = jax.random.split(rng)
            graphs_batch = apply_random_rotation(graphs, step_rng)

            # Update the parameters.
            params, opt_state, accuracy, preds = update_fn(params, opt_state, graphs_batch)

            if config.profile and step == 25:
                libcudart.cudaProfilerStop()

            if step % 5 == 0:
                wandb.log({"accuracy": accuracy, "step": step})

            bar.set_postfix(accuracy=f"{100 * accuracy:.2f}%")

    logging.info(
        f"Training for tensor_product_type={config.tensor_product} with lmax={config.hidden_lmax} took {time.perf_counter() - wall:.1f}s"
    )
    logging.info(f"Final accuracy = {100 * accuracy:.2f}%")
    logging.info(f"Final prediction: {preds}")

    # Check equivariance.
    logging.info("Checking equivariance...")
    apply_fn = lambda graphs: model.apply(params, graphs)
    apply_fn = jax.jit(apply_fn)
    check_equivariance(apply_fn, graphs)
