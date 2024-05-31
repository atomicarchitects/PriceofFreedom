from typing import Tuple, Dict, Iterator

from absl import logging
import os
import functools
import yaml


import jax
import jax.numpy as jnp
import optax
import tqdm
import jraph
import ml_collections
import wandb

from src.data.qm9 import QM9Dataset
from src import models
from src import tensor_products


def get_datasets(
    config: ml_collections.ConfigDict,
) -> Dict[str, Iterator[jraph.GraphsTuple]]:
    """Get the datasets based on the configuration."""

    if config.dataset == "qm9":
        ds = QM9Dataset(
            root=config.root,
            target_property=config.target_property,
            radial_cutoff=config.radial_cutoff,
            add_self_edges=config.add_self_edges,
            splits={
                "train": config.num_train_molecules,
                "val": config.num_val_molecules,
            },
            seed=config.dataset_seed,
        )
        return ds.get_datasets(batch_size=config.batch_size)

    raise ValueError(f"Unknown dataset {config.dataset}.")


def create_model(config: ml_collections.ConfigDict):
    """Create the model based on the configuration."""
    if config.tensor_product == "cgtp_naive":
        tensor_products_fn = functools.partial(
            tensor_products.TensorProductNaive,
            output_linear=False,
            irrep_normalization="norm",
        )
    elif config.tensor_product == "cgtp_opt":
        tensor_products_fn = functools.partial(
            tensor_products.TensorProductOptimized,
            output_linear=False,
            irrep_normalization="norm",
        )
    elif config.tensor_product == "gaunt":
        tensor_products_fn = functools.partial(
            tensor_products.GauntTensorProduct,
            num_channels=config.tensor_product.num_channels,
            res_alpha=config.tensor_product.res_alpha,
            res_beta=config.tensor_product.res_beta,
            quadrature=config.tensor_product.quadrature,
        )
    elif config.tensor_product == "vector_gaunt":
        tensor_products_fn = functools.partial(
            tensor_products.VectorGauntTensorProduct,
            num_channels=config.tensor_product.num_channels,
            res_alpha=config.tensor_product.res_alpha,
            res_beta=config.tensor_product.res_beta,
            quadrature=config.tensor_product.quadrature,
        )

    if config.model == "simple":
        return models.SimpleNetwork(
            sh_lmax=config.sh_lmax,
            lmax=config.lmax,
            init_node_features=config.init_node_features,
            max_atomic_number=config.max_atomic_number,
            num_hops=config.num_hops,
            output_dims=1,
            tensor_product_fn=tensor_products_fn,
        )

    raise ValueError(f"Unknown model {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict):
    """Create the optimizer based on the configuration."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate=config.learning_rate)
    raise ValueError(f"Unknown optimizer {config.optimizer}.")


def train_and_evaluate(config, workdir):
    """Train and evaluate a model based on the configuration."""

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Load the datasets.
    logging.info("Loading datasets...")
    datasets = get_datasets(config)

    # Create the model.
    model = create_model(config)
    params = model.init(jax.random.PRNGKey(0), graphs=next(datasets["train"]))
    apply_fn = jax.jit(model.apply)

    # Optimizer
    tx = create_optimizer(config)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(
        params: optax.Params, graphs: jraph.GraphsTuple
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """A simple mean squared error loss."""
        preds = apply_fn(params, graphs)
        labels = graphs.globals
        assert preds.shape == labels.shape, (preds.shape, labels.shape)
        loss = (preds - labels) ** 2
        loss = jnp.mean(loss)
        return loss, preds

    @jax.jit
    def update_fn(params, opt_state, graphs):
        """Update the parameters of the model one time."""
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, preds), grads = grad_fn(params, graphs)

        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, preds

    def evaluate(params, dataset, num_steps):
        total_loss = 0.0
        for _ in tqdm.trange(num_steps, desc="Validation"):
            graphs = next(dataset)
            loss, _ = loss_fn(params, graphs)
            total_loss += loss
        return total_loss / num_steps

    logging.info("Starting training...")
    with tqdm.tqdm(range(config.num_training_steps)) as bar:
        for step in bar:
            graphs = next(datasets["train"])
            params, opt_state, loss, _ = update_fn(params, opt_state, graphs)

            bar.set_postfix(loss=f"{loss:.2f}")
            if step % 100 == 0:
                wandb.log({"train_loss": loss})

            if step % config.evaluate_every == 0:
                val_loss = evaluate(
                    params, datasets["val"], num_steps=config.num_eval_steps
                )
                wandb.log({"val_loss": val_loss})
