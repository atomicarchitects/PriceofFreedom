"""Utility functions for creating models."""

import functools
import ml_collections
import e3nn_jax as e3nn
import optax

from src.models import readout, simple_network
from src.tensor_products import tensor_products


def create_tensor_product(tensor_product_config: ml_collections.ConfigDict):
    """Create the tensor product based on the configuration."""
    if tensor_product_config.type == "clebsch-gordan-dense":
        return functools.partial(
            tensor_products.ClebschGordanTensorProductDense,
            apply_output_linear=tensor_product_config.apply_output_linear,
            irrep_normalization=tensor_product_config.irrep_normalization,
        )
    elif tensor_product_config.type == "clebsch-gordan-sparse":
        return functools.partial(
            tensor_products.ClebschGordanTensorProductSparse,
            apply_output_linear=tensor_product_config.apply_output_linear,
            irrep_normalization=tensor_product_config.irrep_normalization,
        )
    elif tensor_product_config.type == "gaunt-s2grid":
        return functools.partial(
            tensor_products.GauntTensorProductAllParitiesS2Grid,
            num_channels=tensor_product_config.num_channels,
            res_alpha=tensor_product_config.res_alpha,
            res_beta=tensor_product_config.res_beta,
            quadrature=tensor_product_config.quadrature,
        )
    raise ValueError(f"Unknown tensor product {tensor_product_config}.")


def create_readout(config: ml_collections.ConfigDict):
    """Create the readout based on the dataset."""
    if config.dataset == "tetris":
        return readout.TetrisReadout()
    raise ValueError(f"Unknown dataset {config.dataset}.")


def create_output_irreps_per_layer(config: ml_collections.ConfigDict):
    """Create the output irreps per layer based on the configuration."""

    if config.dataset == "tetris":
        layer_irreps = 4 * (e3nn.Irreps("0e") + e3nn.Irreps("0o"))
        layer_irreps += e3nn.s2_irreps(lmax=config.hidden_lmax, p_val=1, p_arg=-1)[1:]
        layer_irreps += e3nn.s2_irreps(lmax=config.hidden_lmax, p_val=-1, p_arg=-1)[1:]
        layer_irreps *= config.hidden_irreps_multiplicity
        layer_irreps = layer_irreps.regroup()
        return [layer_irreps] * config.num_layers

    raise ValueError(f"Unknown dataset {config.dataset}.")


def create_model(config: ml_collections.ConfigDict):
    """Create the model based on the configuration."""
    output_irreps_per_layer = create_output_irreps_per_layer(config)
    tensor_products_fn = create_tensor_product(config.tensor_product)
    readout = create_readout(config)
    if config.model == "simple":
        return simple_network.SimpleNetwork(
            sh_lmax=config.sh_lmax,
            init_embed_dims=config.init_embed_dims,
            max_atomic_number=config.max_atomic_number,
            mlp_hidden_dims=config.mlp_hidden_dims,
            mlp_num_layers=config.mlp_num_layers,
            output_irreps_per_layer=output_irreps_per_layer,
            tensor_product_fn=tensor_products_fn,
            readout=readout,
        )

    raise ValueError(f"Unknown model {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict):
    """Create the optimizer based on the configuration."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate=config.learning_rate)
    raise ValueError(f"Unknown optimizer {config.optimizer}.")
