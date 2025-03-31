"""Config for a simple E(3)-equivariant model."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Model configuration.
    config.model = "simple"
    config.sh_lmax = 2
    config.hidden_lmax = 2
    config.hidden_irreps_multiplicity = 8
    config.init_embed_dims = 16
    config.max_atomic_number = 10
    config.mlp_hidden_dims = 32
    config.mlp_num_layers = 1
    config.num_layers = 3

    # Tensor product configuration, added later.
    config.tensor_product = ml_collections.ConfigDict()

    # Training configuration.
    config.dataset = "tetris"
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    config.num_training_steps = 1000
    config.profile = False
    return config
