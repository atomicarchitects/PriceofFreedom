"""Config for a simple E(3)-equivariant model."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Model configuration.
    config.model = "simple"
    config.sh_lmax = 2
    config.lmax = 2
    config.init_node_features = 16
    config.max_atomic_number = 10
    config.num_hops = 3
    config.tensor_product = "cgtp_naive"

    # Training configuration.
    config.dataset = "qm9"
    config.dataset_seed = 0
    config.root = "data"
    config.target_property = "mu"
    config.radial_cutoff = 2.0
    config.add_self_edges = True
    config.num_train_molecules = 110000
    config.num_val_molecules = 10000
    config.batch_size = 32
    config.num_training_steps = 100000
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # Evaluation configuration.
    config.evaluate_every = 10000
    config.num_eval_steps = 1000

    return config
