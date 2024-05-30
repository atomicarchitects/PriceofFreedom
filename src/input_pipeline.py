"""Exposes the datasets in a convenient format."""

from typing import Dict

import ml_collections
import tensorflow as tf

from src.data import md17


def get_datasets(
    config: ml_collections.ConfigDict,
) -> Dict[str, tf.data.Dataset]:
    """Returns datasets of batched GraphsTuples, organized by split."""
    if config.dataset == "md17":
        return md17.get_md17_datasets(
            molecule=config.molecule,
            batch_size=config.batch_size,
            cutoff=config.cutoff,
            add_self_edges=config.add_self_edges,
            splits=config.splits,
            seed=config.rng_seed,
            datadir=config.datadir,
        )

    raise ValueError(f"Unknown dataset {config.dataset}.")
