"""Main file for running the training pipeline."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import wandb

from src import train


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("use_wandb", True, "Whether to log to Weights & Biases.")
flags.DEFINE_list("wandb_tags", [], "Tags to add to the Weights & Biases run.")
flags.DEFINE_string(
    "wandb_name",
    None,
    "Name of the Weights & Biases run. Uses the Weights & Biases default if not specified.",
)
flags.DEFINE_string("wandb_notes", None, "Notes for the Weights & Biases run.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # We only support single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Initialize wandb.
    os.makedirs(os.path.join(FLAGS.workdir, "wandb"), exist_ok=True)

    wandb.login()
    wandb.init(
        project="price_of_freedom",
        config=FLAGS.config.to_dict(),
        dir=FLAGS.workdir,
        tags=FLAGS.wandb_tags,
        name=FLAGS.wandb_name,
        notes=FLAGS.wandb_notes,
    )

    # Start training!
    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
