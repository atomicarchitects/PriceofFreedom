"""Main file for running the training pipeline."""
import os

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
import wandb

from qm9 import train


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

    # Make sure the dataloader is deterministic.
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # We only support single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    # Check and freeze config.
    config = FLAGS.config

    # Initialize wandb.
    if FLAGS.use_wandb:
        os.makedirs(os.path.join(FLAGS.workdir, "wandb"), exist_ok=True)

        wandb.login()
        wandb.init(
            project="price_of_freedom",
            config=config.to_dict(),
            dir=FLAGS.workdir,
            sync_tensorboard=True,
            tags=FLAGS.wandb_tags,
            name=FLAGS.wandb_name,
            notes=FLAGS.wandb_notes,
        )

    # Start training!
    train.train_and_evaluate(config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)