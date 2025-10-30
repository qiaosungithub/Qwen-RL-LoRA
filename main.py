from absl import app, flags
from ml_collections import config_flags

import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_enum(
    "mode",
    enum_values=["local_debug", "remote_debug", "remote_run"],
    default="remote_run",
    help="Running mode.",
)  # NOTE: This variable isn't used currently, but maintained for future use. This at least ensures that there is no more variable that must be passed in from the command line.

flags.DEFINE_bool("debug", False, "Debugging mode.")
config_flags.DEFINE_config_file(
    "config",
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")


    return train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir", "mode", "config"])
    app.run(main)