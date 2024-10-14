import enum
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import gin
import jax
import seqio
import tensorflow as tf

from t5x import config_utils
from t5x import gin_utils

from .tasks import TevaTasks


@enum.unique
class RunMode(enum.Enum):
    TRAIN = "train"


flags.DEFINE_enum_class(
    "run_mode",
    default=None,
    enum_class=RunMode,
    help='The mode to run teva under',
)

flags.DEFINE_multi_enum_class(
    "tasks_to_load",
    default=None,
    enum_class=TevaTasks,
    help="One or more tasks to load for the run"
)

# End of Teva flags
# -----------------------
# T5X flags

flags.DEFINE_multi_string(
    'gin_file',
    default=None,
    help=(
        'Path to gin configuration file. Multiple paths may be passed and '
        'will be imported in the given order, with later configurations  '
        'overriding earlier ones.'
    ),
)

flags.DEFINE_multi_string(
    'gin_bindings', default=[], help='Individual gin bindings.'
)

flags.DEFINE_list(
    'gin_search_paths',
    default=['./t5x'],
    help=(
        'Comma-separated list of gin config path prefixes to be prepended '
        'to suffixes given via `--gin_file`. If a file appears in. Only the '
        'first prefix that produces a valid path for each suffix will be '
        'used.'
    ),
)

flags.DEFINE_string(
    'tfds_data_dir',
    None,
    'If set, this directory will be used to store datasets prepared by '
    'TensorFlow Datasets that are not available in the public TFDS GCS '
    'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
    'all `Task`s.',
)

flags.DEFINE_list(
    'seqio_additional_cache_dirs',
    [],
    'Directories to search for cached Tasks in addition to defaults.',
)

flags.DEFINE_boolean(
    'multiprocess_gpu',
    False,
    help=(
        'Initialize JAX distributed system for multi-host GPU, using '
        '`coordinator_address`, `process_count`, and `process_index`.'
    ),
)

flags.DEFINE_string(
    'coordinator_address',
    None,
    help='IP address:port for multi-host GPU coordinator.',
)

flags.DEFINE_integer(
    'process_count', None, help='Number of processes for multi-host GPU.'
)

flags.DEFINE_integer('process_index', None, help='Index of this process.')
flags.DEFINE_integer(
    'initialization_timeout',
    None,
    help=(
        'Timeout for jax.distributed.initialize. Default used is the '
        'default as specified in jax.distributed.initialize. '
    ),
)

FLAGS = flags.FLAGS


def main(argv: Sequence[str]):
    _main(argv)


def _main(argv: Sequence[str]):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # OOM fix. Prevents TF from seeing GPUs to stop conflict with JAX.
    # This must go after InitGoogle(), which is called by
    # gin_utils.run(main).
    tf.config.experimental.set_visible_devices([], 'GPU')

    if FLAGS.multiprocess_gpu:
        logging.info(
            'Initializing distributed system for multi-host GPU:\n'
            '  coordinator_address: %s\n  process_count: %s\n  process_index: %s',
            FLAGS.coordinator_address,
            FLAGS.process_count,
            FLAGS.process_index,
        )

        if FLAGS.initialization_timeout:
            if jax.__version__ < '0.4.15':
                raise ValueError(
                    'Specified'
                    f' --initialization_timeout={FLAGS.initialization_timeout}, but'
                    ' jax=={jax.__version__} does not support this yet. Use'
                    ' jax>=0.4.15'
                )
            jax.distributed.initialize(
                FLAGS.coordinator_address,
                FLAGS.process_count,
                FLAGS.process_index,
                initialization_timeout=FLAGS.initialization_timeout,
            )
        else:
            jax.distributed.initialize(
                FLAGS.coordinator_address, FLAGS.process_count, FLAGS.process_index
            )
    
    if FLAGS.tfds_data_dir:
        seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)

    seqio.add_global_cache_dirs(FLAGS.seqio_additional_cache_dirs)

    # -----
    # TODO: Support all the same commands as t5x: train, eval, infer, precompile
    # -----
    if FLAGS.run_mode == RunMode.TRAIN:
        from teva.tasks import setup_tasks
        from t5x.train import train, _DEFAULT_GIN_SEARCH_PATHS

        setup_tasks_using_gin = gin.configurable(setup_tasks)
        train_using_gin = gin.configurable(train)

        gin_utils.parse_gin_flags(
            # User-provided gin paths take precedence if relative paths conflict.
            FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
            FLAGS.gin_file,
            FLAGS.gin_bindings,
        )

        setup_tasks_using_gin(tasks=FLAGS.tasks_to_load)
        exit()

        # TODO: @theyorubayesian: Store state using alternative methods if necessary.
        step, state = train_using_gin()

    jax.effects_barrier()


if __name__ == "__main__":    
    config_utils.run(main)
