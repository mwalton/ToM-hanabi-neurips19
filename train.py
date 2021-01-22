from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from third_party.dopamine import logger

import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', '/hanabi/',
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', 'logs',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')

flags.DEFINE_string('agent_type', 'Rainbow', 'Self-play policy to run on environment')
flags.DEFINE_integer('num_players', 2, 'Number of Hanabi Players')
flags.DEFINE_string('tf_device', '/gpu:*', "Device to run model on")

# paper indicates history-less (sz 1) here; in Rainbow class, stack_size is # of frames and set to 1.
# It looks like hist_size still stacks the default 4 obs tho (inconsistent w/ paper)
flags.DEFINE_integer('history_size', 1, 'History length to stack (handle partial obs, default 4 evidently motivated by atari)')

flags.DEFINE_integer('checkpoint_every_n', 1, 'How frequently to checkpoint model, careful w/ this takes several seconds')
flags.DEFINE_integer('log_every_n', 1, 'How frequently to save logfile')
flags.DEFINE_integer('num_iterations', 20000, 'Number of training iterations to run')
flags.DEFINE_integer('belief_level', 1, "Theory of mind belief level; -1: vanilla agent (no belief) 0: my belief about my hand 1: my belief about your belief about your hand")
flags.DEFINE_string('belief_mode', 'replace', "concat: append belief to obs vector. replace: replace partial knowledge components w/ belief")
flags.DEFINE_integer('n_b0_samples', 1, "Number of samples to draw from b0 to estimate b1; if 1 sample, most likely hand will be sampled")
flags.DEFINE_bool('comms_reward', True, "Provide agent with incentive to minimize divergence between other agents' belief and their hand")
flags.DEFINE_float('beta', 2.0, "weight associated with intrinstic communication reward")
flags.DEFINE_string('mode', 'eval', "train or eval; eval mode will play on-policy for 1000 games from the last checkpoint")

def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir is None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/{}'.format(FLAGS.base_dir, FLAGS.logging_dir))

  game_type = 'Hanabi-Full-Minimal' if FLAGS.belief_mode == 'replace' else 'Hanabi-Full'
  environment = run_experiment.create_environment(game_type=game_type, num_players=FLAGS.num_players)
  belief_model = run_experiment.create_belief_model(FLAGS.belief_level, FLAGS.belief_mode, FLAGS.num_players, num_b0_samples=FLAGS.n_b0_samples, comms_reward=FLAGS.comms_reward, beta=FLAGS.beta)
  obs_stacker = run_experiment.create_obs_stacker(environment, belief_model, history_size=FLAGS.history_size)
  agent = run_experiment.create_agent(environment, obs_stacker, agent_type=FLAGS.agent_type, tf_device=FLAGS.tf_device)

  checkpoint_dir = '{}/{}'.format(FLAGS.base_dir, FLAGS.checkpoint_dir)
  start_iteration, experiment_checkpointer = (
      run_experiment.initialize_checkpointing(agent,
                                              experiment_logger,
                                              checkpoint_dir,
                                              FLAGS.checkpoint_file_prefix))

  run_experiment.run_experiment(agent, environment, start_iteration,
                                obs_stacker, belief_model,
                                experiment_logger, experiment_checkpointer,
                                checkpoint_dir,
                                logging_file_prefix=FLAGS.logging_file_prefix,
                                checkpoint_every_n=FLAGS.checkpoint_every_n,
                                log_every_n=FLAGS.log_every_n,
                                num_iterations=FLAGS.num_iterations,
                                mode=FLAGS.mode)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
