"""Worker that "vectorizes" environments."""
import collections
import copy

import gym
import numpy as np

from garage import TrajectoryBatch
from garage.sampler.env_update import EnvUpdate
from garage.sampler.worker import DefaultWorker


class VecWorker(DefaultWorker):
    """Worker with a single policy and multiple environemnts.

    Alternates between taking a single step in all environments and asking the
    policy for an action for every environment. This allows computing a batch
    of actions, which is generally much more efficient than computing a single
    action when using neural networks.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_path_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker this update is
            occurring in. This argument is used to set a different seed for
            each worker.
        n_envs (int): Number of environment copies to use.

    """

    DEFAULT_N_ENVS = 8

    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 n_envs=DEFAULT_N_ENVS):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._n_envs = n_envs
        self._completed_rollouts = []
        self._needs_agent_reset = True
        self._needs_env_reset = True
        self._envs = [None] * n_envs
        self._agents = [None] * n_envs
        self._path_lengths = [0] * self._n_envs

    def update_agent(self, agent_update):
        """Update an agent, assuming it implements garage.Policy.

        Args:
            agent_update (np.ndarray or dict or garage.Policy): If a
                tuple, dict, or np.ndarray, these should be parameters to
                agent, which should have been generated by calling
                `policy.get_param_values`. Alternatively, a policy itself. Note
                that other implementations of `Worker` may take different types
                for this parameter.

        """
        super().update_agent(agent_update)
        self._needs_agent_reset = True

    def update_env(self, env_update):
        """Use any non-None env_update as a new environment.

        A simple env update function. If env_update is not None, it should be
        the complete new environment.

        This allows changing environments by passing the new environment as
        `env_update` into `obtain_samples`.

        Args:
            env_update(gym.Env or EnvUpdate or None): The environment to
                replace the existing env with. Note that other implementations
                of `Worker` may take different types for this parameter.

        Raises:
            TypeError: If env_update is not one of the documented types.
            ValueError: If the wrong number of updates is passed.

        """
        if isinstance(env_update, list):
            if len(env_update) != self._n_envs:
                raise ValueError('If separate environments are passed for '
                                 'each worker, there must be exactly n_envs '
                                 '({}) environments, but received {} '
                                 'environments.'.format(
                                     self._n_envs, len(env_update)))
            for env_index, env_up in enumerate(env_update):
                self._update_env_inner(env_up, env_index)
        elif env_update is not None:
            for env_index in range(self._n_envs):
                self._update_env_inner(copy.deepcopy(env_update), env_index)

    def _update_env_inner(self, env_update, env_index):
        """Update a single environment.

        Args:
            env_update(gym.Env or EnvUpdate or None): The environment to
                replace the existing env with. Note that other implementations
                of `Worker` may take different types for this parameter.
            env_index (int): Number of the environment to update.

        Raises:
            TypeError: If env_update is not one of the documented types.

        """
        if isinstance(env_update, EnvUpdate):
            self._envs[env_index] = env_update(self._envs[env_index])
            self._needs_env_reset = True
        elif isinstance(env_update, gym.Env):
            if self._envs[env_index] is not None:
                self._envs[env_index].close()
            self._envs[env_index] = env_update
            self._needs_env_reset = True
        else:
            raise TypeError('Unknown environment update type.')

    def start_rollout(self):
        """Begin a new rollout."""
        if self._needs_agent_reset or self._needs_env_reset:
            n = len(self._envs)
            self.agent.reset([True] * n)
            if self._needs_env_reset:
                self._prev_obs = np.asarray(
                    [env.reset() for env in self._envs])
            else:
                # Avoid calling reset on environments that are already at the
                # start of a rollout.
                for i, env in enumerate(self._envs):
                    if self._path_lengths[i] > 0:
                        self._prev_obs[i] = env.reset()
            self._path_lengths = [0 for _ in range(n)]
            self._observations = [[] for _ in range(n)]
            self._actions = [[] for _ in range(n)]
            self._rewards = [[] for _ in range(n)]
            self._terminals = [[] for _ in range(n)]
            self._env_infos = [collections.defaultdict(list) for _ in range(n)]
            self._agent_infos = [
                collections.defaultdict(list) for _ in range(n)
            ]

    def _gather_rollout(self, rollout_number, last_observation):
        traj = TrajectoryBatch(
            self._envs[rollout_number].spec,
            np.asarray(self._observations[rollout_number]),
            np.asarray([last_observation]),
            np.asarray(self._actions[rollout_number]),
            np.asarray(self._rewards[rollout_number]),
            np.asarray(self._terminals[rollout_number]),
            self._env_infos[rollout_number], self._agent_infos[rollout_number],
            np.asarray([self._path_lengths[rollout_number]]))
        self._completed_rollouts.append(traj)
        self._observations[rollout_number] = []
        self._actions[rollout_number] = []
        self._rewards[rollout_number] = []
        self._terminals[rollout_number] = []
        self._path_lengths[rollout_number] = 0
        self._prev_obs[rollout_number] = self._envs[rollout_number].reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff at least one of the paths was completed.

        """
        finished = False
        actions, agent_info = self.agent.get_actions(self._prev_obs)
        next_obs = []
        completes = [False] * len(self._envs)
        for i, action in enumerate(actions):
            next_o, r, d, env_info = self._envs[i].step(action)
            if self._path_lengths[i] >= self._max_path_length or d:
                # This path is ending.
                self._gather_rollout(i, next_o)
                completes[i] = True
                finished = True
            elif self._path_lengths[i] < self._max_path_length:
                # This path hasn't timed out.
                next_obs.append(next_o)
                self._observations[i].append(self._prev_obs[i])
                self._rewards[i].append(r)
                self._actions[i].append(actions[i])
                for k, v in agent_info.items():
                    self._agent_infos[i][k].append(v[i])
                for k, v in env_info.items():
                    self._env_infos[i][k].append(v)
                self._path_lengths[i] += 1
                self._terminals[i].append(d)
        if finished:
            self.agent.reset(completes)
        return finished

    def collect_rollout(self):
        """Collect all completed rollouts.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
        if len(self._completed_rollouts) == 1:
            result = self._completed_rollouts[0]
        else:
            result = TrajectoryBatch.concatenate(*self._completed_rollouts)
        self._completed_rollouts = []
        return result

    def shutdown(self):
        """Close the worker's environments."""
        for env in self._envs:
            env.close()
