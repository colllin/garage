import random

import numpy as np
import pytest

from garage.envs.grid_world_env import GridWorldEnv
from garage.experiment.task_sampler import EnvPoolSampler
from garage.np.policies import ScriptedPolicy
from garage.sampler import LocalSampler, VecWorker, WorkerFactory
from garage.tf.envs import TfEnv

SEED = 100
N_TRAJ = 5
MAX_PATH_LENGTH = 9


@pytest.fixture
def env():
    return TfEnv(GridWorldEnv(desc='4x4'))


@pytest.fixture
def random_policies():
    # These constants are for a 4x4 gridworld
    return [
        ScriptedPolicy(scripted_actions=np.random.randint(4, size=16))
        for _ in range(N_TRAJ)
    ]


@pytest.fixture
def policy():
    return ScriptedPolicy(
        scripted_actions=[2, 2, 1, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1])


def make_random_env():
    desc = [list('SFFF'), list('FHFH'), list('FFFH'), list('HFFG')]
    for line in desc:
        random.shuffle(line)
    random.shuffle(desc)
    desc = [''.join(line) for line in desc]
    return TfEnv(GridWorldEnv(desc=desc))


@pytest.fixture
def random_envs():
    return [make_random_env() for _ in range(N_TRAJ)]


def test_rollout(env, policy):
    worker = VecWorker(seed=SEED,
                       max_path_length=MAX_PATH_LENGTH,
                       worker_number=0,
                       n_envs=N_TRAJ)
    worker.update_agent(policy)
    worker.update_env(env)
    traj = worker.rollout()
    assert len(traj.lengths) == N_TRAJ
    traj2 = worker.rollout()
    assert len(traj2.lengths) == N_TRAJ
    worker.shutdown()


def assert_trajs_eq(ground_truth_traj, test_traj):
    # We should have the exact same trajectories.
    assert (ground_truth_traj.lengths == test_traj.lengths).all()
    # To ensure the next line is actually testing something
    assert test_traj.actions.var() > 0
    assert (ground_truth_traj.actions == test_traj.actions).all()
    assert (ground_truth_traj.observations == test_traj.observations).all()


def test_in_local_sampler(random_policies, random_envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers,
                                                    random_policies,
                                                    random_envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers,
                                                   [random_policies],
                                                   [random_envs])
    n_samples = 100
    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    vec_trajs = true_sampler.obtain_samples(0, n_samples, None)
    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_reset_optimization(random_policies, random_envs):
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers,
                                                    random_policies,
                                                    random_envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers,
                                                   [random_policies],
                                                   [random_envs])
    n_samples = 3 * MAX_PATH_LENGTH
    true_sampler.obtain_samples(0, n_samples, None)
    true_sampler.obtain_samples(0, n_samples, None)

    new_envs = [make_random_env() for _ in range(N_TRAJ)]

    true_trajs = true_sampler.obtain_samples(0, n_samples, None, new_envs)
    vec_trajs = true_sampler.obtain_samples(0, n_samples, None, new_envs)

    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()


def test_init_with_env_updates(random_policies, random_envs):
    task_sampler = EnvPoolSampler(random_envs)
    envs = task_sampler.sample(N_TRAJ)
    true_workers = WorkerFactory(seed=100,
                                 n_workers=N_TRAJ,
                                 max_path_length=MAX_PATH_LENGTH)
    true_sampler = LocalSampler.from_worker_factory(true_workers,
                                                    random_policies, envs)
    vec_workers = WorkerFactory(seed=100,
                                n_workers=1,
                                worker_class=VecWorker,
                                worker_args=dict(n_envs=N_TRAJ),
                                max_path_length=MAX_PATH_LENGTH)
    vec_sampler = LocalSampler.from_worker_factory(vec_workers,
                                                   [random_policies], [envs])
    n_samples = 100
    true_trajs = true_sampler.obtain_samples(0, n_samples, None)
    vec_trajs = true_sampler.obtain_samples(0, n_samples, None)

    assert vec_trajs.lengths.sum() >= n_samples
    assert_trajs_eq(true_trajs, vec_trajs)

    true_sampler.shutdown_worker()
    vec_sampler.shutdown_worker()
