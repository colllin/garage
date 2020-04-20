#!/usr/bin/env python3
"""This is an example to train TRPO on MT10 environment."""
# pylint: disable=no-value-for-parameter
import click
from metaworld.benchmarks import MT10
import torch

from garage import wrap_experiment
from garage.envs import MultiEnvWrapper
from garage.envs.multi_env_wrapper import round_robin_strategy
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@wrap_experiment(snapshot_mode='all')
def torch_trpo_mt10(ctxt, seed, epochs, batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    set_seed(seed)
    tasks = MT10.get_train_tasks().all_task_names
    envs = []
    for task in tasks:
        envs.append(MT10.from_task(task))
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          metaworld_mt=True)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = LinearFeatureBaseline(env_spec=env.spec)

    runner = LocalRunner(ctxt)
    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                max_kl_step=0.02,
                max_path_length=128,
                discount=0.99,
                gae_lambda=0.95)

    runner.setup(algo, env)
    runner.train(n_epochs=epochs, batch_size=batch_size)


torch_trpo_mt10()
