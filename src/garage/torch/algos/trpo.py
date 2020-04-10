"""Trust Region Policy Optimization."""
from dowel import logger
import torch

from garage.np.optimizers import BatchDataset
from garage.torch.algos import VPG
from garage.torch.optimizers import ConjugateGradientOptimizer


class TRPO(VPG):
    """Trust Region Policy Optimization (TRPO).

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for policy. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' = 1e-3})`
        vf_optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for value function. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' = 1e-3})`
        vf_lr (float): Learning rate for value function parameters.
        max_path_length (int): Maximum length of a single rollout.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        minibatch_size (int): Batch size for optimization.
        optimization_epochs (int): Number of epochs for update.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 policy_optimizer=(ConjugateGradientOptimizer,
                                   dict(max_constraint_value=0.01)),
                 vf_optimizer=torch.optim.Adam,
                 vf_lr=3e-4,
                 max_path_length=100,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 minibatch_size=32,
                 optimization_epochs=10):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         vf_lr=vf_lr,
                         max_path_length=max_path_length,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         minibatch_size=minibatch_size,
                         optimization_epochs=optimization_epochs)

    def _train(self, obs_flat, actions_flat, rewards_flat, returns_flat,
               advs_flat):
        policy_loss = self._train_policy(obs_flat, actions_flat, rewards_flat,
                                         advs_flat)
        logger.log('Policy loss: {}'.format(policy_loss))

        batch_dataset = BatchDataset((obs_flat, returns_flat),
                                     self._minibatch_size)
        for epoch in range(self._optimization_epochs):
            for obs, returns in batch_dataset.iterate():
                vf_loss = self._train_value_function(obs, returns)
            logger.log('Mini epoch: {} | VF Loss: {}'.format(epoch, vf_loss))

    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        with torch.no_grad():
            old_ll = self._old_policy.log_likelihood(obs, actions)

        new_ll = self.policy.log_likelihood(obs, actions)
        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        return surrogate

    def _optimize_policy(self, obs, actions, rewards, advantages):
        r"""Performs a optimization.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.

        """
        self.policy_optimizer.step(
            f_loss=lambda: self._compute_loss_with_adv(obs, actions, rewards,
                                                       advantages),
            f_constraint=lambda: self._compute_kl_constraint(obs))
