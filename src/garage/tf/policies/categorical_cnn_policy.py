"""CategoricalCNNPolicy with model."""
import akro
import tensorflow as tf

from garage.tf.distributions import Categorical
from garage.tf.models import CNNModel
from garage.tf.models import MLPModel
from garage.tf.models import Sequential
from garage.tf.policies import StochasticPolicy


class CategoricalCNNPolicy(StochasticPolicy):
    """CategoricalCNNPolicy.

    A policy that contains a CNN and a MLP to make prediction based on
    a categorical distribution.

    It only works with akro.Discrete action space.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        conv_filter_sizes(tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter for
            first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        conv_filters(tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        conv_strides(tuple[int]): The stride of the sliding window. For
            example, (1, 2) means there are two convolutional layers. The
            stride of the filter for first layer is 1 and that of the second
            layer is 2.
        conv_pad (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        name (str): Policy name, also the variable scope of the policy.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means the MLP of this policy consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 conv_filters,
                 conv_filter_sizes,
                 conv_strides,
                 conv_pad,
                 name='CategoricalCNNPolicy',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        if not isinstance(env_spec.action_space, akro.Discrete):
            raise ValueError(
                'CategoricalCNNPolicy only works with akro.Discrete action '
                'space.')

        if not isinstance(env_spec.observation_space, akro.Box) or \
                not len(env_spec.observation_space.shape) in (2, 3):
            raise ValueError(
                '{} can only process 2D, 3D akro.Image or'
                ' akro.Box observations, but received an env_spec with '
                'observation_space of type {} and shape {}'.format(
                    type(self).__name__,
                    type(env_spec.observation_space).__name__,
                    env_spec.observation_space.shape))

        super().__init__(name, env_spec)
        self.obs_dim = env_spec.observation_space.shape
        self.action_dim = env_spec.action_space.n

        self.model = Sequential(
            CNNModel(filter_dims=conv_filter_sizes,
                     num_filters=conv_filters,
                     strides=conv_strides,
                     padding=conv_pad,
                     hidden_nonlinearity=hidden_nonlinearity,
                     name='CNNModel'),
            MLPModel(output_dim=self.action_dim,
                     hidden_sizes=hidden_sizes,
                     hidden_nonlinearity=hidden_nonlinearity,
                     hidden_w_init=hidden_w_init,
                     hidden_b_init=hidden_b_init,
                     output_nonlinearity=output_nonlinearity,
                     output_w_init=output_w_init,
                     output_b_init=output_b_init,
                     layer_normalization=layer_normalization,
                     name='MLPModel'))

        self._initialize()

    def _initialize(self):
        if isinstance(self.env_spec.observation_space, akro.Image):
            state_input = tf.compat.v1.placeholder(tf.uint8,
                                                   shape=(None, ) +
                                                   self.obs_dim)
            state_input = tf.cast(state_input, tf.float32)
            state_input /= 255.0
        else:
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) +
                                                   self.obs_dim)

        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            self.model.build(state_input)

        self._f_prob = tf.compat.v1.get_default_session().make_callable(
            self.model.outputs, feed_list=[self.model.input])

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if primitive supports vectorized operations.
        """
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        """Build a symbolic graph of the distribution parameters.

        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            state_info_vars (dict[np.ndarray]): Extra state information, e.g.
                previous action.
            name (str): Name for symbolic graph.

        Returns:
            dict[tf.Tensor]: Outputs of the symbolic graph of distribution
                parameters.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            if isinstance(self.env_spec.observation_space, akro.Image):
                obs_var = tf.cast(obs_var, tf.float32) / 255.0

            prob = self.model.build(obs_var, name=name)
        return dict(prob=prob)

    def dist_info(self, obs, state_infos=None):
        """Get distribution parameters.

        Args:
            obs (np.ndarray): Observation input.
            state_infos (dict[np.ndarray]): Extra state information, e.g.
                previous action.

        Returns:
            dict[np.ndarray]: Distribution parameters.

        """
        prob = self._f_prob(obs)
        return dict(prob=prob)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict[str: np.ndarray]: Action distribution.

        """
        if len(observation.shape) < len(self.obs_dim):
            observation = self.env_spec.observation_space.unflatten(
                observation)
        prob = self._f_prob([observation])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Predicted actions.
            dict[str: np.ndarray]: Action distributions.

        """
        if len(observations[0].shape) < len(self.obs_dim):
            observations = self.env_spec.observation_space.unflatten_n(
                observations)
        probs = self._f_prob(observations)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        """Policy distribution.

        Returns:
            garage.tf.distributions.Categorical: Policy distribution.

        """
        return Categorical(self.action_dim)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
