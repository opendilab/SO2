from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from ding.torch_utils import pytorch_util as ptu
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ..common import RegressionHead, ReparameterizationHead


@MODEL_REGISTRY.register('ensemble_qac')
class ENSEMBLEQAC(nn.Module):
    r"""
    Overview:
        The QAC model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic']

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            actor_head_type: str,
            twin_critic: bool = False,
            actor_head_hidden_size: int = 64,
            actor_head_layer_num: int = 2,
            critic_head_hidden_size: int = 64,
            critic_ensemble_size: int = 2,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None,
    ) -> None:
        r"""
        Overview:
            Init the QAC Model according to arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation's space.
            - action_shape (:obj:`Union[int, SequenceType]`): Action's space.
            - actor_head_type (:obj:`str`): Whether choose ``regression`` or ``reparameterization``.
            - twin_critic (:obj:`bool`): Whether include twin critic.
            - actor_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to actor-nn's ``Head``.
            - actor_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for actor's nn.
            - critic_head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` to pass to critic-nn's ``Head``.
            - critic_head_layer_num (:obj:`int`):
                The num of layers used in the network to compute Q value output for critic's nn.
            - activation (:obj:`Optional[nn.Module]`):
                The type of activation function to use in ``MLP`` the after ``layer_fn``,
                if ``None`` then default set to ``nn.ReLU()``
            - norm_type (:obj:`Optional[str]`):
                The type of normalization to use, see ``ding.torch_utils.fc_block`` for more details.
        """
        super(ENSEMBLEQAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape: int = squeeze(action_shape)
        self.actor_head_type = actor_head_type
        assert self.actor_head_type in ['regression', 'reparameterization']
        if self.actor_head_type == 'regression':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                RegressionHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    final_tanh=True,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        elif self.actor_head_type == 'reparameterization':
            self.actor = nn.Sequential(
                nn.Linear(obs_shape, actor_head_hidden_size), activation,
                ReparameterizationHead(
                    actor_head_hidden_size,
                    action_shape,
                    actor_head_layer_num,
                    sigma_type='conditioned',
                    activation=activation,
                    norm_type=norm_type
                )
            )
        self.critic = ParallelizedEnsembleFlattenMLP(
            ensemble_size=critic_ensemble_size,
            hidden_sizes=[critic_head_hidden_size]*3,
            input_size=obs_shape + action_shape,
            output_size=1,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
        )

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        r"""
        Overview:
            Use bbservation and action tensor to predict output.
            Parameter updates with QAC's MLPs forward setup.
        Arguments:
            Forward with ``'compute_actor'``:
                - inputs (:obj:`torch.Tensor`):
                    The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                    Whether ``actor_head_hidden_size`` or ``critic_head_hidden_size`` depend on ``mode``.

            Forward with ``'compute_critic'``, inputs (`Dict`) Necessary Keys:
                - ``obs``, ``action`` encoded tensors.

            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of network forward.

                Forward with ``'compute_actor'``, Necessary Keys (either):
                    - action (:obj:`torch.Tensor`): Action tensor with same size as input ``x``.
                    - logit (:obj:`torch.Tensor`):
                        Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.

                Forward with ``'compute_critic'``, Necessary Keys:
                    - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Actor Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Critic Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is``action_shape``
            - logit (:obj:`torch.FloatTensor`): :math:`(B, N2)`, where B is batch size and N3 is ``action_shape``

        Actor Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])

        Critic Examples:
            >>> inputs = {'obs': torch.randn(4,N), 'action': torch.randn(4,1)}
            >>> model = QAC(obs_shape=(N, ),action_shape=1,actor_head_type='regression')
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)

        """
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(self, inputs: torch.Tensor) -> Dict:
        r"""
        Overview:
            Use encoded embedding tensor to predict output.
            Execute parameter updates with ``'compute_actor'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - inputs (:obj:`torch.Tensor`):
                The encoded embedding tensor, determined with given ``hidden_size``, i.e. ``(B, N=hidden_size)``.
                ``hidden_size = actor_head_hidden_size``
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Outputs of forward pass encoder and head.

        ReturnsKeys (either):
            - action (:obj:`torch.Tensor`): Continuous action tensor with same size as ``action_shape``.
            - logit (:obj:`torch.Tensor`):
                Logit tensor encoding ``mu`` and ``sigma``, both with same size as input ``x``.
        Shapes:
            - inputs (:obj:`torch.Tensor`): :math:`(B, N0)`, B is batch size and N0 corresponds to ``hidden_size``
            - action (:obj:`torch.Tensor`): :math:`(B, N0)`
            - logit (:obj:`list`): 2 elements, mu and sigma, each is the shape of :math:`(B, N0)`.
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, B is batch size.
        Examples:
            >>> # Regression mode
            >>> model = QAC(64, 64, 'regression')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> assert actor_outputs['action'].shape == torch.Size([4, 64])
            >>> # Reparameterization Mode
            >>> model = QAC(64, 64, 'reparameterization')
            >>> inputs = torch.randn(4, 64)
            >>> actor_outputs = model(inputs,'compute_actor')
            >>> actor_outputs['logit'][0].shape # mu
            >>> torch.Size([4, 64])
            >>> actor_outputs['logit'][1].shape # sigma
            >>> torch.Size([4, 64])
        """
        x = self.actor(inputs)
        if self.actor_head_type == 'regression':
            return {'action': x['pred']}
        elif self.actor_head_type == 'reparameterization':
            return {'logit': [x['mu'], x['sigma']]}

    def compute_critic(self, inputs: Dict) -> Dict:
        r"""
        Overview:
            Execute parameter updates with ``'compute_critic'`` mode
            Use encoded embedding tensor to predict output.
        Arguments:
            - ``obs``, ``action`` encoded tensors.
            - mode (:obj:`str`): Name of the forward mode.
        Returns:
            - outputs (:obj:`Dict`): Q-value output.

        ReturnKeys:
            - q_value (:obj:`torch.Tensor`): Q value tensor with same size as batch size.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N1)`, where B is batch size and N1 is ``obs_shape``
            - action (:obj:`torch.Tensor`): :math:`(B, N2)`, where B is batch size and N2 is ``action_shape``
            - q_value (:obj:`torch.FloatTensor`): :math:`(B, )`, where B is batch size.

        Examples:
            >>> inputs = {'obs': torch.randn(4, N), 'action': torch.randn(4, 1)}
            >>> model = QAC(obs_shape=(N, ),action_shape=1,actor_head_type='regression')
            >>> model(inputs, mode='compute_critic')['q_value'] # q value
            tensor([0.0773, 0.1639, 0.0917, 0.0370], grad_fn=<SqueezeBackward1>)

        """

        obs, action = inputs['obs'], inputs['action']
        assert len(obs.shape) == 2
        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=1)
        x = self.critic(x)['pred']
        return {'q_value': x}

def identity(x):
    return x

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.fcs = []
        self.batch_norms = []

        # data normalization
        self.input_mu = nn.Parameter(ptu.zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(ptu.ones(input_size), requires_grad=False).float()

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight, w_scale)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.batch_norm:
                bn = nn.BatchNorm1d(next_size)
                self.__setattr__('batch_norm%d' % i, bn)
                self.batch_norms.append(bn)

            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            ptu.orthogonal_init(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = (input - self.input_mu) / (self.input_std + 1e-6)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x, num_q):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        # return x @ self.W + self.b
        return x @ self.W[:num_q] + self.b[:num_q]


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]
        self.num_q = ensemble_size

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []

        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                ptu.orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, inputs):
        # flat_inputs = torch.cat(inputs, dim=-1)
        flat_inputs = inputs
        state_dim = inputs[0].shape[-1]
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.num_q, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h, self.num_q)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h, self.num_q)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        return {'pred': output.squeeze(-1)}

    def sample(self, *inputs, **kwargs):
        preds = self.forward(*inputs, **kwargs)
        
        return torch.min(preds, dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError



# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -5

# class TanhGaussianPolicy(Mlp):
#     def __init__(
#             self,
#             hidden_sizes,
#             obs_dim,
#             action_dim,
#             std=None,
#             init_w=1e-3,
#             restrict_obs_dim=0,
#             **kwargs
#     ):
#         super().__init__(
#             hidden_sizes,
#             input_size=obs_dim,
#             output_size=action_dim,
#             init_w=init_w,
#             **kwargs
#         )
#         self.log_std = None
#         self.std = std
#         self.restrict_obs_dim = restrict_obs_dim

#         if std is None:
#             last_hidden_size = obs_dim
#             if len(hidden_sizes) > 0:
#                 last_hidden_size = hidden_sizes[-1]
#             self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
#             self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
#             self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
#         else:
#             init_logstd = ptu.ones(1, action_dim) * np.log(std)
#             self.log_std = torch.nn.Parameter(init_logstd, requires_grad=True)

#     def forward(
#             self,
#             obs,
#     ):
#         """
#         :param obs: Observation
#         :param deterministic: If True, do not sample
#         :param return_log_prob: If True, return a sample and its log probability
#         """
#         if len(obs.shape) == 1:
#             obs = obs[self.restrict_obs_dim:]
#         else:
#             obs = obs[:,self.restrict_obs_dim:]

#         h = obs
#         for i, fc in enumerate(self.fcs):
#             h = self.hidden_activation(fc(h))
#             if getattr(self, 'layer_norm', False) and (self.layer_norm is not None):
#                 h = self.layer_norm(h)
#         mean = self.last_fc(h)
#         if self.std is None:
#             log_std = self.last_fc_log_std(h)
#             log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#             std = torch.exp(log_std)
#         else:
#             log_std = self.log_std * ptu.ones(*mean.shape)
#             std = log_std.exp()
#         return  {'mu': mean, 'sigma': std}



# from torch.distributions import Distribution as TorchDistribution
# from torch.distributions import Normal as TorchNormal


# def atanh(x):
#     one_plus_x = (1 + x).clamp(min=1e-6)
#     one_minus_x = (1 - x).clamp(min=1e-6)
#     return 0.5*torch.log(one_plus_x/ one_minus_x)


# class Distribution(TorchDistribution):

#     def sample_and_logprob(self):
#         s = self.sample()
#         log_p = self.log_prob(s)
#         return s, log_p

#     def rsample_and_logprob(self):
#         s = self.rsample()
#         log_p = self.log_prob(s)
#         return s, log_p

#     def mle_estimate(self):
#         return self.mean

#     def get_diagnostics(self):
#         return {}

# class TanhNormal(Distribution):
#     """
#     Represent distribution of X where
#         X ~ tanh(Z)
#         Z ~ N(mean, std)

#     Note: this is not very numerically stable.
#     """
#     def __init__(self, normal_mean, normal_std, epsilon=1e-6):
#         """
#         :param normal_mean: Mean of the normal distribution
#         :param normal_std: Std of the normal distribution
#         :param epsilon: Numerical stability epsilon when computing log-prob.
#         """
#         self.normal_mean = normal_mean
#         self.normal_std = normal_std
#         self.normal = TorchNormal(normal_mean, normal_std)
#         self.epsilon = epsilon

#     def sample_n(self, n, return_pre_tanh_value=False):
#         z = self.normal.sample_n(n)
#         if return_pre_tanh_value:
#             return torch.tanh(z), z
#         else:
#             return torch.tanh(z)

#     def log_prob(self, value, pre_tanh_value=None):
#         """

#         :param value: some value, x
#         :param pre_tanh_value: arctanh(x)
#         :return:
#         """
#         if pre_tanh_value is None:
#             pre_tanh_value = atanh(value)
            
#         return self.normal.log_prob(pre_tanh_value) - torch.log(
#             1 - value * value + self.epsilon
#         )

#     def sample(self, return_pretanh_value=False):
#         """
#         Gradients will and should *not* pass through this operation.

#         See https://github.com/pytorch/pytorch/issues/4620 for discussion.
#         """
#         z = self.normal.sample().detach()

#         if return_pretanh_value:
#             return torch.tanh(z), z
#         else:
#             return torch.tanh(z)

#     def rsample(self, return_pretanh_value=False):
#         """
#         Sampling in the reparameterization case.
#         """
#         z = (
#             self.normal_mean +
#             self.normal_std *
#             TorchNormal(
#                 ptu.zeros(self.normal_mean.size()),
#                 ptu.ones(self.normal_std.size())
#             ).sample()
#         )
#         z.requires_grad_()

#         if return_pretanh_value:
#             return torch.tanh(z), z
#         else:
#             return torch.tanh(z)
