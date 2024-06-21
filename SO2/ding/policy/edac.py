from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from ding.rl_utils import create_noise_generator


@POLICY_REGISTRY.register('edac')
class EDACPolicy(Policy):
    r"""
       Overview:
           Policy class of SAC algorithm.

           https://arxiv.org/pdf/1801.01290.pdf

       Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other(Shape)
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         td3            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           4  | ``model.policy_``   int         256            | Linear layer size for policy    |
              | ``embedding_size``                             | network.                        |
           5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
              | ``embedding_size``                             | network.                        |
           6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
              | ``embedding_size``                             | network.                        | model.value_network
              |                                                |                                 | is False.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
              | ``_rate_q``                                    | network.                        | model.value_network
              |                                                |                                 | is True.
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
              | ``_rate_policy``                               | network.                        | model.value_network
              |                                                |                                 | is True.
           9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
              | ``_rate_value``                                | network.                        | model.value_network
              |                                                |                                 | is False.
           10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | `\alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.repara_``   bool        True           | Determine whether to use        |
              | ``meterization``                               | reparameterization trick.       |
           12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | `\alpha`.                       | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in halfcheetah env.
           14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
       """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sac',
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            actor_head_type='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            critic_ensemble_size=2,
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,

            # (float type) learning_rate_q: Learning rate for soft q network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_q=3e-4,
            # (float type) learning_rate_policy: Learning rate for policy network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_policy=3e-4,
            # (float type) learning_rate_value: Learning rate for value network.
            # `learning_rate_value` should be initialized, when model.value_network is True.
            # Please set to 3e-4, when model.value_network is True.
            learning_rate_value=3e-4,

            # (float type) learning_rate_alpha: Learning rate for auto temperature parameter `\alpha`.
            # Default to 3e-4.
            learning_rate_alpha=3e-4,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,

            # (float type) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,

            # (bool type) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            auto_alpha=True,
            # (bool type) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
        ),
        collect=dict(
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(
                # (int) Evaluate every "eval_freq" training iterations.
                eval_freq=5000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=1000000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_network = False  # TODO self._cfg.model.value_network
        self._twin_critic = self._cfg.model.twin_critic
        self._proxy_network = self._cfg.learn.get('proxy_network', None)
        self._cfg.learn.only_value = False

        if self._proxy_network:
            self._model.proxy_network = copy.deepcopy(self._model.critic)
    
        # Weight Init for the last output layer
        init_w = self._cfg.learn.init_w
        self._model.actor[2].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].mu.bias.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.bias.data.uniform_(-init_w, init_w)

        self.q_criterion = torch.nn.MSELoss(reduction='none')
        self._actor_update_freq = self._cfg.learn.get('actor_update_freq', 1)
        self._normalized_reward = self._cfg.learn.get('normalized_reward', False)
        self._reward_mean = self._cfg.learn.get('reward_mean', 0)
        self._reward_std = self._cfg.learn.get('reward_std', 1)
        # Optimizers
        if self._value_network:
            self._optimizer_value = Adam(
                self._model.value_critic.parameters(),
                lr=self._cfg.learn.learning_rate_value,
            )
        if self._proxy_network:
            self._optimizer_proxy = Adam(
                self._model.proxy_network.parameters(),
                lr=self._cfg.learn.learning_rate_q,
            )
        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
            weight_decay=self._cfg.learn.get('weight_decay', 0)
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor
        # Init auto alpha
        if self._cfg.learn.auto_alpha:
            self._target_entropy = self._cfg.learn.get('target_entropy', -np.prod(self._cfg.model.action_shape))
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
                assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = torch.FloatTensor([self._cfg.learn.alpha]).to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._alpha], lr=self._cfg.learn.learning_rate_alpha)
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha], requires_grad=False, device=self._device, dtype=torch.float32
            )
            self._auto_alpha = False

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._value_norm=self._cfg.learn.get('value_norm', False)
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)

        if self._cfg.learn.get('noise', None):
            self._noise_generator = create_noise_generator(noise_type='gauss', 
                                    noise_kwargs={
                                            'mu': 0.0,
                                            'sigma': self._cfg.learn.noise_sigma
                                        },)
            self.action_range = {'min': -1, 'max': 1}
        #     self._target_model = model_wrap(
        #         self._target_model,
        #         wrapper_name='action_noise',
        #         noise_type='gauss',
        #         noise_kwargs={
        #             'mu': 0.0,
        #             'sigma': self._cfg.learn.noise_sigma
        #         },
        #         noise_range=self._cfg.learn.noise_range
        #     )

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0


    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr, loss, target_q_value and other \
                running information.
        """
        loss_dict = {}
        # aug_data_len = data[0].pop('aug_data_len')
        # aug_data_min_dist = data[0].pop('aug_data_min_dist')
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs, action, reward, next_obs, done = data['obs'], data['action'], data['reward'], data['next_obs'], data['done']
        weight = data.get('weight', torch.ones_like(data['done']).float())
        if weight is None:
            weight =  torch.ones_like(data['done']).float()
        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        
        if self._value_norm:
            self._running_mean_std.update(torch.mean(q_value.detach(), dim=0).cpu().numpy())
            q_value = q_value / self._running_mean_std.std

        # 2. predict target value depend self._value_network.
        # target q value.
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            next_action = torch.tanh(pred)
            y = 1 - next_action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            next_log_prob = dist.log_prob(pred).unsqueeze(-1)
            next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)
            
            if self._cfg.learn.get('noise', None):
                noise = self._noise_generator(action.shape, action.device)
                if self._cfg.learn.noise_range is not None:
                    noise = noise.clamp(self._cfg.learn.noise_range['min'], 
                                        self._cfg.learn.noise_range['max'])
                next_action += noise
                if self.action_range is not None:
                    next_action = next_action.clamp(self.action_range['min'], 
                                                    self.action_range['max'])
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            # find min one as target q value
            target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)
            target_q_value = (1-done) * target_q_value

        if self._value_norm:
            self._running_mean_std.update(torch.mean(target_q_value.detach(), dim=0).cpu().numpy())
            target_q_value = target_q_value / self._running_mean_std.std


        # 3. compute q loss
        td_error_per_sample = self.q_criterion(q_value, reward + self._gamma * target_q_value)
        td_error_per_sample = td_error_per_sample.mean(0)
        loss_dict['critic_loss'] = (td_error_per_sample * weight).mean()
        # 4. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        self._optimizer_q.step()

        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0 and self._cfg.learn.online:
            # 5. evaluate to get action distribution
            (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)
            y = 1 - action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = dist.log_prob(pred).unsqueeze(-1)
            log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

            eval_data = {'obs': obs, 'action': action}
            new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
            new_q_value = torch.min(new_q_value, dim=0)[0]

            if self._value_norm:
                self._running_mean_std.update(new_q_value.detach().cpu().numpy())
                new_q_value = new_q_value / self._running_mean_std.std

            # 7. compute policy loss
            policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

            loss_dict['policy_loss'] = policy_loss

            # 8. update policy network
            if not self._cfg.learn.only_value:
                self._optimizer_policy.zero_grad()
                loss_dict['policy_loss'].backward()
                self._optimizer_policy.step()

        # 9. compute alpha loss
        if self._auto_alpha and not self._cfg.learn.only_value:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            # 'q_fc_norm': self._get_fc_weight_norm(self._model.critic),
            'policy_fc_norm': self._get_fc_weight_norm(self._model.actor),
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            'q_value': q_value.detach().mean().item(),
            'sigma': sigma.detach().mean().item(),
            # 'aug_data_len': float(aug_data_len),
            # 'aug_data_min_dist': float(aug_data_min_dist),
            **loss_dict
        }

    def _get_fc_weight_norm(self, net):
        with torch.no_grad():
            return torch.sqrt(sum([torch.sum(m.weight.clone()**2) for m in net.modules() if isinstance(m, torch.nn.Linear)]))


    def _compute_norm(self, tensor_list):
        sq_sum = 0
        for t in tensor_list:
            sq_sum += torch.sum(t ** 2)
        return torch.sqrt(sq_sum)

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._value_network:
            ret.update({'optimizer_value': self._optimizer_value.state_dict()})
        if self._auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        try:
            state_dict['policy']={}
            state_dict['policy']['0.weight'                ] = state_dict['trainer/policy']['fc0.weight'            ]
            state_dict['policy']['0.bias'                  ] = state_dict['trainer/policy']['fc0.bias'              ]
            state_dict['policy']['2.main.0.weight'         ] = state_dict['trainer/policy']['fc1.weight'            ]
            state_dict['policy']['2.main.0.bias'           ] = state_dict['trainer/policy']['fc1.bias'              ]
            state_dict['policy']['2.main.2.weight'         ] = state_dict['trainer/policy']['fc2.weight'            ]
            state_dict['policy']['2.main.2.bias'           ] = state_dict['trainer/policy']['fc2.bias'              ]
            state_dict['policy']['2.mu.weight'             ] = state_dict['trainer/policy']['last_fc.weight'        ]
            state_dict['policy']['2.mu.bias'               ] = state_dict['trainer/policy']['last_fc.bias'          ]
            state_dict['policy']['2.log_sigma_layer.weight'] = state_dict['trainer/policy']['last_fc_log_std.weight']
            state_dict['policy']['2.log_sigma_layer.bias'  ] = state_dict['trainer/policy']['last_fc_log_std.bias'  ]

            for key in state_dict['trainer/qfs']:
                state_dict['trainer/qfs'][key]=state_dict['trainer/qfs'][key][:self._cfg.model.critic_ensemble_size]
            for key in state_dict['trainer/target_qfs']:
                state_dict['trainer/target_qfs'][key]=state_dict['trainer/target_qfs'][key][:self._cfg.model.critic_ensemble_size]

            self._learn_model.actor.load_state_dict(state_dict['policy'])
            if not self._proxy_network:
                self._learn_model.critic.load_state_dict(state_dict['trainer/qfs'])
            self._target_model.actor.load_state_dict(state_dict['policy'])
            if not self._proxy_network:
                self._target_model.critic.load_state_dict(state_dict['trainer/target_qfs'])
        except Exception as e:
            self._learn_model.load_state_dict(state_dict['model'])
            self._target_model.load_state_dict(state_dict['target_model'])
        # self._log_alpha=state_dict['trainer/log_alpha']

        # self._optimizer_q.load_state_dict(state_dict['trainer/qfs_optim'])
        # self._optimizer_policy.load_state_dict(state_dict['trainer/policy_optim'])
        # if self._auto_alpha:
        #     self._alpha_optim.load_state_dict(state_dict['trainer/alpha_optim'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._collect_model.forward(data, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            action = torch.tanh(dist.rsample())
            output = {'logit': (mu, sigma), 'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - policy_output (:obj:`dict`): Output of policy collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        if self._normalized_reward:
            reward = (timestep.reward - self._reward_mean) / self._reward_std
        else:
            reward = timestep.reward
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu)  # deterministic_eval
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'ensemble_qac', ['ding.model.template.ensemble_qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        value_loss = ['value_loss'] if self._value_network else []
        return [
            'alpha_loss',
            'policy_loss',
            'critic_loss',
            'cur_lr_q',
            'cur_lr_p',
            'target_q_value',
            'alpha',
            'td_error',
            'q_fc_norm', 
            'policy_fc_norm',
            'aug_data_len',
            'aug_data_min_dist',
            'q_value',
            'sigma'
        ] + twin_critic + alpha_loss + value_loss
