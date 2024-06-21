from re import DEBUG
from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.optim import SGD
import math
from ding.torch_utils import Adam, to_device, MedianFinder
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .dqn import DQNPolicy
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('cql_adaloss')
class CQLAdaLossPolicy(Policy):
    r"""
       Overview:
           Policy class of CQL algorithm.
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
        cuda=False,
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
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,

            # (bool type) value_network: Determine whether to use value network as the
            # original SAC paper (arXiv 1801.01290).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            # value_network=False,
            actor_head_type='reparameterization',
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
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
            
            num_actions=10,
        ),
        collect=dict(
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
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
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

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
        self._num_actions = self._cfg.learn.num_actions
        self._forward_learn_cnt = 0  # count iterations

        self.min_q_version = 3
        self.min_q_weight = self._cfg.learn.min_q_weight
        self.with_lagrange = self._cfg.learn.with_lagrange
        self.lagrange_thresh = self._cfg.learn.lagrange_thresh
        if self.with_lagrange:
            self.target_action_gap = self.lagrange_thresh
            self.log_alpha_prime = torch.tensor(0.).to(self._device).requires_grad_()
            self.alpha_prime_optimizer = Adam(
                [self.log_alpha_prime],
                lr=self._cfg.learn.learning_rate_q,
            )


        # Weight Init
        init_w = self._cfg.learn.init_w
        self._model.actor[2].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].mu.bias.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        self._model.actor[2].log_sigma_layer.bias.data.uniform_(-init_w, init_w)
        if self._cfg.learn.critic_init:
            if self._twin_critic:
                self._model.critic[0][2].last.weight.data.uniform_(-init_w, init_w)
                self._model.critic[0][2].last.bias.data.uniform_(-init_w, init_w)
                self._model.critic[1][2].last.weight.data.uniform_(-init_w, init_w)
                self._model.critic[1][2].last.bias.data.uniform_(-init_w, init_w)
            else:
                self._model.critic[2].last.weight.data.uniform_(-init_w, init_w)
                self._model.critic[2].last.bias.data.uniform_(-init_w, init_w)

        # Optimizers
        if self._value_network:
            self._optimizer_value = Adam(
                self._model.value_critic.parameters(),
                lr=self._cfg.learn.learning_rate_value,
            )
        if self._cfg.learn.optimizer.type=='adam':
            self._optimizer_q = Adam(
                self._model.critic.parameters(),
                lr=self._cfg.learn.learning_rate_q,
                weight_decay=self._cfg.learn.optimizer.weight_decay,
                grad_clip_type=self._cfg.learn.optimizer.get('grad_clip_type', None),
                clip_value=self._cfg.learn.optimizer.get('clip_value', None),
            )
            self._optimizer_policy = Adam(
                self._model.actor.parameters(),
                lr=self._cfg.learn.learning_rate_policy,
                weight_decay=self._cfg.learn.optimizer.weight_decay,
                grad_clip_type=self._cfg.learn.optimizer.get('grad_clip_type', None),
                clip_value=self._cfg.learn.optimizer.get('clip_value', None),
            )
        elif self._cfg.learn.optimizer.type=='sgd':
            self._optimizer_q = SGD(
                self._model.critic.parameters(),
                lr=self._cfg.learn.learning_rate_q,
                momentum=self._cfg.learn.optimizer.momentum,
                weight_decay=self._cfg.learn.optimizer.weight_decay
            )
            self._optimizer_policy = SGD(
                self._model.actor.parameters(),
                lr=self._cfg.learn.learning_rate_policy,
                momentum=self._cfg.learn.optimizer.momentum,
                weight_decay=self._cfg.learn.optimizer.weight_decay
            )
        if self._cfg.learn.lr_scheduler.flag==True:
            if self._cfg.learn.lr_scheduler.type=='Cosine':
                self._lr_scheduler_q = CosineAnnealingLR(self._optimizer_q, T_max=self._cfg.learn.lr_scheduler.T_max, eta_min=self._cfg.learn.learning_rate_q*0.01)
                self._lr_scheduler_policy = CosineAnnealingLR(self._optimizer_policy, T_max=self._cfg.learn.lr_scheduler.T_max, eta_min=self._cfg.learn.learning_rate_policy*0.01)
            elif self._cfg.learn.lr_scheduler.type=='MultiStep':
                self._lr_scheduler_q = MultiStepLR(self._optimizer_q, milestones=self._cfg.learn.lr_scheduler.milestones, gamma=self._cfg.learn.lr_scheduler.gamma)
                self._lr_scheduler_policy = MultiStepLR(self._optimizer_policy, milestones=self._cfg.learn.lr_scheduler.milestones, gamma=self._cfg.learn.lr_scheduler.gamma)
        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor
        # Init auto alpha
        if self._cfg.learn.auto_alpha:
            self._target_entropy = -np.prod(self._cfg.model.action_shape)
            self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
            self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
            self._auto_alpha = True
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()
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
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0
        true_q_list,_=self._get_dataset_meta_info()
        self.true_q_mean=np.mean(true_q_list)
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='mean_scale':
            self._MedianFinder=MedianFinder(buffer_size=self._cfg.learn.min_q_weight_td_error_buffer_size)
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='learned':
            self.min_q_weight_meta=torch.tensor(
                [self._cfg.learn.min_q_weight_td_error_learned_base], requires_grad=False, device=self._device, dtype=torch.float32
            )            
            true_q_list,_=self._get_dataset_meta_info()
            if self._cfg.learn.get('min_q_weight_td_error_learned_true_q_mult',None):
                self.true_q=np.mean(true_q_list)*self._cfg.learn.min_q_weight_td_error_learned_true_q_mult
            else:
                self.true_q=np.percentile(true_q_list, self._cfg.learn.min_q_weight_td_error_learned_percentile)
            if self._cfg.learn.get('min_q_weight_true_q_change',None):
                self.true_q_list=true_q_list
                self.true_q_begin=np.percentile(self.true_q_list, self._cfg.learn.get('min_q_weight_true_q_change_begin'))
                self.true_q_end=np.percentile(self.true_q_list, self._cfg.learn.get('min_q_weight_true_q_change_end'))
            
    def _get_dataset_meta_info(self):
        import d4rl
        import gym
        import math
        env = gym.make(self._cfg.learn.min_q_weight_td_error_true_q_env_id)
        if self._cfg.learn.get('min_q_weight_true_q_per_sample',None):
            dataset, _, _ = d4rl.qlearning_dataset_with_q(env, gamma=self._cfg.learn.discount_factor)
            true_q_list=dataset['q_values']
            return true_q_list,0
        else:
            dataset = d4rl.sequence_dataset(env)
            true_q_list=[]
            true_episode_reward_list=[]
            episode_length=[]
            for episode in dataset:
                rewards=episode['rewards']
                weights = np.power(self._gamma, list(range(len(rewards))))
                episode_length.append(len(rewards))
                true_q_list.append(np.sum(rewards*weights))  
                true_episode_reward_list.append(np.sum(rewards))              
            return true_q_list,true_episode_reward_list



    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        loss_dict = {}
        ada_loss_q_value=default_collate(data).get('q_value')
        if self._cuda:
            ada_loss_q_value = to_device(ada_loss_q_value, self._device)
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)
        if self._cfg.learn.lr_scheduler.flag==True:
            if self._lr_scheduler_q.last_epoch<=self._cfg.learn.lr_scheduler.T_max:
                self._lr_scheduler_q.step()
                self._lr_scheduler_policy.step()
            else:
                self._lr_scheduler_q.last_epoch+=1
                self._lr_scheduler_policy.last_epoch+=1
        # if self._cfg.learn.get('min_q_weight_cos', None):
        #     self.min_q_weight = self._cfg.learn.min_q_weight*(self._optimizer_q.param_groups[0]['lr']/self._cfg.learn.learning_rate_q)
        self._learn_model.train()
        self._target_model.train()
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        done = data.get('done')
        # predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # predict target value depend self._value_network.
        if self._value_network:
            # predict v value
            v_value = self._learn_model.forward(obs, mode='compute_value_critic')['v_value']
            with torch.no_grad():
                next_v_value = self._target_model.forward(next_obs, mode='compute_value_critic')['v_value']
        else:
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

                dist = Independent(Normal(mu, sigma), 1)
                pred = dist.rsample()
                next_action = torch.tanh(pred)
                y = 1 - next_action.pow(2) + 1e-6               
                next_log_prob = dist.log_prob(pred).unsqueeze(-1)
                next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)
                next_data = {'obs': next_obs, 'action': next_action}
                target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
                target_q_value_0=target_q_value
                # the value of a policy according to the maximum entropy objective
                if self._twin_critic:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],  target_q_value[1])
                    target_q_value_0 = target_q_value[0]

        target_value = next_v_value if self._value_network else target_q_value
        # just for print target_v
        if done is not None:
            target_v = self._gamma * (1 - done) * target_value + reward
        else:
            target_v = self._gamma * target_value + reward
        # =================
        # q network
        # =================
        # compute q loss
        if data['weight'] is None:
            data['weight'] = torch.ones_like(reward)
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_value, reward, done, data['weight'])
            _, td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            loss_dict['critic_loss'] = td_error_per_sample0 * data['weight']
            q_data1 = v_1step_td_data(q_value[1], target_value, reward, done, data['weight'])
            _, td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            loss_dict['twin_critic_loss'] = td_error_per_sample1 * data['weight']
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_value, reward, done, data['weight'])
            _, td_error_per_sample = v_1step_td_error(q_data, self._gamma)
            loss_dict['critic_loss'] = td_error_per_sample * data['weight']

        # add CQL
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(data, self._num_actions)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions({'obs': next_obs}, self._num_actions)
        
        # random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1) # .cuda()
        random_actions_tensor = torch.FloatTensor(curr_actions_tensor.shape).uniform_(-1, 1).to(curr_actions_tensor.device)

        obs_repeat = obs.unsqueeze(1).repeat(1, self._num_actions, 1).view(obs.shape[0] *
                                                                           self._num_actions, obs.shape[1])
        act_repeat = data['action'].unsqueeze(1).repeat(1, self._num_actions, 1).view(data['action'].shape[0] *
                                                                                      self._num_actions,
                                                                                      data['action'].shape[1])
        # q_pred = self._get_q_value({'obs': obs_repeat, 'action': act_repeat})
        # q_pred = self._get_q_value({'obs': obs, 'action': data['action']})
        q_rand = self._get_q_value({'obs': obs_repeat, 'action': random_actions_tensor})
        # q2_rand = self._get_q_value(obs, random_actions_tensor, network=self.qf2)
        q_curr_actions = self._get_q_value({'obs': obs_repeat, 'action': curr_actions_tensor})
        # q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q_next_actions = self._get_q_value({'obs': obs_repeat, 'action': new_curr_actions_tensor})
        # q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)
        
        cat_q1 = torch.cat(
            [q_rand[0], q_value[0].view(-1,1,1), q_next_actions[0], q_curr_actions[0]], 1
        )
        cat_q2 = torch.cat(
            [q_rand[1], q_value[1].view(-1,1,1), q_next_actions[1], q_curr_actions[1]], 1
        )
        std_q1 = torch.std(cat_q1)
        # std_q1 = torch.std(cat_q1, dim=1)
        # std_q2 = torch.std(cat_q2, dim=1)
        if self.min_q_version == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            # cat_q1.shape=(256, 30, 1)
            cat_q1 = torch.cat(
                    [q_rand[0] - random_density, q_next_actions[0] - new_log_pis.detach(), q_curr_actions[0] - curr_log_pis.detach()], 1
                )
            cat_q2 = torch.cat(
                    [q_rand[1] - random_density, q_next_actions[1] - new_log_pis.detach(), q_curr_actions[1] - curr_log_pis.detach()], 1
                )
        if self._cfg.learn.get('min_q_weight_td_error', None):
            if self._cfg.learn.min_q_weight_td_error_type=='identity':
                self.min_q_weight = td_error_per_sample.detach().mean().item()
                if self._cfg.learn.get('min_q_weight_td_error_ratio', None):
                    self.min_q_weight=self.min_q_weight*self._cfg.learn.min_q_weight_td_error_ratio               
            elif self._cfg.learn.min_q_weight_td_error_type=='sqrt':
                self.min_q_weight = math.sqrt(td_error_per_sample.detach().mean().item())
            elif self._cfg.learn.min_q_weight_td_error_type=='q_value':
                self.min_q_weight = q_value[0].detach().mean().item()
            elif self._cfg.learn.min_q_weight_td_error_type=='q_value_inverse':
                self.min_q_weight = 1
            elif self._cfg.learn.min_q_weight_td_error_type=='q_value_square':
                self.min_q_weight = (q_value[0].detach().mean().item())**2
                if self._cfg.learn.get('min_q_weight_td_error_ratio', None):
                    self.min_q_weight=self.min_q_weight*self._cfg.learn.min_q_weight_td_error_ratio                   
            elif self._cfg.learn.min_q_weight_td_error_type=='mean_scale':
                self.min_q_weight=td_error_per_sample.detach().mean().item()
                self.min_q_weight=self.min_q_weight/self._MedianFinder.updateMedian(self.min_q_weight)*self._cfg.learn.min_q_weight_td_error_scale
            elif self._cfg.learn.min_q_weight_td_error_type=='percentage':
                self.min_q_weight=1
            elif self._cfg.learn.min_q_weight_td_error_type=='learned':
                if self._cfg.learn.get('min_q_weight_true_q_change',None):
                    if self._cfg.learn.get('min_q_weight_true_q_change_pureq',None):
                        iter=self._cfg.learn.get('min_q_weight_true_q_change_iter')
                        self.true_q=min([(self._forward_learn_cnt/iter)*(self.true_q_end-self.true_q_begin)+self.true_q_begin,self.true_q_end])
                        percentile=0
                    else:
                        begin=self._cfg.learn.get('min_q_weight_true_q_change_begin')
                        end=self._cfg.learn.get('min_q_weight_true_q_change_end')
                        iter=self._cfg.learn.get('min_q_weight_true_q_change_iter')
                        percentile=min([int((self._forward_learn_cnt/iter)*(end-begin)+begin),end])
                        self.true_q=np.percentile(self.true_q_list, percentile)
                flag=True
                if self._cfg.learn.get('min_q_weight_td_error_learned_bar_ratio',None):
                    if abs(q_value[0].detach().mean().item()-self.true_q)<=abs(self.true_q*self._cfg.learn.min_q_weight_td_error_learned_bar_ratio):
                        flag=False
                if flag:     
                    if self._cfg.learn.get('min_q_weight_td_error_learned_version',None)=='linear':
                        self.min_q_weight_meta+=self._cfg.learn.min_q_weight_td_error_learned_rate*np.sign(q_value[0].detach().mean().item()-self.true_q)
                    elif self._cfg.learn.get('min_q_weight_td_error_learned_version',None)=='abs':
                        self.min_q_weight_meta[0]=q_value[0].detach().mean().item()-self.true_q
                    else:
                        self.min_q_weight_meta+=self._cfg.learn.min_q_weight_td_error_learned_rate*(q_value[0].detach().mean().item()-self.true_q)
                    meta_min=0.5
                    if self._cfg.learn.get('min_q_weight_td_error_learned_min', None):
                        meta_min=self._cfg.learn.min_q_weight_td_error_learned_min
                    self.min_q_weight_meta=torch.clamp(self.min_q_weight_meta, min=meta_min)
                    self.min_q_weight=self.min_q_weight_meta[0]
            else:
                raise NotImplementedError            
        min_qf1_loss = (torch.logsumexp(cat_q1, dim=1,) * self.min_q_weight).squeeze()
        min_qf2_loss = (torch.logsumexp(cat_q2, dim=1,) * self.min_q_weight).squeeze()
        
        """Subtract the log likelihood of data"""
        iod = self._cfg.learn.get('in_dataset_penalty_ratio', 1)
        if self._cfg.learn.get('min_q_weight_on_all_penalty',None):
            notin_ratio = self._cfg.learn.get('min_q_weight_on_all_penalty_notin_ratio', 1)
            min_qf1_loss = min_qf1_loss - iod*q_value[0]*notin_ratio
            min_qf2_loss = min_qf2_loss - iod*q_value[1]*notin_ratio
        else:
            min_qf1_loss = min_qf1_loss - iod*q_value[0] * self.min_q_weight
            min_qf2_loss = min_qf2_loss - iod*q_value[1] * self.min_q_weight
        
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss.mean() - min_qf2_loss.mean())*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        if self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_inverse':
            beta = torch.clamp(self._cfg.learn.get('adaptive_loss_q_value_inverse_ratio', 1)/(torch.clamp(ada_loss_q_value,min=1e-8)),min=0.0,max=10)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_mean_inverse':
            beta = torch.clamp(self._cfg.learn.get('adaptive_loss_q_value_inverse_ratio', 1)/(torch.clamp(ada_loss_q_value.mean(),min=1e-8)),min=0.0,max=10)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_pred_inverse':
            beta = torch.clamp(self._cfg.learn.get('adaptive_loss_q_value_inverse_ratio', 1)/(torch.clamp(q_value[0].detach(),min=1e-8)),min=0.0,max=10)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_pred_mean_inverse':
            beta = torch.clamp(self._cfg.learn.get('adaptive_loss_q_value_inverse_ratio', 1)/(torch.clamp(q_value[0].mean().detach(),min=1e-8)),min=0.0,max=10)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_sigmoid':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                tem=600
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                tem=400
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                tem=275
            beta = 1 - F.sigmoid(ada_loss_q_value-tem)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_linear':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta = torch.clamp((1000 - ada_loss_q_value)/300,min=0.0)
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((450 - ada_loss_q_value)/200,min=0.0)
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((350 - ada_loss_q_value)/140,min=0.0)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_linear_v1':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta = torch.clamp((1000 - ada_loss_q_value)/40,min=0.0)
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((450 - ada_loss_q_value)/18,min=0.0)
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((350 - ada_loss_q_value)/14,min=0.0)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_linear_v2':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta = torch.clamp((1000 - ada_loss_q_value)/20,min=0.0)
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((450 - ada_loss_q_value)/9,min=0.0)
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((350 - ada_loss_q_value)/7,min=0.0)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_linear_v3':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta = torch.clamp((1000 - ada_loss_q_value)/100,min=0.0)
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((450 - ada_loss_q_value)/45,min=0.0)
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                beta=torch.clamp((350 - ada_loss_q_value)/35,min=0.0)
        elif self._cfg.learn.get('adaptive_rl_loss_version', None)=='q_value_normalization':
            if 'halfcheetah' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                interval=(-50,1200)
            elif 'walker' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                interval=(-5,600)
            elif 'hopper' in self._cfg.learn.min_q_weight_td_error_true_q_env_id:
                interval=(-5,400)
            ada_loss_q_value=(ada_loss_q_value-interval[0])/(interval[1]-interval[0])
            ada_loss_q_value=torch.clamp(ada_loss_q_value,min=0.0,max=1.0)
            beta=(1-ada_loss_q_value)**self._cfg.learn.adaptive_rl_loss_q_value_normalization_exponent
            beta=beta*self._cfg.learn.adaptive_rl_loss_q_value_normalization_ratio
        else:
            beta=1
        loss_dict['critic_loss'] = loss_dict['critic_loss']*beta + min_qf1_loss
        if self._twin_critic:
            loss_dict['twin_critic_loss'] = loss_dict['twin_critic_loss']*beta + min_qf2_loss
        
        # adaptive loss by Q value
        loss_weight = torch.clamp(ada_loss_q_value/self.true_q_mean,0)
        if self._cfg.learn.get('adaptive_loss_version', None)=='relu':
            loss_weight=torch.where(loss_weight > 1, torch.ones_like(loss_weight), torch.zeros_like(loss_weight))
        elif self._cfg.learn.get('adaptive_loss_version', None)=='one':
            loss_weight = torch.ones_like(loss_weight)
        else:
            loss_weight_ratio=self._cfg.learn.get("ada_loss_q_value_loss_weight_ratio", 1)
            loss_weight_max=loss_weight*loss_weight_ratio
            loss_weight_min=loss_weight/loss_weight_ratio
            loss_weight=torch.where(loss_weight > 1, loss_weight_max, loss_weight)
            loss_weight=torch.where(loss_weight < 1, loss_weight_min, loss_weight)
        loss_dict['critic_loss']=(loss_dict['critic_loss']*loss_weight).mean()
        # update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward(retain_graph=True)
        if self._twin_critic:
            loss_dict['twin_critic_loss']=(loss_dict['twin_critic_loss']*loss_weight).mean()
            loss_dict['twin_critic_loss'].backward()
        self._optimizer_q.step()

        # evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # =================
        # value network
        # =================
        # compute value loss
        if self._value_network:
            # new_q_value: (bs, ), log_prob: (bs, act_shape) -> target_v_value: (bs, )
            target_v_value = (new_q_value.unsqueeze(-1)).mean(dim=-1)
            loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

            # update value network
            self._optimizer_value.zero_grad()
            loss_dict['value_loss'].backward()
            self._optimizer_value.step()

        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # compute alpha loss
        if self._auto_alpha:
            log_prob = log_prob.detach() + self._target_entropy
            loss_dict['alpha_loss'] = -(self._log_alpha * log_prob).mean()

            self._alpha_optim.zero_grad()
            loss_dict['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        loss_dict['total_loss'] = sum(loss_dict.values())

        info_dict = {}
        if self._value_network:
            info_dict['cur_lr_v'] = self._optimizer_value.defaults['lr']
            

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        if self._forward_learn_cnt>1100000:
            import sys
            sys.exit(0)
        # target update
        self._target_model.update(self._learn_model.state_dict())
        ret={
            'cur_lr_q': self._optimizer_q.param_groups[0]['lr'],
            'cur_lr_p': self._optimizer_policy.param_groups[0]['lr'],
            'q_fc_norm': self._get_fc_weight_norm(self._model.critic),
            'policy_fc_norm': self._get_fc_weight_norm(self._model.actor),
            'min_q_weight': float(self.min_q_weight),
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_value': target_value.detach().mean().item(),
            'target_v': target_v.detach().mean().item(),
            'target_q_value_0':target_q_value_0.detach().mean().item(),
            'target_q_value_0_std':target_q_value_0.detach().std().item(),
            'q_value_0': q_value[0].detach().mean().item(),
            'q_value_0_std': q_value[0].detach().std().item(),
            'q_value_1': q_value[1].detach().mean().item(),
            'min_qf1_loss': min_qf1_loss.detach().mean().item(),
            'min_qf2_loss': min_qf2_loss.detach().mean().item(),
            'std_q1': std_q1.detach().mean().item(),
            **info_dict,
            **loss_dict
        }
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='mean_scale':
            ret['td_error_median']=self._MedianFinder.getMedian()
        elif self._cfg.learn.get('min_q_weight_td_error_type',None)=='learned':
            ret['true_q']=self.true_q
        if self._cfg.learn.get('min_q_weight_true_q_change',None):
            ret['percentile']=float(percentile)
        # print(ret['td_error_median'])
        return ret
    
    def _get_fc_weight_norm(self, net):
        with torch.no_grad():
            return torch.sqrt(sum([torch.sum(m.weight.clone()**2) for m in net.modules() if isinstance(m, torch.nn.Linear)]))
            
    def _state_dict_learn(self) -> Dict[str, Any]:
        if self._cfg.learn.lr_scheduler.flag==True:
            ret = {
                'model': self._learn_model.state_dict(),
                'target_model': self._target_model.state_dict(),
                'optimizer_q': self._optimizer_q.state_dict(),
                'optimizer_policy': self._optimizer_policy.state_dict(),
                'lr_scheduler_q': self._lr_scheduler_q.state_dict(),
                'lr_scheduler_policy': self._lr_scheduler_policy.state_dict(),
                'forward_learn_cnt':self._forward_learn_cnt,
            }
        else:
            ret = {
                'model': self._learn_model.state_dict(),
                'target_model': self._target_model.state_dict(),
                'optimizer_q': self._optimizer_q.state_dict(),
                'optimizer_policy': self._optimizer_policy.state_dict(),
                'forward_learn_cnt':self._forward_learn_cnt,
            }
        if self._value_network:
            ret.update({'optimizer_value': self._optimizer_value.state_dict()})
        if self._auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='mean_scale':
            ret.update({'median_buffer': self._MedianFinder.dump()})
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='learned':
            ret.update({'min_q_weight_meta':self.min_q_weight_meta})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        self._forward_learn_cnt=state_dict['forward_learn_cnt']
        if self._cfg.learn.lr_scheduler.flag==True:
            self._lr_scheduler_q.load_state_dict(state_dict['lr_scheduler_q'])
            self._lr_scheduler_policy.load_state_dict(state_dict['lr_scheduler_policy'])
        
        if self._value_network:
            self._optimizer_value.load_state_dict(state_dict['optimizer_value'])
        if self._auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='mean_scale':
            self._MedianFinder.load(state_dict['median_buffer'])
        if self._cfg.learn.get('min_q_weight_td_error_type',None)=='learned':
            self.min_q_weight_meta.data=state_dict['min_q_weight_meta']

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        # TODO remove noise
        # self._collect_model = model_wrap(
        #     self._model,
        #     wrapper_name='action_noise',
        #     noise_type='gauss',
        #     noise_kwargs={
        #         'mu': 0.0,
        #         'sigma': self._cfg.collect.noise_sigma
        #     },
        #     noise_range=None
        # )
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
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

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
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
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
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
        return 'qac', ['ding.model.template.qac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        if self._auto_alpha:
            return super()._monitor_vars_learn() + [
                'alpha_loss', 'policy_loss', 'critic_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 
                'alpha', 'td_error', 'target_value', 'target_v', 'q_value_0', 'q_value_1', 'min_qf1_loss', 'min_qf2_loss'
            ] + twin_critic
        else:
            return super()._monitor_vars_learn() + [
                'policy_loss', 'critic_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 'q_fc_norm', 'policy_fc_norm', 'min_q_weight', 'td_error_median','true_q','percentile',
                'alpha', 'td_error', 'target_value', 'target_v', 'q_value_0', 'q_value_1', 'min_qf1_loss', 'min_qf2_loss','q_value_0_std','std_q1'
            ] + twin_critic

    def _get_policy_actions(self, data: Dict, num_actions=10, epsilon: float=1e-6) -> List:
        
        # evaluate to get action distribution
        obs = data['obs']
        obs = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        
        # evaluate action log prob depending on Jacobi determinant.
        y = 1 - action.pow(2) + epsilon
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)
        
        return action, log_prob.view(-1, num_actions, 1)
    
    def _get_q_value(self, data: Dict, keep=True):
        new_q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = [value.view(-1, self._num_actions, 1) for value in new_q_value]
        else:
            new_q_value = new_q_value.view(-1, self._num_actions, 1)
        if self._twin_critic and not keep:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        return new_q_value