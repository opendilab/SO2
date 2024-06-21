from io import IOBase
from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.optim import SGD

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn
from .ddpg import DDPGPolicy


@POLICY_REGISTRY.register('td3_bc')
class TD3BCPolicy(DDPGPolicy):
    r"""
    Overview:
        Policy class of TD3_BC algorithm.

        Since DDPG and TD3 share many common things, we can easily derive this TD3_BC
        class from DDPG class by changing ``_actor_update_freq``, ``_twin_critic`` and noise in model wrapper.

        https://arxiv.org/pdf/2106.06860.pdf

    Property:
        learn_mode, collect_mode, eval_mode

    Config:

    == ====================  ========    ==================  =================================   =======================
    ID Symbol                Type        Default Value       Description                         Other(Shape)
    == ====================  ========    ==================  =================================   =======================
    1  ``type``              str         td3_bc              | RL policy register name, refer    | this arg is optional,
                                                             | to registry ``POLICY_REGISTRY``   | a placeholder
    2  ``cuda``              bool        True                | Whether to use cuda for network   |
    3  | ``random_``         int         25000               | Number of randomly collected      | Default to 25000 for
       | ``collect_size``                                    | training samples in replay        | DDPG/TD3, 10000 for
       |                                                     | buffer when training starts.      | sac.
    4  | ``model.twin_``     bool        True                | Whether to use two critic         | Default True for TD3,
       | ``critic``                                          | networks or only one.             | Clipped Double
       |                                                     |                                   | Q-learning method in
       |                                                     |                                   | TD3 paper.
    5  | ``learn.learning``  float       1e-3                | Learning rate for actor           |
       | ``_rate_actor``                                     | network(aka. policy).             |
    6  | ``learn.learning``  float       1e-3                | Learning rates for critic         |
       | ``_rate_critic``                                    | network (aka. Q-network).         |
    7  | ``learn.actor_``    int         2                   | When critic network updates       | Default 2 for TD3, 1
       | ``update_freq``                                     | once, how many times will actor   | for DDPG. Delayed
       |                                                     | network update.                   | Policy Updates method
       |                                                     |                                   | in TD3 paper.
    8  | ``learn.noise``     bool        True                | Whether to add noise on target    | Default True for TD3,
       |                                                     | network's action.                 | False for DDPG.
       |                                                     |                                   | Target Policy Smoo-
       |                                                     |                                   | thing Regularization
       |                                                     |                                   | in TD3 paper.
    9  | ``learn.noise_``    dict        | dict(min=-0.5,    | Limit for range of target         |
       | ``range``                       |      max=0.5,)    | policy smoothing noise,           |
       |                                 |                   | aka. noise_clip.                  |
    10 | ``learn.-``         bool        False               | Determine whether to ignore       | Use ignore_done only
       | ``ignore_done``                                     | done flag.                        | in halfcheetah env.
    11 | ``learn.-``         float       0.005               | Used for soft update of the       | aka. Interpolation
       | ``target_theta``                                    | target network.                   | factor in polyak aver
       |                                                     |                                   | aging for target
       |                                                     |                                   | networks.
    12 | ``collect.-``       float       0.1                 | Used for add noise during co-     | Sample noise from dis
       | ``noise_sigma``                                     | llection, through controlling     | tribution, Ornstein-
       |                                                     | the sigma of distribution         | Uhlenbeck process in
       |                                                     |                                   | DDPG paper, Guassian
       |                                                     |                                   | process in ours.
    == ====================  ========    ==================  =================================   =======================
   """

    # You can refer to DDPG's default config for more details.
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='td3_bc',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in TD3.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default False in TD3.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        model=dict(
            # (bool) Whether to use two critic networks or only one.
            # Clipped Double Q-Learning for Actor-Critic in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            twin_critic=True,
        ),
        learn=dict(
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # (float) Learning rates for critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=2,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            noise=True,
            # (float) Sigma for smoothing noise added to target policy.
            noise_sigma=0.2,
            # (dict) Limit for range of target policy smoothing noise, aka. noise_clip.
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
            alpha=2.5,
        ),
        collect=dict(
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
            # (bool) Whether to normalize the features of every state in the provided dataset.
            normalize_states=True,
        ),
        eval=dict(
            evaluator=dict(
                # (int) Evaluate every "eval_freq" training iterations.
                eval_freq=5000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config.
        """
        super(TD3BCPolicy, self)._init_learn()
        self._alpha = self._cfg.learn.alpha
        self.lmbda = self._alpha
        # actor and critic optimizer
        if self._cfg.learn.get('optimizer', None) is None:
            self._optimizer_actor = Adam(
                self._model.actor.parameters(),
                lr=self._cfg.learn.learning_rate_actor,
                grad_clip_type='clip_norm',
                # weight_decay=self._cfg.learn.optimizer.weight_decay,
                clip_value=1.0,
            )
            self._optimizer_critic = Adam(
                self._model.critic.parameters(),
                lr=self._cfg.learn.learning_rate_critic,
                grad_clip_type='clip_norm',
                # weight_decay=self._cfg.learn.optimizer.weight_decay,
                clip_value=1.0,
            )
        elif self._cfg.learn.optimizer.type=='adam':
            self._optimizer_actor = Adam(
                self._model.actor.parameters(),
                lr=self._cfg.learn.learning_rate_actor,
                grad_clip_type='clip_norm',
                weight_decay=self._cfg.learn.optimizer.weight_decay,
                clip_value=1.0,
            )
            self._optimizer_critic = Adam(
                self._model.critic.parameters(),
                lr=self._cfg.learn.learning_rate_critic,
                grad_clip_type='clip_norm',
                weight_decay=self._cfg.learn.optimizer.weight_decay,
                clip_value=1.0,
            )
        elif self._cfg.learn.optimizer.type=='sgd':
            self._optimizer_actor = SGD(
                self._model.actor.parameters(),
                lr=self._cfg.learn.learning_rate_actor,
                momentum=self._cfg.learn.optimizer.momentum,
                weight_decay=self._cfg.learn.optimizer.weight_decay,
            )
            self._optimizer_critic = SGD(
                self._model.critic.parameters(),
                lr=self._cfg.learn.learning_rate_critic,
                momentum=self._cfg.learn.optimizer.momentum,
                weight_decay=self._cfg.learn.optimizer.weight_decay,
            )
        if self._cfg.learn.get('lr_scheduler', None) and self._cfg.learn.lr_scheduler.flag==True:
            if self._cfg.learn.lr_scheduler.type=='Cosine':
                self._lr_scheduler_critic = CosineAnnealingLR(self._optimizer_critic, T_max=self._cfg.learn.lr_scheduler.T_max, eta_min=self._cfg.learn.learning_rate_critic*0.01)
                self._lr_scheduler_actor = CosineAnnealingLR(self._optimizer_actor, T_max=self._cfg.learn.lr_scheduler.T_max, eta_min=self._cfg.learn.learning_rate_actor*0.01)
            elif self._cfg.learn.lr_scheduler.type=='MultiStep':
                self._lr_scheduler_critic = MultiStepLR(self._optimizer_critic, milestones=self._cfg.learn.lr_scheduler.milestones, gamma=self._cfg.learn.lr_scheduler.gamma)
                self._lr_scheduler_actor = MultiStepLR(self._optimizer_actor, milestones=self._cfg.learn.lr_scheduler.milestones, gamma=self._cfg.learn.lr_scheduler.gamma)
        
        if self._cfg.learn.get('lmbda_type', None)=='learned':
            self.lmbda_meta=torch.tensor(
                [self._cfg.learn.lmbda_learned_base], requires_grad=False, device=self._device, dtype=torch.float32
            )            
            true_q_list,_=self._get_dataset_meta_info()
            self.true_q=np.percentile(true_q_list, self._cfg.learn.lmbda_learned_percentile)
            if self._cfg.learn.get('min_q_weight_true_q_change',None):
                self.true_q_list=true_q_list
                self.true_q_begin=np.percentile(self.true_q_list, self._cfg.learn.get('min_q_weight_true_q_change_begin'))
                self.true_q_end=np.percentile(self.true_q_list, self._cfg.learn.get('min_q_weight_true_q_change_end'))

    # def _get_dataset_meta_info(self):
    #     import d4rl
    #     import gym
    #     import math
    #     env = gym.make(self._cfg.learn.lmbda_true_q_env_id)
    #     dataset = d4rl.sequence_dataset(env)
    #     true_q_list=[]
    #     true_episode_reward_list=[]
    #     episode_length=[]
    #     for episode in dataset:
    #         rewards=episode['rewards']
    #         weights = np.power(self._gamma, list(range(len(rewards))))
    #         episode_length.append(len(rewards))
    #         true_q_list.append(np.sum(rewards*weights))  
    #         true_episode_reward_list.append(np.sum(rewards))              
    #     return true_q_list,true_episode_reward_list   
    def _get_dataset_meta_info(self):
        from time import sleep,time
        import random
        for i in range(100):
            try:
                import d4rl  # register d4rl enviroments with open ai gym
            except Exception as e:
                random.seed(time())
                sleep(random.randint(1,120))
                print(e)
        import gym
        env = gym.make(self._cfg.learn.lmbda_true_q_env_id)
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
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # critic learn forward
        # ====================
        if self._cfg.learn.lr_scheduler.flag==True:
            if self._lr_scheduler_critic.last_epoch<=self._cfg.learn.lr_scheduler.T_max:
                self._lr_scheduler_critic.step()
                self._lr_scheduler_actor.step()
            else:
                self._lr_scheduler_critic.last_epoch+=1
                self._lr_scheduler_actor.last_epoch+=1
        self._learn_model.train()
        self._target_model.train()
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._use_reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        q_value_dict = {}
        if self._twin_critic:
            q_value_dict['q_value'] = q_value[0].mean()
            q_value_dict['q_value_twin'] = q_value[1].mean()
        else:
            q_value_dict['q_value'] = q_value.mean()
        # target q value.
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
        if self._cfg.learn.get('q_value_similarity_loss_weight_type',None):
            if self._cfg.learn.get('q_value_similarity_loss_weight_type',None) == 'mse':
                mse_loss = torch.nn.MSELoss(reduction='none')
                similarity = mse_loss(next_data['action'], data['next_action'])
                similarity = - similarity.mean(-1)
                similarity= (similarity-(-4))/4
            elif self._cfg.learn.get('q_value_similarity_loss_weight_type',None) == 'cosine':    
                similarity = torch.cosine_similarity(next_data['action'], data['next_action'])
            else:
                raise NotImplementedError
            similarity = similarity * self._cfg.learn.get('q_value_similarity_loss_weight_ratio',1)
            if self._cfg.learn.get('q_value_similarity_loss_weight_bound',None):
                similarity[similarity<self._cfg.learn.q_value_similarity_loss_weight_bound]=0
            if data['weight'] is None:
                data['weight'] = similarity
                data['weight'][data['done'].long()] = 1
        if self._twin_critic:
            # TD3: two critic networks
            target_q_value = torch.min(target_q_value[0], target_q_value[1])  # find min one as target q value
            # critic network1
            td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample1 = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
            # critic network2(twin network)
            td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
            critic_twin_loss, td_error_per_sample2 = v_1step_td_error(td_data_twin, self._gamma)
            loss_dict['critic_twin_loss'] = critic_twin_loss
            td_error_per_sample = (td_error_per_sample1 + td_error_per_sample2) / 2
        else:
            # DDPG: single critic network
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
            critic_loss, td_error_per_sample = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
        # ================
        # critic update
        # ================
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        # ===============================
        # actor learn forward and update
        # ===============================
        # actor updates every ``self._actor_update_freq`` iters
        if self._cfg.learn.get('similarity_loss_weight_type',None):
            similarity_policy=torch.zeros(1)
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            if self._twin_critic:
                q_value = self._learn_model.forward(actor_data, mode='compute_critic')['q_value'][0]
            else:
                q_value = self._learn_model.forward(actor_data, mode='compute_critic')['q_value']
            if self._cfg.learn.get('similarity_loss_weight_type',None):
                if self._cfg.learn.get('similarity_loss_weight_type',None) == 'mse':
                    mse_loss = torch.nn.MSELoss(reduction='none')
                    similarity_policy = mse_loss(actor_data['action'], data['action'])
                    similarity_policy = - similarity_policy.mean(-1)
                    similarity_policy= (similarity_policy-(-4))/4
                elif self._cfg.learn.get('similarity_loss_weight_type',None) == 'cosine':    
                    similarity_policy = torch.cosine_similarity(actor_data['action'], data['action'])
                else:
                    raise NotImplementedError
                similarity_policy = similarity_policy * self._cfg.learn.get('similarity_loss_weight_ratio',1)
                actor_loss = (-q_value*similarity_policy).mean()
            elif self._cfg.learn.get('not_bc',None):
                actor_loss = -q_value.mean()
            else:
                actor_loss = -q_value.mean()
                # add behavior cloning loss weight(\lambda)
                bc_loss = F.mse_loss(actor_data['action'], data['action'])
                if self._cfg.learn.lmbda_type=='fix':
                    self.lmbda = self._alpha
                    actor_loss = self.lmbda * actor_loss + bc_loss
                elif self._cfg.learn.lmbda_type=='q_value':
                    self.lmbda = self._alpha / q_value.abs().mean().detach()
                    actor_loss = self.lmbda * actor_loss + bc_loss
                elif self._cfg.learn.lmbda_type=='td_error':
                    self.lmbda = td_error_per_sample.detach().mean().item()
                    if self._cfg.learn.get('lmbda_ratio', None):
                        self.lmbda=self.lmbda*self._cfg.learn.lmbda_ratio
                    actor_loss = actor_loss + self.lmbda * bc_loss
                elif self._cfg.learn.lmbda_type=='q_value_v2':
                    self.lmbda = q_value.abs().mean().detach()
                    actor_loss = actor_loss + self.lmbda * bc_loss
                elif self._cfg.learn.lmbda_type=='learned':
                    if self._cfg.learn.get('min_q_weight_true_q_change',None):
                        if self._cfg.learn.get('min_q_weight_true_q_change_pureq',None):
                            iter=self._cfg.learn.get('min_q_weight_true_q_change_iter')
                            self.true_q=min([(self._forward_learn_cnt/iter)*(self.true_q_end-self.true_q_begin)+self.true_q_begin,self.true_q_end])
                    if self._cfg.learn.get('lmbda_learned_version',None)=='v2':
                        self.lmbda_meta+=self._cfg.learn.lmbda_learned_rate*(self.true_q-q_value.detach().mean().item())
                        lmbda_meta_min=-100
                        lmbda_meta_max=0.1
                        if self._cfg.learn.get('lmbda_learned_meta_min', None):
                            lmbda_meta_min=self._cfg.learn.lmbda_learned_meta_min
                        if self._cfg.learn.get('lmbda_learned_meta_max', None):
                            lmbda_meta_max=self._cfg.learn.lmbda_learned_meta_max
                        self.lmbda_meta=torch.clamp(self.lmbda_meta, min=lmbda_meta_min, max=lmbda_meta_max)
                        self.lmbda=self.lmbda_meta[0]
                        actor_loss =  self.lmbda * actor_loss +bc_loss
                    elif self._cfg.learn.get('lmbda_learned_version',None)=='v3':
                        self.lmbda_meta+=self._cfg.learn.lmbda_learned_rate*(q_value.detach().mean().item()-self.true_q)
                        lmbda_meta_min=-1000
                        lmbda_meta_max=1000
                        if self._cfg.learn.get('lmbda_learned_meta_min', None):
                            lmbda_meta_min=self._cfg.learn.lmbda_learned_meta_min
                        if self._cfg.learn.get('lmbda_learned_meta_max', None):
                            lmbda_meta_max=self._cfg.learn.lmbda_learned_meta_max
                        self.lmbda_meta=torch.clamp(self.lmbda_meta, min=lmbda_meta_min, max=lmbda_meta_max)
                        self.lmbda=self.lmbda_meta[0]
                        self.lmbda1 = self._alpha / q_value.abs().mean().detach()
                        actor_loss = self.lmbda1 * actor_loss + self.lmbda * bc_loss
                    else:
                        self.lmbda_meta+=self._cfg.learn.lmbda_learned_rate*(q_value.detach().mean().item()-self.true_q)
                        if self._cfg.learn.get('lmbda_learned_meta_min', None):
                            self.lmbda_meta=torch.clamp(self.lmbda_meta, min=self._cfg.learn.lmbda_learned_meta_min)
                        else:
                            self.lmbda_meta=torch.clamp(self.lmbda_meta, min=0.5)
                        self.lmbda=self.lmbda_meta[0]
                        actor_loss = actor_loss + self.lmbda * bc_loss
                else:
                    raise NotImplementedError
            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        # =============
        # after update
        # =============
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        if self._forward_learn_cnt>1100000:
            import sys
            sys.exit(0)
        ret = {
            'cur_lr_critic': self._optimizer_critic.param_groups[0]['lr'],
            'cur_lr_actor': self._optimizer_actor.param_groups[0]['lr'],
            'critic_fc_norm': self._get_fc_weight_norm(self._model.critic),
            'actor_fc_norm': self._get_fc_weight_norm(self._model.actor),
            # 'q_value': np.array(q_value).mean(),
            'action': data.get('action').mean(),
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.abs().mean(),
            'lmbda': float(self.lmbda),
            **loss_dict,
            **q_value_dict,
        }
        if self._cfg.learn.get('lmbda_type',None)=='learned':
            ret['true_q']=self.true_q
        if self._cfg.learn.get('similarity_loss_weight_type',None):
            ret['similarity_policy'] = float(similarity_policy.mean())
        if self._cfg.learn.get('q_value_similarity_loss_weight_type',None):
            ret['similarity_q'] = float(similarity.mean())
        return ret

    def _get_fc_weight_norm(self, net):
        with torch.no_grad():
            return torch.sqrt(sum([torch.sum(m.weight.clone()**2) for m in net.modules() if isinstance(m, torch.nn.Linear)]))
    
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
        if self.cfg.collect.normalize_states:
            data = (data - self._mean) / self._std
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def set_norm_statistics(self, mean: float, std: float) -> None:
        r"""
        Overview:
            Set (mean, std) for state normalization.
        Arguments:
            - mean (:obj:`float`): Float type data, the mean of state in offlineRL dataset.
            - std (:obj:`float`): Float type data, the std of state in offlineRL dataset.
        Returns:
            - None
        """
        self._mean = mean
        self._std = std

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = [
            'cur_lr_critic', 'cur_lr_actor', 'critic_fc_norm', 'actor_fc_norm',
            'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'q_value_twin', 'lmbda', 'similarity_policy', 'similarity_q', 
            'action', 'td_error'
        ]
        if self._twin_critic:
            ret += ['critic_twin_loss']
        if self._cfg.learn.get('lmbda_type',None)=='learned':
            ret += ['true_q']
        return ret