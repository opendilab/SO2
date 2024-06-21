from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
import random
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset

from torch.utils.data import DataLoader

def normalize(data):
    mean = data.mean(0, keepdims=True)
    std = data.std(0, keepdims=True) + 1e-10
    return (data - mean) / std

def aug_with_offline_data(data, dataset, aug_num=10, aug_dist_threshold=5, include_next_obs=True):
    if include_next_obs:
        obs = torch.from_numpy(dataset.raw_dataset['observations'])
        action = torch.from_numpy(dataset.raw_dataset['actions'])
        next_obs = torch.from_numpy(dataset.raw_dataset['next_observations'])
        database = torch.cat([obs, action, next_obs], dim=-1)
        mean = database.mean(0, keepdims=True)
        std = database.std(0, keepdims=True) + 1e-10
        database = ((database - mean) / std).cuda().unsqueeze(0)
        query = torch.cat([torch.cat([i['obs'], i['action'], i['next_obs']], dim=-1) for i in data], dim=0)
        query = ((query - mean) / std).cuda().unsqueeze(0)
        if len(query.shape) == 2:
            query.unsqueeze_(0)
        dist = torch.cdist(query, database, p=2).squeeze(0)
        values, indices = torch.topk(dist, aug_num, -1, largest=False)
        min_value = values.min().item()
        values = values < aug_dist_threshold
        indices = indices[values].view(-1)
        aug_data = [dataset.data[i] for i in indices.tolist()]
    else:
        obs = torch.from_numpy(dataset.raw_dataset['observations'])
        action = torch.from_numpy(dataset.raw_dataset['actions'])
        database = torch.cat([obs, action], dim=-1)
        mean = database.mean(0, keepdims=True)
        std = database.std(0, keepdims=True) + 1e-10
        database = ((database - mean) / std).cuda().unsqueeze(0)
        query = torch.cat([torch.cat([i['obs'], i['action']], dim=-1) for i in data], dim=0)
        query = ((query - mean) / std).cuda().unsqueeze(0)
        if len(query.shape) == 2:
            query.unsqueeze_(0)
        dist = torch.cdist(query, database, p=2).squeeze(0)
        values, indices = torch.topk(dist, 1, -1, largest=False)
        min_value = values.min().item()
        values = values < aug_dist_threshold
        if values.sum()>0:
            indice = indices[values].view(-1).item()
            max_indice = min(indice+aug_num, len(dataset.data))
            aug_data = [dataset.data[i] for i in range(indice, max_indice)]
        else:
            aug_data = []
    data.extend(aug_data)
    return data, min_value

def serial_pipeline_offline2online_v2(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e6),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    dataset = create_dataset(cfg)
    
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    for d in dataset.data:
        d['reward']=d['reward'].reshape(-1)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False
    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        collector.reset_policy(policy.collect_mode)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=0)

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        collect_kwargs = commander.step()
        # Collect data by default config n_sample/n_episode
        if collector.envstep>250000:
            import sys
            sys.exit(0)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        if cfg.policy.learn.aug_num>0:
            new_data, min_dist = aug_with_offline_data(
                new_data, 
                dataset, 
                aug_num=cfg.policy.learn.aug_num, 
                aug_dist_threshold=cfg.policy.learn.aug_dist_threshold,
                include_next_obs = cfg.policy.learn.aug_include_next_obs,
            )
        else:
            min_dist = 0
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                logging.warning(
                    "Replay buffer's data can only train for {} steps. ".format(i) +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break
            train_data[0]['aug_data_len']=float(len(new_data))
            train_data[0]['aug_data_min_dist']=float(min_dist)

            learner.train(train_data, collector.envstep)
            # if learner.policy.get_attribute('priority'):
                # replay_buffer.update(learner.priority_info)

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy, stop
