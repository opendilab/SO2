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


def serial_pipeline_offline2online(
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
    if cfg.policy.learn.get('offline_buffer_size', None):
        cfg.policy.other.replay_buffer.replay_buffer_size=cfg.policy.learn.offline_buffer_size
    else:
        cfg.policy.other.replay_buffer.replay_buffer_size=len(dataset)
    offline_buffer=create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name, instance_name='offline_buffer')
    for d in dataset.data:
        d['reward']=d['reward'].reshape(-1)
    import random
    offline_buffer.push(random.sample(dataset.data, int(len(dataset.data)*cfg.policy.learn.get('offline_data_ratio', 1))), cur_collector_envstep=collector.envstep)
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
    for _ in range(learner.train_iter, cfg.policy.learn.offline_pretrain_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            if stop:
                break
        train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        cfg.policy.learn.online = False
        # learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = True
        learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = False
    
    if cfg.policy.get('random_collect_size', 0) > 0:
        collector.reset_policy(policy.collect_mode)
        collect_kwargs = commander.step()
        new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=0)
    
    for _ in range(cfg.policy.learn.offline_pretrain_iterations, 
            cfg.policy.learn.get('online_pretrain_iterations', cfg.policy.learn.offline_pretrain_iterations)):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            if stop:
                break
        if cfg.policy.learn.get('concat_online_linear', False):
            concat_online_ratio = min(cfg.policy.learn.concat_online_ratio / cfg.policy.learn.concat_online_tmax * collector.envstep, cfg.policy.learn.concat_online_ratio)
        else:
            concat_online_ratio = cfg.policy.learn.concat_online_ratio
        batch_size = min(int(256/concat_online_ratio), 8192)
        online_bs = int(batch_size * concat_online_ratio)
        offline_bs = batch_size - online_bs
        train_data = replay_buffer.sample(online_bs, learner.train_iter)
        if cfg.policy.learn.get('pretrain_with_offline', False):
            # train_data = replay_buffer.sample(online_bs, learner.train_iter)
            offlinetrain_data = offline_buffer.sample(offline_bs, learner.train_iter)
            train_data.extend(offlinetrain_data)
        policy._cfg.learn.only_value = True
        learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = False

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        collect_kwargs = commander.step()
        if collector.envstep>150000:
            import sys
            sys.exit(0)
        # Collect data by default config n_sample/n_episode
        if cfg.policy.learn.concat_online_ratio > 0:
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            offline_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data

        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if cfg.policy.learn.concat_online_ratio > 0:
                cfg.policy.learn.online = True
                if cfg.policy.learn.get('concat_online_linear', False):
                    concat_online_ratio = min(cfg.policy.learn.concat_online_ratio / cfg.policy.learn.concat_online_tmax * collector.envstep, cfg.policy.learn.concat_online_ratio)
                else:
                    concat_online_ratio = cfg.policy.learn.concat_online_ratio
                batch_size = min(int(256/concat_online_ratio), 8192)
                online_bs = int(batch_size * concat_online_ratio)
                offline_bs = batch_size - online_bs
                train_data = replay_buffer.sample(online_bs, learner.train_iter)
                offlinetrain_data = offline_buffer.sample(offline_bs, learner.train_iter)
                train_data.extend(offlinetrain_data)
            else:
                if random.uniform(0, 1) < cfg.policy.learn.online_ratio:
                    train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                    cfg.policy.learn.online = False # use min_q_weight in forward not init in cql 
                    if train_data is None:
                        # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                        logging.warning(
                            "Replay buffer's data can only train for {} steps. ".format(i) +
                            "You can modify data collect config, e.g. increasing n_sample, n_episode."
                        )
                        break
                else:
                    train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                    cfg.policy.learn.online = False
            learner.train(train_data, collector.envstep)
            # if learner.policy.get_attribute('priority'):
                # replay_buffer.update(learner.priority_info)

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy, stop
