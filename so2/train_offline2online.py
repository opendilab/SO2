from typing import Tuple
from ditk import logging
import os
import random
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, create_buffer, create_serial_collector
from ding.config import compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset


def train_offline2online(
    input_cfg: Tuple[dict, dict],
    seed: int = 0,
    max_env_step: int = 150000,
):
    cfg, create_cfg = input_cfg
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Create main components: env, policy, dataset
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy)
    dataset = create_dataset(cfg)

    # Create worker components: learner, collector, evaluator, replay buffer, offline buffer
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

    cfg.policy.other.replay_buffer.replay_buffer_size = len(dataset)
    offline_buffer = create_buffer(
        cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name, instance_name='offline_buffer'
    )
    for d in dataset.data:
        d['reward'] = d['reward'].reshape(-1)
        d['collect_iter'] = -1
    sample_size = int(len(dataset) * cfg.policy.learn.offline_data_ratio)
    offline_buffer.push(random.sample(dataset.data, sample_size), cur_collector_envstep=collector.envstep)

    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    # Offline pretrain
    policy._cfg.learn.only_value = True
    for _ in range(cfg.policy.learn.offline_pretrain_iterations):
        train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        learner.train(train_data, collector.envstep)
    policy._cfg.learn.only_value = False
    logging.info('Offline pretrain phase is finished')

    # Random collect
    collector.reset_policy(policy.collect_mode)
    new_data = collector.collect(n_sample=cfg.policy.random_collect_size)
    replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
    logging.info('Random collect phase is finished')

    # Online pretrain
    policy._cfg.learn.only_value = True
    for _ in range(cfg.policy.learn.offline_pretrain_iterations, cfg.policy.learn.online_pretrain_iterations):
        concat_online_ratio = cfg.policy.learn.concat_online_ratio
        batch_size = min(int(256 / concat_online_ratio), 8192)
        online_bs = int(batch_size * concat_online_ratio)
        offline_bs = batch_size - online_bs
        train_data = replay_buffer.sample(online_bs, learner.train_iter)
        learner.train(train_data, collector.envstep)
    policy._cfg.learn.only_value = False
    logging.info('Online pretrain phase is finished, online training starts...')

    # Online training
    stop = False
    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # max training env steps
        if collector.envstep > max_env_step:
            logging.info('Training in finished')
            break
        # Collect data by default config n_sample/n_episode
        if cfg.policy.learn.concat_online_ratio > 0:
            new_data = collector.collect(train_iter=learner.train_iter)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            offline_buffer.push(new_data, cur_collector_envstep=collector.envstep)

        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if cfg.policy.learn.concat_online_ratio > 0:
                batch_size = min(int(256 / cfg.policy.learn.concat_online_ratio), 8192)
                online_bs = int(batch_size * cfg.policy.learn.concat_online_ratio)
                offline_bs = batch_size - online_bs
                train_data = replay_buffer.sample(online_bs, learner.train_iter)
                offline_train_data = offline_buffer.sample(offline_bs, learner.train_iter)
                train_data.extend(offline_train_data)
            else:
                if random.uniform(0, 1) < cfg.policy.learn.online_ratio:
                    train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                    if train_data is None:
                        logging.warning(
                            "Replay buffer's data can only train for {} steps. ".format(i) +
                            "You can modify data collect config, e.g. increasing n_sample, n_episode."
                        )
                        break
                else:
                    train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)

    # Learner's after_run hook.
    learner.call_hook('after_run')
