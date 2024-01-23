from copy import deepcopy
from d4rl import set_dataset_path

from train_offline2online import train_offline2online


def offline_train(args):
    env = args.env_id.split('-')[0]
    if env == 'halfcheetah':
        from dizoo.mujoco.config.halfcheetah_sac_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 10
    elif env == 'hopper':
        from dizoo.mujoco.config.hopper_sac_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 50
    elif env == 'walker2d':
        from dizoo.mujoco.config.walker2d_sac_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 10

    if args.env_id == 'halfcheetah-random-v2':
        path = args.ckpt_path
    else:
        raise NotImplementedError(f"not implememnt env: {args.env_id}")
    main_config.policy.learn.learner = dict()
    main_config.policy.learn.learner.hook = dict()
    # main_config.policy.learn.learner.hook.load_ckpt_before_run = load_path
    main_config.policy.learn.learner.hook.log_show_after_iter = 2000
    main_config.policy.learn.learner.hook.save_ckpt_after_iter = int(1e9)  # disable save_ckpt_after_iter
    main_config.policy.learn.learner.hook.save_ckpt_after_run = True
    create_config.policy.type = 'edac'
    create_config.policy.import_names = ['ding.policy.edac']

    # offline2online
    main_config.policy.learn.offline_pretrain_iterations = 0
    main_config.policy.learn.update_per_collect = 1
    main_config.policy.learn.concat_online_ratio = 0.1
    main_config.policy.learn.batch_size = 256
    main_config.policy.learn.online_ratio = 0.5  # just for use offline buffer, but online train
    main_config.policy.learn.online_pretrain_iterations = 100
    main_config.policy.learn.offline_data_ratio = 1
    main_config.policy.learn.without_timeouts_done = True

    main_config.policy.collect.data_type = 'd4rl'
    main_config.policy.random_collect_size = 5000
    main_config.env.env_id = args.env_id
    main_config.policy.eval.evaluator = dict(eval_freq=1000)

    # smooth target policy
    main_config.policy.learn.noise = True
    main_config.policy.learn.noise_sigma = 0.3
    main_config.policy.learn.noise_range = dict(
        min=-0.6,
        max=0.6,
    )
    main_config.exp_name = f'seed{args.seed}'
    config = deepcopy([main_config, create_config])
    train_offline2online(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_id', '-e', type=str, default='halfcheetah-random-v2')
    parser.add_argument('--ckpt_path', '-c', type=str)
    args = parser.parse_args()

    offline_train(args)
