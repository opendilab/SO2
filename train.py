from copy import deepcopy
from ding.entry import serial_pipeline_offline2online
from d4rl import set_dataset_path
import os


def offline_train(args):
    env=args.env_id.split('-')[0]
    if env=='halfcheetah':
        from dizoo.mujoco.config.halfcheetah_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size=10
    elif env=='hopper':
        from dizoo.mujoco.config.hopper_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size=50
    elif env=='walker2d':
        from dizoo.mujoco.config.walker2d_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size=10

    # edac

    main_config.policy.learn.learner=dict()
    main_config.policy.learn.learner.hook=dict()
    
    
    if args.ckpt_path is not None:
        main_config.policy.learn.learner.load_path=args.ckpt_path
    else:
        main_config.policy.learn.learner.load_path=f'ckpt/{args.env_id}.ckpt'

    main_config.policy.learn.learner.hook.load_ckpt_before_run = main_config.policy.learn.learner.load_path
    main_config.policy.learn.learner.hook.log_show_after_iter=2000
    main_config.policy.learn.learner.hook.save_ckpt_after_iter=10000000000
    main_config.policy.learn.learner.hook.save_ckpt_after_run=True
    create_config.policy.type='edac'
    create_config.policy.import_names=['ding.policy.edac']

    # set experiment name
    main_config.exp_name = f'{args.env_id}_seed_{args.seed}_upc10_v2'
 
    # offline2online
    main_config.policy.learn.online = False # for interface consistence
    main_config.policy.learn.offline_pretrain_iterations=0
    main_config.policy.learn.online_pretrain_iterations=0
    # value network update frequency
    main_config.policy.learn.update_per_collect=10
    # actor network update frequency (update_per_collect/actor_update_freq)
    main_config.policy.learn.actor_update_freq=10
    main_config.policy.learn.concat_online_ratio=0.1
    main_config.policy.learn.batch_size=256
    main_config.policy.collect.n_sample=1
    main_config.policy.learn.online_ratio=0.5 # just for use offline buffer, but online train
    # main_config.policy.learn.offline_buffer_size=3000000
    main_config.policy.learn.offline_data_ratio=1
    main_config.policy.learn.without_timeouts_done=True

    main_config.policy.collect.data_type='d4rl'
    main_config.policy.collect.data_path=None
    main_config.policy.random_collect_size=5000
    main_config.env.env_id=args.env_id
    main_config.policy.eval.evaluator=dict()
    main_config.policy.eval.evaluator.eval_freq=1000
    # collect data
    # main_config.policy.collect.n_sample=1

    # smooth target policy
    main_config.policy.learn.noise = True
    main_config.policy.learn.noise_sigma=0.3
    main_config.policy.learn.noise_range=dict(min=-0.6,max=0.6,)
    config = deepcopy([main_config, create_config])
    serial_pipeline_offline2online(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_id', '-e', type=str, default='walker2d-medium-v2')
    parser.add_argument('--ckpt_path', '-c', type=str, default=None)
    args = parser.parse_args()

    # os.environ['D4RL_DATASET_DIR'] = ''
    # set_dataset_path('')
    offline_train(args)
