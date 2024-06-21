import torch
from copy import deepcopy

from ding.entry import serial_pipeline_offline, collect_demo_data, eval, serial_pipeline


def offline_train(args):
    # from dizoo.mujoco.config.halfcheetah_cql_default_config import main_config, create_config
    # main_config.policy.learn.critic_init=False
    # main_config.exp_name = 'cql-train_d4rl-halfcheetah-expert-v0_fix-shape_version4-noweight-init'
    # main_config.env.env_id='halfcheetah-expert-v0'

    # from dizoo.mujoco.config.walker2d_cql_default_config import main_config, create_config
    # main_config.policy.learn.critic_init=False
    # main_config.exp_name = 'cql-train_d4rl-walker2d-expert-v0_fix-shape_version4-noweight-init'
    # main_config.env.env_id='walker2d-expert-v0'

    # from dizoo.mujoco.config.halfcheetah_cql_default_config import main_config, create_config
    # main_config.policy.learn.critic_init=True
    # main_config.exp_name = 'cql-train_d4rl-halfcheetah-expert-v0_fix-shape_version5-weight-init'
    # main_config.env.env_id='halfcheetah-expert-v0'

    # from dizoo.mujoco.config.walker2d_cql_default_config import main_config, create_config
    # main_config.policy.learn.critic_init=True
    # main_config.exp_name = 'cql-train_d4rl-walker2d-expert-v0_fix-shape_version5-weight-init'
    # main_config.env.env_id='walker2d-expert-v0'
    
    from dizoo.mujoco.config.walker2d_cql_default_config import main_config, create_config
    main_config.policy.learn.critic_init=True
    main_config.exp_name = 'debug'
    main_config.env.env_id='walker2d-expert-v0'

    config = deepcopy([main_config, create_config])
    serial_pipeline_offline(config, seed=args.seed)


def eval_ckpt(args):
    from dizoo.mujoco.config.hopper_sac_default_config import main_config, create_config
    main_config.exp_name = 'sac'
    main_config.policy.learn.learner=dict()
    main_config.policy.learn.learner.hook=dict()
    main_config.policy.learn.learner.load_path = main_config.exp_name+'/ckpt/ckpt_best.pth.tar'
    main_config.policy.learn.learner.hook.load_ckpt_before_run =  main_config.exp_name+'/ckpt/ckpt_best.pth.tar'
    config = deepcopy([main_config, create_config])
    eval(config, seed=args.seed, load_path=main_config.policy.learn.learner.hook.load_ckpt_before_run)


def generate(args):
    from dizoo.mujoco.config.hopper_sac_data_generation_config import main_config, create_config
    main_config.exp_name = 'sac'
    main_config.policy.learn.learner.load_path = main_config.exp_name+'/ckpt/ckpt_best.pth.tar'
    main_config.policy.learn.save_path = main_config.exp_name+'/expert_determistic'
    main_config.policy.other.replay_buffer.replay_buffer_size = 1000000
    main_config.policy.learn.data_type = 'hdf5'
    config = deepcopy([main_config, create_config])
    state_dict = torch.load(main_config.policy.learn.learner.load_path, map_location='cpu')
    collect_demo_data(config, collect_count=main_config.policy.other.replay_buffer.replay_buffer_size,
                      seed=args.seed, expert_data_path=main_config.policy.learn.save_path, state_dict=state_dict)

def train_expert(args):
    from dizoo.mujoco.config.hopper_sac_default_config import main_config, create_config
    main_config.exp_name = 'sac_q_next_nogamma'
    config = deepcopy([main_config, create_config])
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    # train_expert(args)
    # eval_ckpt(args)
    # generate(args)
    offline_train(args)
