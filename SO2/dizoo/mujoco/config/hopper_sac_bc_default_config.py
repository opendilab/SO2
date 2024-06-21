from easydict import EasyDict

hopper_sac_bc_default_config = dict(
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            actor_head_type='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            learner=dict(
                hook=dict(
                    # save_ckpt_after_iter=100,
                    log_show_after_iter=10000,
                ),
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            data_type='hdf5',
            data_path='./td3/expert.pkl',
            normalize_states=True,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=10000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

hopper_sac_bc_default_config = EasyDict(hopper_sac_bc_default_config)
main_config = hopper_sac_bc_default_config

hopper_sac_bc_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac_bc',
        import_names=['ding.policy.sac_bc'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_sac_bc_default_create_config = EasyDict(hopper_sac_bc_default_create_config)
create_config = hopper_sac_bc_default_create_config
