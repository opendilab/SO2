from easydict import EasyDict

halfcheetah_cql_default_config = dict(
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=20000,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        random_collect_size=10000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            actor_head_type='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=1e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            with_lagrange=False,
            lagrange_thresh=-1.0,
            min_q_weight=5.0,
            critic_init=True,
            learner=dict(
                hook=dict(
                    # save_ckpt_after_iter=100,
                    log_show_after_iter=10000,
                ),
            ),
            lr_scheduler=dict(
                flag=False,
                T_max=3000000,
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            data_type='d4rl',
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=10000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

halfcheetah_cql_default_config = EasyDict(halfcheetah_cql_default_config)
main_config = halfcheetah_cql_default_config

halfcheetah_cql_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='cql',
        import_names=['ding.policy.cql'],
    ),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_cql_default_create_config = EasyDict(halfcheetah_cql_default_create_config)
create_config = halfcheetah_cql_default_create_config
