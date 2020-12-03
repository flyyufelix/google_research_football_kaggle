python3 ../train_ppo.py \
    core.gpu_id=0 \
    env.env_name=11_vs_11_easy_stochastic \
    env.obs_representation=float115 \
    model.model_name=mlp_float115 \
    model.mlp_input=115 \
    model.pretrained_model=model \
    log.exp_name=exp1


