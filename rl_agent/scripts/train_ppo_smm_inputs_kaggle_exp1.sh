python3 ../train_ppo.py \
    core.gpu_id=0 \
    env.env_name=11_vs_11_kaggle \
    model.model_name=impala_cnn \
    model.pretrained_model=pretrained_model \
    log.exp_name=exp1 \
    env.use_kaggle_wrapper=True \
    env.rewards=scoring \
    train.use_prierarchy_loss=False


