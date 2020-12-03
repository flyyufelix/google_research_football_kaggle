python3 ../train_model.py \
    model.model_name=mlp_float115 \
    data.dataset_name=float115 \
    data.use_only_left_team=True \
    data.filter_by_score=False \
    data.filter_by_team=False \
    core.gpu_id=0 \
    train.batch_size=256 \
    log.exp_name=exp1
