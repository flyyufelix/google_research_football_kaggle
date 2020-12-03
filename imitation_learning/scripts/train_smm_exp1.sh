python3 ../train_model.py \
    model.model_name=impala_cnn \
    data.dataset_name=spatial_smm \
    data.use_only_left_team=True \
    data.filter_by_score=True \
    data.filter_by_team=False \
    core.gpu_id=0 \
    log.exp_name=exp1
