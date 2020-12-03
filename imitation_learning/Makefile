SHELL=/bin/bash

# DAG for preparing training data
prepare_data: download_episodes make_frames

etl_and_train: download_episodes make_frames train_model

download_episodes:
	python3 ./scripts/download_episodes.py

make_frames:
	python3 ./scripts/make_obs_frames.py

train_model:
	python3 ./train_model.py
