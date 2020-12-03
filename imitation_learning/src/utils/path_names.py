import os
from os.path import dirname, join

"""
Store global path names
"""
root_dir = "/usr/src/app/kaggle_environments" # Docker container workdir
raw_episodes_path = join(root_dir,"input/episodes_data")
stacked_smm_path = join(root_dir,"input/stacked_smm")
obs_frames_path = join(root_dir,"input/obs_frames_data")
