import os
import glob
import json
import collections
import numpy as np
import pandas as pd
import pdb
import pickle
from tqdm import tqdm
from os.path import join, dirname

from torch.utils.data import Dataset, DataLoader, Subset

import gfootball.env as football_env
from gfootball.env import observation_preprocessing, wrappers

import sys

# Need this for relative import
sys.path.append("..")

from src.utils.path_names import raw_episodes_path, obs_frames_path

def make_obs_frames():
    """
    Transform raw episodes data into single observation frames
    """

    obs_frames_dir = obs_frames_path
    if not os.path.exists(obs_frames_dir):
        os.makedirs(obs_frames_dir, exist_ok=True)

    episodes_metadata_csv = join(dirname(raw_episodes_path),"episodes_metadata.csv")
    episodes_metadata_df = pd.read_csv(episodes_metadata_csv)

    #episode_list = glob.glob(raw_episodes_path + "/*.json")

    # Store metadata for stacked smm frames (i.e. frame name and corresponding ground-truth label)
    frames_metadata_file = join(dirname(raw_episodes_path), "obs_frames_metadata.csv")

    episode_length = 3002 # Hardcoded in Google Football Environment

    #episode_list = [episode for episode in episode_list if "_info" not in episode]

    err_samples = 0

    print("Loading Episodes...")
    with open(frames_metadata_file, "w") as frames_metadata_csv:

        frames_metadata_csv.write("frame_name,action,score,left_team,team_name,team_id\n")

        for idx in tqdm(episodes_metadata_df.index):

            episode = join(raw_episodes_path, str(episodes_metadata_df.loc[idx]['epid'])+'.json')

            with open(episode) as json_file: 

                episode_data = json.load(json_file)

                for step in range(episode_length):
                     
                    score = str(episodes_metadata_df.loc[idx]['score'])
                    left_team = str(episodes_metadata_df.loc[idx]['left_team'])
                    team_name = episodes_metadata_df.loc[idx]['team_name']
                    team_id = str(episodes_metadata_df.loc[idx]['team_id'])
                    #print(score, left_team, team_name, team_id)

                    try:
                        # Extract raw observation of the player we control (i.e. single player from left team)
                        raw_obs = episode_data['steps'][step][0]['observation']['players_raw'][0]

                        # Extact Action taken
                        action = episode_data['steps'][step][0]['action']

                        # Only save frames with action taken
                        if action is not None and len(action) > 0:

                            episode_number = episode.split('/')[-1].replace('.json','')
                            frame_name = episode_number + '_' + str(step+1)
                            frame_path = join(obs_frames_path, frame_name)
                            with open(frame_path, "wb") as handle:
                                pickle.dump(raw_obs, handle)
                            frames_metadata_csv.write(frame_name + ',' + str(action[0]) + ',' + score + ',' + left_team + ',' + team_name + ',' + team_id + '\n')
                    except:
                        err_samples += 1
                        print("Err: " + str(err_samples))


if __name__ == "__main__":
    make_obs_frames()





















