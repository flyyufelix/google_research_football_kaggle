import os
import glob
import json
import collections
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import pdb
import pickle
from os.path import join, dirname

from torch.utils.data import Dataset, DataLoader, Subset

import gfootball.env as football_env
from gfootball.env import observation_preprocessing, wrappers
from gfootball.env.wrappers import Simple115StateWrapper

import sys

# Need this for relative import
sys.path.append("..")

from src.utils.path_names import raw_episodes_path, obs_frames_path

class EpisodeDataset(Dataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        self.episode_length = 3002 # Hardcoded in Google Football Environment
        self.train = train
        self.transforms = transforms
        self.stack_frames = data_config.stack_frames

        self.df = df

    def __getitem__(self, idx):
        """
        Return raw observation frame 
        """
        # Retrieve raw observation
        frame_name = self.df.loc[idx,'frame_name']
        with open(join(obs_frames_path, frame_name)) as pkl_file:
            raw_obs = pickle.load(pkl_file)

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return raw_obs, action
        else:
            return raw_obs

    def __len__(self):
        return self.df.shape[0]

class SpatialMinimapDataset(EpisodeDataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        super(SpatialMinimapDataset, self).__init__(data_config, df, train, transforms)

    def __getitem__(self, idx):
        """
        Return Stacked Spatial Minimap Representation (SMM)
        Reference: https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#Observation%20Wrappers
        """

        #print("IDX: ", idx)

        # For Frame Stack
        stacked_obs = collections.deque([], maxlen=self.stack_frames)

        frame_name = self.df.loc[idx,'frame_name']
        frame_step = int(frame_name.split('_')[1])
        if frame_step >= 5 and idx >= 5:
            for frame_idx in list(range(idx+1))[-self.stack_frames:]:
                frame_name = self.df.loc[frame_idx,'frame_name']
                with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
                    raw_obs = pickle.load(pkl_file)
                    smm_obs = observation_preprocessing.generate_smm([raw_obs])[0]
                    smm_obs = smm_obs / 255.0
                    stacked_obs.append(smm_obs)

        else:
            with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
                raw_obs = pickle.load(pkl_file)
                smm_obs = observation_preprocessing.generate_smm([raw_obs])[0]
                smm_obs = smm_obs / 255.0
                stacked_obs.extend([smm_obs] * self.stack_frames)

        smm_frame = np.concatenate(list(stacked_obs), axis=-1)

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return smm_frame, int(action)
        else:
            return smm_frame

    def __len__(self):
        return self.df.shape[0]

class Float115Dataset(EpisodeDataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        super(Float115Dataset, self).__init__(data_config, df, train, transforms)

    def __getitem__(self, idx):
        """
        Return Float115_v2 Representation
        Reference: https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#Observation%20Wrappers
        """

        frame_name = self.df.loc[idx,'frame_name']
        with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
            raw_obs = pickle.load(pkl_file)
            float115_frame = Simple115StateWrapper.convert_observation([raw_obs], True)[0]

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return float115_frame, int(action)
        else:
            return float115_frame

    def __len__(self):
        return self.df.shape[0]

class HybridDataset(EpisodeDataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        super(HybridDataset, self).__init__(data_config, df, train, transforms)

    def __getitem__(self, idx):
        """
        Return Stacked Spatial Minimap (SMM) and Float115_v2 Representation
        Reference: https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#Observation%20Wrappers
        """

        # For Frame Stack
        stacked_obs = collections.deque([], maxlen=self.stack_frames)

        frame_name = self.df.loc[idx,'frame_name']
        frame_step = int(frame_name.split('_')[1])
        if frame_step >= 5 and idx >= 5:
            for frame_idx in list(range(idx+1))[-self.stack_frames:]:
                frame_name = self.df.loc[frame_idx,'frame_name']
                with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
                    raw_obs = pickle.load(pkl_file)
                    smm_obs = observation_preprocessing.generate_smm([raw_obs])[0]
                    smm_obs = smm_obs / 255.0
                    stacked_obs.append(smm_obs)

        else:
            with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
                raw_obs = pickle.load(pkl_file)
                smm_obs = observation_preprocessing.generate_smm([raw_obs])[0]
                smm_obs = smm_obs / 255.0
                stacked_obs.extend([smm_obs] * self.stack_frames)

        smm_frame = np.concatenate(list(stacked_obs), axis=-1)

        # Float115 Obs
        float115_frame = Simple115StateWrapper.convert_observation([raw_obs], True)[0]

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return (smm_frame, float115_frame), int(action)
        else:
            return (smm_frame, float115_frame)

    def __len__(self):
        return self.df.shape[0]

class RelativePosDataset(EpisodeDataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        super(RelativePosDataset, self).__init__(data_config, df, train, transforms)

    def __getitem__(self, idx):
        """
        Encode the relative position of all players and ball position to the player we controlled

        49 Dimension Vector
        """

        frame_name = self.df.loc[idx,'frame_name']
        with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:

            raw_obs = pickle.load(pkl_file)

            active_player = raw_obs["left_team"][raw_obs["active"]]

            feat_vec = []

            # If have ball possession
            if raw_obs["ball_owned_player"] == active_player and raw_obs["ball_owned_team"] == 0:
                feat_vec += [1.0]
            else:
                feat_vec += [0.0]

            distances = []
            angles = []

            # Distance and Angles from goal
            goal = [0.0,0.0]
            distances.append(get_distance(active_player,goal))
            angles.append(get_angle(active_player,goal))

            # Distance and Angles from ball
            ball = raw_obs["ball"]
            distances.append(get_distance(active_player,ball))
            angles.append(get_angle(active_player,ball))

            # Distance and Angles from Teammates (i.e. Left team)
            for left_player in raw_obs["left_team"]:
                distances.append(get_distance(active_player,left_player))
                angles.append(get_angle(active_player,left_player))

            # Distance and Angles from Opponents (i.e. Right team)
            for right_player in raw_obs["right_team"]:
                distances.append(get_distance(active_player,right_player))
                angles.append(get_angle(active_player,right_player))

            feat_vec += distances
            feat_vec += angles

        feat_vec = np.array(feat_vec)

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return feat_vec, int(action)
        else:
            return feat_vec

    def __len__(self):
        return self.df.shape[0]

class LidarFormatDataset(EpisodeDataset):

    def __init__(self, data_config, df, train=True, transforms=None):

        super(LidarFormatDataset, self).__init__(data_config, df, train, transforms)

    def __getitem__(self, idx):
        """
        Reference Lidar Format. From the vantage point of the player we control, perform a 360 degree scan and return the closest object for each degree within a range

        725 Dimension Vector
        """

        frame_name = self.df.loc[idx,'frame_name']
        with open(join(obs_frames_path, frame_name), 'rb') as pkl_file:
            raw_obs = pickle.load(pkl_file)

            active_player = raw_obs["left_team"][raw_obs["active"]]

            max_range = 0.3

            feat_vec = []

            # If have ball possession
            if raw_obs["ball_owned_player"] == active_player and raw_obs["ball_owned_team"] == 0:
                feat_vec += [1.0]
            else:
                feat_vec += [0.0]

            # Distance and Angles from goal
            goal = [0.0,0.0]
            feat_vec.append(get_distance(active_player,goal))
            feat_vec.append(get_angle(active_player,goal))

            # Distance and Angles from ball
            ball = raw_obs["ball"]
            feat_vec.append(get_distance(active_player,ball))
            feat_vec.append(get_angle(active_player,ball))

            # Obtain Lidar Scan for Teammates (i.e. Left team)
            angle_bins = list(np.zeros(360))
            for left_player in raw_obs["left_team"]:
                distance = get_distance(active_player,left_player)
                angle = get_angle(active_player,left_player)
                if (angle_bins[math.floor(angle)] == 0 or distance < angle_bins[math.floor(angle)]) and distance < max_range:
                    angle_bins[math.floor(angle)] = distance
            feat_vec += angle_bins

            # Obtain Lidar Scan for Opponents (i.e. Right team)
            angle_bins = list(np.zeros(360))
            for right_player in raw_obs["right_team"]:
                distance = get_distance(active_player,right_player)
                angle = get_angle(active_player,right_player)
                if (angle_bins[math.floor(angle)] == 0 or distance < angle_bins[math.floor(angle)]) and distance < max_range:
                    angle_bins[math.floor(angle)] = distance
            feat_vec += angle_bins

        feat_vec = np.array(feat_vec)

        # Retrieve action
        action = self.df.loc[idx,'action']

        if self.train:
            return feat_vec, int(action)
        else:
            return feat_vec

    def __len__(self):
        return self.df.shape[0]

def get_angle(my_obj, target_obj):
    """
    Return angle of target_obj from the perspective of my_obj
    """
    angle = math.atan2(target_obj[0]-my_obj[0], target_obj[1]-my_obj[1])*180/math.pi
    if angle < 0:
        return 360 + angle
    return angle

def get_distance(obj_a, obj_b):
    """
    Return Eucleadian Distance of 2 objects 
    """
    return math.sqrt( (obj_a[0] - obj_b[0])**2 + (obj_a[1] - obj_b[1])**2 )

def get_spatial_smm(data_config, df):
    dataset = SpatialMinimapDataset(data_config, df)
    return dataset

def get_float115(data_config, df):
    dataset = Float115Dataset(data_config, df)
    return dataset

def get_relative_pos(data_config, df):
    dataset = RelativePosDataset(data_config, df)
    return dataset

def get_lidar_format(data_config, df):
    dataset = LidarFormatDataset(data_config, df)
    return dataset

def get_hybrid(data_config, df):
    dataset = HybridDataset(data_config, df)
    return dataset

def get_dataset(data_config, df):
    print("Load Dataset: ", data_config.dataset_name)
    f = globals().get("get_" + data_config.dataset_name)
    return f(data_config, df)
