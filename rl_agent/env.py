#!/usr/bin/env python3

import gym
import json
import os
import argparse
import random
import multiprocessing as mp
import collections
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

from kaggle_environments import make
import gfootball.env as football_env
from gfootball.env import observation_preprocessing, wrappers
from gfootball.env.wrappers import Simple115StateWrapper

class FootballEnvWrapper(gym.Env):
    """
    Customized Environment that wraps around GRF gym environment
    """

    def __init__(self, env_name="11_vs_11_easy_stochastic", obs_representation="stacked_smm", rewards="scoring,checkpoints", logdir="/tmp/football", env_id=0):

        super(FootballEnvWrapper, self).__init__()

        print("Env: " + env_name)

        self.env = football_env.create_environment(
            env_name=env_name,
            stacked=False,
            representation='raw',
            rewards=rewards,
            logdir=logdir,
            write_goal_dumps=False and (env_id == 0),
            write_full_episode_dumps=False and (env_id == 0),
            write_video=False and (env_id == 0),
            render=False,
            dump_frequency=30)

        self.obs_representation = obs_representation

        # For Frame Stack
        self.stacked_obs = collections.deque([], maxlen=4)

        # Define observation space and action space
        # They must be gym.spaces objects

        self.action_space = gym.spaces.Discrete(19) # 19 actions

        if obs_representation == "smm":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72,96,4), dtype=np.uint8)
        elif obs_representation == "stacked_smm":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72,96,16), dtype=np.uint8)
        elif obs_representation == "float115":
            self.observation_space = gym.spaces.Box(low=-30.0, high=30.0, shape=(115,), dtype=np.float32)
        elif obs_representation == "pixels":
            pass
        elif obs_representation == "raw":
            # Use OBSParser
            self.observation_space = gym.spaces.Box(low=-30.0, high=30.0, shape=(207,), dtype=np.float32)

        self.ball_owned_team = -1
        self.rewards = rewards

    def step(self, action):

        # Step through the environment

        raw_obs, reward, done, info = self.env.step([action])

        # Obtain raw observation
        raw_obs = raw_obs[0]

        # Extract metainfo from obs
    
        # Reward Shaping (If applicable)

        if "ball_possession" in self.rewards:

            # Reward winning ball possession and penalize lossing ball possession
            prev_ball_owned_team = self.ball_owned_team
            cur_ball_owned_team = self.raw_obs['ball_owned_team']

            # Win ball possession
            if prev_ball_owned_team == 1 and cur_ball_owned_team == 0:
                reward += 0.1

            # Lose ball possession
            if prev_ball_owned_team == 0 and cur_ball_owned_team == 1:
                reward -= 0.1

            self.ball_owned_team = cur_ball_owned_team
        
        # Scale Rewards
        #reward = reward * 10

        if self.obs_representation == "smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
        elif self.obs_representation == "stacked_smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
            if not self.stacked_obs:
                self.stacked_obs.extend([obs] * 4)
            else:
                self.stacked_obs.append(obs)
            obs = np.concatenate(list(self.stacked_obs), axis=-1)
        elif self.obs_representation == "float115":
            obs = Simple115StateWrapper.convert_observation([raw_obs], True)[0]
        elif self.obs_representation == "pixels":
            pass
        elif self.obs_representation == "raw":
            obs,(l_score,r_score,custom_reward) = OBSParser.parse(obs)

        # Extract MetaInfo like scoring from raw_obs
        __,(l_score,r_score,__) = OBSParser.parse(raw_obs)

        info['l_score'] = l_score
        info['r_score'] = r_score

        # Use goal difference as custom reward for now
        return obs, reward, done, info

    def reset(self):

        raw_obs = self.env.reset()

        # Raw observations (See https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)
        #obs = obs['players_raw'][0]
        raw_obs = raw_obs[0]

        if self.obs_representation == "smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
        elif self.obs_representation == "stacked_smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
            if not self.stacked_obs:
                self.stacked_obs.extend([obs] * 4)
            else:
                self.stacked_obs.append(obs)
            obs = np.concatenate(list(self.stacked_obs), axis=-1)
        elif self.obs_representation == "float115":
            obs = Simple115StateWrapper.convert_observation([raw_obs], True)[0]
        elif self.obs_representation == "pixels":
            pass
        elif self.obs_representation == "raw":
            obs,_ = OBSParser.parse(obs)

        return obs

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass

class KaggleEnvWrapper(gym.Env):
    """
    Customized Environment that wraps around Kaggle environment
    """

    def __init__(self, adversarial_agent="do_nothing", env_name="11_vs_11_kaggle", obs_representation="stacked_smm", rewards="scoring,checkpoints", logdir="/tmp/football", env_id=0):

        super(KaggleEnvWrapper, self).__init__()

        print("Env: " + env_name)

        # Train agent against Baseline
        self.agents = [None, adversarial_agent] 
        self.env = make("football", 
                        debug=False, 
                        configuration={
                            "save_video": False,
                            "scenario_name": env_name,
                            "rewards": rewards,
                            "running_in_notebook": False
                        })

        self.trainer = None

        self.obs_representation = obs_representation

        # For Frame Stack
        self.stacked_obs = collections.deque([], maxlen=4)

        # Define observation space and action space
        # They must be gym.spaces objects

        self.action_space = gym.spaces.Discrete(19) # 19 actions

        if obs_representation == "smm":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72,96,4), dtype=np.uint8)
        elif obs_representation == "stacked_smm":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72,96,16), dtype=np.uint8)
        elif obs_representation == "float115":
            self.observation_space = gym.spaces.Box(low=-30.0, high=30.0, shape=(115,), dtype=np.float32)
        elif obs_representation == "pixels":
            pass
        elif obs_representation == "raw":
            # Use OBSParser
            self.observation_space = gym.spaces.Box(low=-30.0, high=30.0, shape=(207,), dtype=np.float32)

        self.ball_owned_team = -1
        self.rewards = rewards

    def step(self, action):

        # Step through the environment
        raw_obs, reward, done, info = self.trainer.step([action])

        # Obtain raw observation
        raw_obs = raw_obs['players_raw'][0]

        # Extract metainfo from obs
    
        # Reward Shaping (If applicable)

        if "ball_possession" in self.rewards:

            # Reward winning ball possession and penalize lossing ball possession
            prev_ball_owned_team = self.ball_owned_team
            cur_ball_owned_team = self.raw_obs['ball_owned_team']

            # Win ball possession
            if prev_ball_owned_team == 1 and cur_ball_owned_team == 0:
                reward += 0.1

            # Lose ball possession
            if prev_ball_owned_team == 0 and cur_ball_owned_team == 1:
                reward -= 0.1

            self.ball_owned_team = cur_ball_owned_team
        
        # Scale Rewards
        #reward = reward * 10

        if self.obs_representation == "smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
        elif self.obs_representation == "stacked_smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
            if not self.stacked_obs:
                self.stacked_obs.extend([obs] * 4)
            else:
                self.stacked_obs.append(obs)
            obs = np.concatenate(list(self.stacked_obs), axis=-1)
        elif self.obs_representation == "float115":
            obs = Simple115StateWrapper.convert_observation([raw_obs], True)[0]
        elif self.obs_representation == "pixels":
            pass
        elif self.obs_representation == "raw":
            obs,(l_score,r_score,custom_reward) = OBSParser.parse(obs)

        # Extract MetaInfo like scoring from raw_obs
        __,(l_score,r_score,__) = OBSParser.parse(raw_obs)

        info['l_score'] = l_score
        info['r_score'] = r_score

        # Use goal difference as custom reward for now
        return obs, reward, done, info

    def reset(self):

		# Train our agent against the baseline
		# See https://github.com/Kaggle/kaggle-environments#Training
        self.trainer = self.env.train(self.agents)

        raw_obs = self.trainer.reset()

        # Raw observations (See https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)
        raw_obs = raw_obs['players_raw'][0]

        if self.obs_representation == "smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
        elif self.obs_representation == "stacked_smm":
            obs = observation_preprocessing.generate_smm([raw_obs])[0]
            if not self.stacked_obs:
                self.stacked_obs.extend([obs] * 4)
            else:
                self.stacked_obs.append(obs)
            obs = np.concatenate(list(self.stacked_obs), axis=-1)
        elif self.obs_representation == "float115":
            obs = Simple115StateWrapper.convert_observation([raw_obs], True)[0]
        elif self.obs_representation == "pixels":
            pass
        elif self.obs_representation == "raw":
            obs,_ = OBSParser.parse(obs)

        return obs

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass

class OBSParser(object):
    """
    Helper class to parse raw observations
    """

    @staticmethod
    def parse(obs):
        # parse left players units
        l_units = [[x[0] for x in obs['left_team']], [x[1] for x in obs['left_team']],
                   [x[0] for x in obs['left_team_direction']], [x[1] for x in obs['left_team_direction']],
                   obs['left_team_tired_factor'], obs['left_team_yellow_card'],
                   obs['left_team_active'], obs['left_team_roles']
                  ]

        l_units = np.r_[l_units].T

        # parse right players units
        r_units = [[x[0] for x in obs['right_team']], [x[1] for x in obs['right_team']],
                   [x[0] for x in obs['right_team_direction']], [x[1] for x in obs['right_team_direction']],
                   obs['right_team_tired_factor'],
                   obs['right_team_yellow_card'],
                   obs['right_team_active'], obs['right_team_roles']
                  ]

        r_units = np.r_[r_units].T

        # combine left and right players units
        # [22x8] matrix
        units = np.r_[l_units, r_units].astype(np.float32)

        # get other information
        
        # Create one-hot-vector for game mode
        game_mode = [0 for _ in range(7)]
        game_mode[obs['game_mode']] = 1

        # Total 31 elements
        scalars = [*obs['ball'],
                   *obs['ball_direction'], 
                   *obs['ball_rotation'],
                   obs['ball_owned_team'],
                   obs['ball_owned_player'],
                   *obs['score'],
                   obs['steps_left'],
                   *game_mode,
                   *obs['sticky_actions']]

        scalars = np.r_[scalars].astype(np.float32)

        # Concatenate units and scalars
        combined_obs = np.concatenate((units.flatten(), scalars))

        # get the actual scores and compute a reward
        l_score,r_score = obs['score'] # Pairs of int indicating number of goals
        reward = l_score - r_score # Use goal difference as reward (Not a good reward, very sparse)
        reward_info = l_score,r_score,reward

        #return (units[np.newaxis, :], scalars[np.newaxis, :]),reward_info
        return combined_obs, reward_info

class VecPyTorch(VecEnvWrapper):
    """
    Move all numpy array to pytorch tensors and register them onto GPU devices (if available)
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))
