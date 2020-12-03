#!/usr/bin/env python3

import gym
import json
import os
import random
import multiprocessing as mp
import collections
import numpy as np
from omegaconf import DictConfig
import hydra
import wandb
from os.path import join
import pdb
import logging
import math
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gfootball.env as football_env
from gfootball.env import observation_preprocessing, wrappers

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from model import get_model
from env import FootballEnvWrapper, KaggleEnvWrapper, ImageToPyTorch, VecPyTorch

#os.environ['WANDB_MODE']="dryrun"

def prepare_dir(config: DictConfig) -> None:
    """
    Prepare directories to store results, logs, model artifacts
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)

def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_up(config):
    """
    Set up the environment
    """
    prepare_dir(config)
    set_seed(config.core.seed)

    # Register GPUs to torch so can use parallel training (i.e. DistributedDataParallel)
    #for device in config.core.gpu_id:
    #    torch.cuda.set_device(f"cuda:{device}")
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.core.gpu_id)

    #torch.backends.cudnn.deterministic = config.core.torch_deterministic
    #torch.backends.cudnn.benchmark = config.core.torch_benchmark


class ImpalaCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(ImpalaCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        layers = []
        depth_in = observation_space.shape[0]
        for depth_out in [depth_in, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(3456, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(observations)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x

class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


@hydra.main(config_name="config")
def main(config: DictConfig) -> None:

    start_time = time.time()

    set_up(config)

    device = torch.device('cuda:'+str(config.core.gpu_id) if torch.cuda.is_available() and config.core.use_gpu else 'cpu')

    # Set up Wandb (Pass config variables to wandb)
    if config.log.use_wandb:
        hparams = {}
        for key, value in config.items():
            hparams.update(value)

        wandb.init(project="GRF_RL_training", config=hparams)

    if config.log.use_wandb:
        log_handler = wandb
    else:
        log_handler = None

    # Lambda Function to Create Environment
    def make_env(i):
        def thunk():
            if not config.env.use_kaggle_wrapper:
                env = FootballEnvWrapper(
                    env_name=config.env.env_name, 
                    obs_representation=config.env.obs_representation, 
                    rewards=config.env.rewards,
                    logdir=config.store.log_path,
                    env_id=i)
            else:
                print("Training against agent: " + join(config.env.adversarial_agent_path,config.env.adversarial_agent))
                env = KaggleEnvWrapper(
                    adversarial_agent=join(config.env.adversarial_agent_path,config.env.adversarial_agent),
                    env_name=config.env.env_name, 
                    obs_representation=config.env.obs_representation, 
                    rewards=config.env.rewards,
                    logdir=config.store.log_path,
                    env_id=i)
            env.seed(i)
            return env
        return thunk

    if config.env.parallel_env:
        envs = SubprocVecEnv([make_env(i) for i in range(config.env.num_envs)])
    else:
        envs = DummyVecEnv([make_env(i) for i in range(config.env.num_envs)])

    policy_kwargs = dict(
        features_extractor_class=ImpalaCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Stable-baselines3 PPO
    model = PPO(
        policy="CnnPolicy", 
        policy_kwargs=policy_kwargs,
        env=envs, 
        learning_rate=config.train.learning_rate,
        n_steps=config.train.num_steps,
        n_epochs=config.train.update_epochs,
        batch_size=config.train.batch_size,
        clip_range=config.train.clip_range,
        gamma=config.train.gamma,
        gae_lambda=config.train.gae_lambda,
        max_grad_norm=config.train.max_grad_norm,
        vf_coef=config.train.vf_coef,
        ent_coef=config.train.ent_coef,
        log_handler=log_handler,
        model_checkpoints_path=config.store.model_path,
        pretrained_model=join(config.model.pretrained_model_path, config.model.pretrained_model),
        use_prierarchy_loss=config.train.use_prierarchy_loss,
        device=device,
        verbose=1
    )
    model.learn(total_timesteps=1000000000, log_interval=6)


if __name__ == "__main__":
    main()

