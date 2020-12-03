import gym
import json
import os
import argparse
import random
import collections
import numpy as np

import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import gfootball.env as football_env
from gfootball.env import observation_preprocessing, wrappers

device = torch.device('cuda:0')

class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.

    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self):
        super().__init__()
        layers = []
        depth_in = 16
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        #self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)
        self.linear = nn.Linear(3456, 256)

        self.actor = nn.Linear(256, 19) # 19 actions
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x

    def get_action(self, x, action=None):
        #x = x.permute(0, 3, 1, 2).contiguous()
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def predict(self, x):
        x = torch.from_numpy(x).to(device).float().unsqueeze(0)
        x = x.permute(0, 3, 1, 2).contiguous()
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        action = probs.sample()
        action = action.cpu().numpy()[0]
        return action

    def get_value(self, x):
        #x = x.permute(0,3,1,2)
        return self.critic(self.forward(x))


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

# Code to run inference here

stacked_obs = collections.deque([], maxlen=4)

# Load pretrained model
policy = ImpalaCNN().to(device)
model_path = "/usr/src/app/kaggle_environments/adversarial_agents/models/model"

state_dict = torch.load(model_path, map_location="cuda:0")
new_state_dict = collections.OrderedDict()
for key, value in state_dict.items():
    if 'features_extractor' in key:
        new_key = key.replace('features_extractor.','')
    elif 'action_net' in key:
        new_key = key.replace('action_net','actor')
    elif 'value_net' in key:
        new_key = key.replace('value_net','critic')
    else:
        new_key = key
    new_state_dict[new_key] = value

policy.load_state_dict(new_state_dict)

def agent(obs):

    # Obs for first player (i.e. player we control)
    obs = obs['players_raw'][0]
    obs = observation_preprocessing.generate_smm([obs])[0]
    if not stacked_obs:
        stacked_obs.extend([obs] * 4)
    else:
        stacked_obs.append(obs)
    obs = np.concatenate(list(stacked_obs), axis=-1)

    action = policy.predict(obs)

    return [int(action)]


