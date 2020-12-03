#!/usr/bin/env python3

import json
import os
import numpy as np

import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.

    See https://arxiv.org/abs/1802.01561.
    """

    def __init__(self, model_config):
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
        self.linear = nn.Linear(3456, 256)

        self.actor = nn.Linear(256, model_config.num_actions)
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

    def get_distribution(self, x):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        return probs

    def get_value(self, x):
        #x = x.permute(0, 3, 1, 2).contiguous()
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

class MLPModel(nn.Module):
    """
    Simple MLP Model
    """

    def __init__(self, model_config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_config.mlp_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, model_config.num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.mlp(x)
        return x

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_distribution(self, x):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        return probs

    def get_value(self, x):
        return self.critic(self.forward(x))

class ResnetAgent(nn.Module):
    """
    Customized Resnet
    """

    def __init__(self, model_config):
         
        super(ResnetAgent, self).__init__()
    
        self.layer1 = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(16, 32, 3, stride=1)),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU())

        self.residual_block_1 = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1, padding=1)))

        self.residual_block_2 = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1, padding=1)))

        self.fc_layer = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.Linear(32 * 34 * 46, 256)), # 32 * 34 * 46 = 50048
            nn.ReLU())

        self.actor = layer_init(nn.Linear(256, model_config.nu_actions), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)
 
    def forward(self, x):
        out = self.layer1(x)
        out = self.residual_block_1(out) + out
        out = self.residual_block_2(out) + out
        out = out.reshape(out.size(0), -1)
        out = self.fc_layer(out)
        return out

    def get_action(self, x, action=None):
        #x = x.permute(0,3,1,2)
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_distribution(self, x):
        #x = x.permute(0,3,1,2)
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        return probs

    def get_value(self, x):
        #x = x.permute(0,3,1,2)
        return self.critic(self.forward(x))

class DeepMLPAgent(nn.Module):
    """
    MLP with more layers
    """

    def __init__(self, model_config):

        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(model_config.mlp_input, 256)),
            nn.Tanh(),
            self._layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            self.layer_init(nn.Linear(256, 1), std=1.),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(model_config.mlp_input, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, model_config.num_actions), std=0.01), # prod to flatten out the action space
        )

    def get_action(self, state, action=None):
        """
        Forward inference to get action
        """
        logits = self.actor(state)

        # Multinomial Distribution (to sample from action spaces with probabilities governed by logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_distribution(self, x):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        return probs

    def get_value(self, state):
        return self.critic(state)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_impala_cnn(model_config):
    model = ImpalaCNN(model_config)
    return model

def get_mlp_float115(model_config):
    model = MLPModel(model_config)
    return model

def get_mlp_relative_pos(model_config):
    model = MLPModel(model_config)
    return model

def get_mlp_lidar_format(model_config):
    model = MLPModel(model_config)
    return model

def get_model(model_config):
    #print("Loading Model: ", model_config.model_name)
    f = globals().get("get_" + model_config.model_name)
    return f(model_config)
