import datetime
import numpy as np

import torch
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
        #self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)
        self.linear = nn.Linear(3456, 256)

        self.actor = nn.Linear(256, model_config.num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        logits = self.actor(x)
        return logits

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

class HybridModel(nn.Module):
    """
    Take stacked SMM and float115_v2 as input
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

        self.mlp = nn.Sequential(
            nn.Linear(model_config.mlp_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mixer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )  

        self.actor = nn.Linear(256, model_config.num_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):

        stacked_smm, float115 = x

        smm_out = stacked_smm.permute(0, 3, 1, 2).contiguous()
        smm_out = self.conv_layers(smm_out)
        smm_out = F.relu(smm_out)
        smm_out = smm_out.view(smm_out.shape[0], -1)
        smm_out = self.linear(smm_out)
        smm_out = F.relu(smm_out)

        float115_out = self.mlp(float115)

        concatenated = torch.cat([smm_out, float115_out], dim=-1)
        mixed = self.mixer(concatenated)

        logits = self.actor(mixed)
        return logits

class MLPModel(nn.Module):
    """
    Take float115_v2 (115 dimension vector) as input
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
        logits = self.actor(x)
        return logits

def get_impala_cnn(model_config):
    model = ImpalaCNN(model_config)
    return model

def get_hybrid(model_config):
    model = HybridModel(model_config)
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
    print("Loading Model: ", model_config.model_name)
    f = globals().get("get_" + model_config.model_name)
    return f(model_config)
