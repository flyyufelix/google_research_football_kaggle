import logging
import os
import cv2
import time
import datetime
import random
from os.path import join, dirname

import pandas as pd
import numpy as np
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.factories.model_factory import get_model
from src.factories.dataset_factory import get_dataset

from src.utils.path_names import raw_episodes_path, obs_frames_path

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_up(config):
    """
    Set up the environment
    """
    prepare_dir(config)
    set_seed(config.data.seed)

    # Register GPUs to torch so can use parallel training (i.e. DistributedDataParallel)
    #for device in config.core.gpu_id:
    #    torch.cuda.set_device(f"cuda:{device}")
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.core.gpu_id)

@hydra.main(config_name="yamls/config")
def main(config: DictConfig) -> None:

    set_up(config)

    device = torch.device("cuda:"+str(config.core.gpu_id) if torch.cuda.is_available() else "cpu")

    model = get_model(config.model).to(device)

	# Set up Wandb (Pass config variables to wandb)
    if config.log.use_wandb:
        hparams = {}
        for key, value in config.items():
            hparams.update(value)

        wandb.init(project="GRF_imitation_learning", config=hparams)

        # Wandb Magic 
        wandb.watch(model)

    optim = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
    criterion = nn.CrossEntropyLoss()

    # Load Data
    metadata_csv_path = join(dirname(obs_frames_path), "obs_frames_metadata.csv")
    metadata_df = pd.read_csv(metadata_csv_path)

    # Only train on agents with score > score threshold
    if config.data.use_only_left_team:
        if config.data.filter_by_score:
            metadata_df = metadata_df[(metadata_df["left_team"]==1) & (metadata_df["score"]>config.data.score_threshold)]
        elif config.data.filter_by_team:
            metadata_df = metadata_df[(metadata_df["left_team"]==1) & (metadata_df["team_name"]==config.data.team_name)]
        else:
            metadata_df = metadata_df[(metadata_df["left_team"]==1)]
    else:
        if config.data.filter_by_score:
            metadata_df = metadata_df[metadata_df["score"]>config.data.score_threshold]
        elif config.data.filter_by_team:
            metadata_df = metadata_df[metadata_df["team_name"]==config.data.team_name]
    metadata_df = metadata_df.reset_index()

    print("Number of frames: " + str(metadata_df.shape[0]))

    if metadata_df.shape[0] == 0:
        print("No Training Samples to train the model!")
        return 

    train_df, test_df = train_test_split(metadata_df, test_size=0.03, shuffle=False)
    test_df = test_df.reset_index()
    train_dataset = get_dataset(config.data, train_df)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=2)
    test_dataset = get_dataset(config.data, test_df)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=2)

    model_counter = 1

    for epoch in range(config.train.num_epochs):

        start_time = time.time()
        correct = 0
        epoch_loss = 0
        eval_loss = 0

        # Train 
        model.train()
        for x, y in tqdm(train_loader):
            if config.model.model_name == "hybrid":
                smm_input = torch.as_tensor(x[0], device=device, dtype=torch.float32)
                float115_input = torch.as_tensor(x[1], device=device, dtype=torch.float32)
                y = torch.as_tensor(y, device=device, dtype=torch.float32)
                x = (smm_input, float115_input)
            else:
                x = torch.as_tensor(x, device=device, dtype=torch.float32)
                y = torch.as_tensor(y, device=device, dtype=torch.float32)
            optim.zero_grad()
            z = model(x)
            loss = criterion(z, y.long())
            loss.backward()
            optim.step()
            pred = torch.argmax(z, dim=1)
            correct += (pred.cpu() == y.long().cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
            epoch_loss += loss.item()

        # Eval
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                if config.model.model_name == "hybrid":
                    smm_input = torch.as_tensor(x[0], device=device, dtype=torch.float32)
                    float115_input = torch.as_tensor(x[1], device=device, dtype=torch.float32)
                    y = torch.as_tensor(y, device=device, dtype=torch.float32)
                    x = (smm_input, float115_input)
                else:
                    x = torch.as_tensor(x, device=device, dtype=torch.float32)
                    y = torch.as_tensor(y, device=device, dtype=torch.float32)
                z = model(x)
                loss = criterion(z, y.long())
                eval_loss += loss.item()

        print('Epoch {:03}: | Train Loss: {:.3f} | Eval Loss: {:.3f} | Training time: {}'.format(epoch + 1, epoch_loss, eval_loss, str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

        # Better way to save model
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(model.state_dict(), config.store.model_path + "/model_" + str(model_counter))

        if config.log.use_wandb:
            wandb.log({'train loss': epoch_loss, 'eval loss': eval_loss})

            # Upload model artifacts to wandb cloud
            print("Upload model to wandb cloud: " + wandb.run.dir)
            wandb.save(config.store.model_path + "/model_" + str(model_counter)) # Save Pytorch model to wanb local dir and upload to wandb cloud dashboard

        model_counter += 1


if __name__ == "__main__":
    main()
