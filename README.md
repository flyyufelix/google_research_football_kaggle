# Google Research Football Kaggle Competition

This repo contains part of the source code I used for [Google Research Football Kaggle Competition](https://www.kaggle.com/c/google-football/). Please refer to [this blog post]() for a detailed account of of my method. 

The source code consists of 2 parts: [imitation learning] and [reinforcement learning]. These are the 2 major components I used for this challenge.  

## Run Instruction ##

### Imitation Learning ### 


First, enter the `imitation_learning` folder and spin up a docker container:

```
$ cd imitation_learning
$ ./build_docker.sh
$ ./run_docker.sh
```

Then, run the following `make` command inside the docker container which will automatically download the historical episodes via kaggle API, perform ETL on the raw episode data, and train the imitation learning model.

```
$ make etl_and_train
```

Please refer to the `Makefile` to see the components of the pipeline.

The configuration variables and hyperparameters can be found in `yamls/config.yaml`. 

### Reinforcement Learning ###

First, enter the `rl_agent` folder and spin up a docker container: 

```
$ cd imitation_learning
$ ./build_docker.sh
$ ./run_docker.sh
```

Then, run the following `make` command inside the docker container which train a PPO agent against a hard coded rule-based bot:

```
$ make train_ppo
```

The configuration variables and PPO hyperparameters can be found in `config.yaml`. Feel free to edit them for your own use.  

Notice that there is NO pretrained model provided in this repo. If you intend to incorporate a pretrained model, change the `load_pretrained` config variable to True in `config.yaml` and add your pretrained model to the `pretrained` folder. 


 
