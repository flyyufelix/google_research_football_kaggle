FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /usr/src/app/kaggle_environments

# Conda boost interferes with gfootball
RUN rm -r /opt/conda/lib/cmake/Boost-1.72.0
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install libsdl2-gfx-dev libsdl2-ttf-dev libsdl2-image-dev xorg
RUN cd /tmp && \
    git clone --single-branch --branch v2.6 https://github.com/google-research/football.git && \
    cd football && \
    sed -i 's/copy2/move/g' gfootball/env/observation_processor.py && \
    sed -i 's/os\.remove/# os.remove/g' gfootball/env/observation_processor.py && \
    sed -i 's/except:/except Exception as e:/g' gfootball/env/observation_processor.py && \
    sed -i 's/logging\.error(traceback\.format_exc())/raise e/g' gfootball/env/observation_processor.py && \
    sed -i 's/logging\.info/print/g' gfootball/env/observation_processor.py && \
    pip3 install . && \
    cd /tmp && rm -rf football

# Additional Dependencies
RUN pip3 install stable-baselines3 hydra-core torchtoolbox

COPY train_model.py /usr/src/app/kaggle_environments
COPY yamls /usr/src/app/kaggle_environments/yamls
COPY src /usr/src/app/kaggle_environments/src
COPY scripts /usr/src/app/kaggle_environments/scripts
COPY input /usr/src/app/kaggle_environments/input
COPY Makefile /usr/src/app/kaggle_environments

