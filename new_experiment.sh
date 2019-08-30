#!/bin/bash

# MOVEME TO PARENT FOLDER OF HANABI PROJ this script creates a new sub-directory containing hanabi project files, starts a containter and builds hanabi env

HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 4)
GPUS=$1
label=$2
name=${USER}_hanabi_${label}_${HASH}

echo "Initializing new project folder ${label}"
git clone git@spork.spawar.navy.mil:mwalton/hanabi-learning-environment.git
mv hanabi-learning-environment ${label}

echo "Launching container named '${name}' with '${GPUS}' GPU"

docker run \
    --gpus $GPUS \
    --name $name \
    -u $(id -u):$(id -g) \
    -v `pwd`/${label}:/hanabi \
    -d \
    -t waltonmyke/hanabi:v2.1 # assumes image origin waltonmyke dockerhub

docker exec ${name} /bin/bash -c 'cd /hanabi ; cmake . ; make'
