#!/bin/bash

# this script is for starting a container for debug purposes only; no GPU, can run local or on remote
# similar to run and run_interactive, modified for OSX & VS code compatibility

HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 4)
GPUS=$1
label=$2
name=${USER}_hanabi_${label}_${HASH}

echo "Launching container named '${name}' with '${GPUS}' GPU"

docker run \
    --gpus $GPUS \
    --name $name \
    -u $(id -u):$(id -g) \
    -v `pwd`:/hanabi \
    -d \
    -t waltonmyke/hanabi:v2.1 # assumes image origin waltonmyke dockerhub
