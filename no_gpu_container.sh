#!/bin/bash

# this script is for starting a container for debug purposes only; no GPU, can run local or on remote
# similar to run and run_interactive, modified for OSX & VS code compatibility

HASH=$(cat /dev/urandom | env LC_CTYPE=C tr -dc a-zA-Z0-9 | head -c 4)
label=$1
name=${USER}_hanabi_${label}_${HASH}

echo "Launching container named '${name}'"

docker run \
    --name $name \
    -v `pwd`:/hanabi \
    -d \
    -t waltonmyke/hanabi:v2.1 # assumes image origin waltonmyke dockerhub
