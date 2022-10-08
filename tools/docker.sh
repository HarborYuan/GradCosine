#!/bin/bash

DATALOC=${DATALOC:-`realpath ../datasets`}
LOGLOC=${LOGLOC:-`realpath ../logger`}
IMG=${IMG:-"harbory/hycls:latest"}

docker run --gpus all -it --rm --ipc=host --net=host \
  --mount src=$(pwd),target=/opt/project,type=bind \
  --mount src=$DATALOC,target=/opt/project/data,type=bind \
  --mount src=$LOGLOC,target=/opt/logger,type=bind \
  $IMG
