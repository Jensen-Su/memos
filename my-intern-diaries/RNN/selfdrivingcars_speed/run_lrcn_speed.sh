#!/bin/bash

TOOLS=./build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=$PYTHONPATH:./:./examples/selfdrivingcars_speed

GPU_ID=0
WEIGHTS=./models/bvlc_alexnet/bvlc_alexnet.caffemodel

./build/tools/caffe train \
    -solver ./examples/selfdrivingcars_speed/lrcn_speed_solver.prototxt
    # -weights $WEIGHTS \
    # -gpu $GPU_ID
