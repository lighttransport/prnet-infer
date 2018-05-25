#!/bin/bash

# source code directory of tensorflow
TF_DIR=`pwd`/../tensorflow-cmake

# bazel build directory of tensorflow where `libtensorflow.so` exists.
# Please specify absolute path, otherwise cmake cannot find lib**.a
TF_BUILD_DIR=`pwd`/../tensorflow-cmake/bazel-bin/tensorflow


cmake -DTENSORFLOW_DIR=${TF_DIR} \
      -DTENSORFLOW_BUILD_DIR=${TF_BUILD_DIR} \
      -Bbuild \
      -H.
