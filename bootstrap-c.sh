#!/bin/bash

rm -rf build

# source code directory of tensorflow for C
TF_C_DIR=$HOME/local/tf-c

cmake -DTENSORFLOW_C_DIR=${TF_C_DIR} \
      -DSANITIZE_ADDRESS=Off \
      -DWITH_DLIB=On \
      -DWITH_GUI=On \
      -Bbuild \
      -H.
