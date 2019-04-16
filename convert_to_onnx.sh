#!/bin/bash

python -m tf2onnx.convert \
    --graphdef prnet_frozen.pb \
    --inputs="Placeholder:0" \
    --outputs="resfcn256/Conv2d_transpose_16/Sigmoid:0" \
    --output prnet.onnx
