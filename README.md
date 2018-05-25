# PRNet inference

## Setup

Build TensorFlow with C++ API supoort.
(TensorFlow Lite does not support enough functionality(e.g. `conv2d_transpose`) at the moment(`r1.8`), so use ordinal TensorFlow.

CMake file for Tensorflow is not well described to create `libtensorflow*` package, So build `libtensorflow_cc.so` using Bazel.

Checkout tensorflow `r1.8`, then build `libtensorflow_cc` with Bazel.

```
$ bazel build -c opt tensorflow:libtensorflow_cc.so
```

## Build

Edit path in `bootstrap.sh`, then

```
$ ./bootstrap.sh
$ cd build
$ make
```

## License

MIT
