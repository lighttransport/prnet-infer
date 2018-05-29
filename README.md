# PRNet inference

## Setup

Build TensorFlow with C++ API supoort.
(TensorFlow Lite does not support enough functionality(e.g. `conv2d_transpose`) at the moment(`r1.8`), so use ordinal TensorFlow.

Checkout tensorflow `r1.8`, then build `libtensorflow_cc` with Bazel.

Please specify `monolithic` option. Other build configuration won't work when linked with user C++ app.

```
$ cd $tensorflow
$ git checkout r1.8
$ bazel build --config opt --config monolithic tensorflow/libtensorflow_cc.so
```

## Build

Edit TensorFlow path in `bootstrap.sh`, then

```
$ ./bootstrap.sh
$ cd build
$ make
```

## License

MIT
