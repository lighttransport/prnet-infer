# PRNet inference in C++

PRNetInfer is an inference program of PRNet https://github.com/YadiraF/PRNet in Tensorflow C++ API.


## Dependences

* TensorFlow `r1.8` or later.
* ProtocolBuffer compiler.
* C++11 compiler.
* CMake 3.5.1.

## Setup

Build TensorFlow with C++ API supoort.
(TensorFlow Lite does not support enough functionality(e.g. `conv2d_transpose`) at the moment(`r1.8`), so use ordinal TensorFlow).

Please check out tensorflow `r1.8`, then build `libtensorflow_cc` with Bazel.

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

## Prepare

Copy `Data` directory from `PRNet` repo and put to `prnet-infer` top directory.

## Run

Create freezed model(T.B.W.)



If you disable dlib support, please prepare 256x256 input image.

Here is an example of creating 256x256 pixel image using ImageMagick.

```
$ convert input.png -resize 256x256! image-256x256.png
```


## License

MIT

### Third party licenses

PRNet

* NanoRT : MIT license.
* ImGui : MIT license.
* glfw : zlib/libpng license
