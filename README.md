# PRNet inference in C+11

PRNetInfer is a C++11 port of PRNet https://github.com/YadiraF/PRNet using Tensorflow with C API(Inference only).

![](images/earing-result.jpg)

## Dependences

* TensorFlow `r1.8` or later.
  * Optional. TensorFlow lite(`r1.12` or later)
* C++11 compiler.
* CMake 3.5.1.

## Supported platforms

* [x] Ubuntu 16.04
* [x] Windows10 + Visual Studio 2017
  * [ ] MinGW build may work.
* [ ] macOS may work

## Setup TensorFlow

Build TensorFlow with C API supoort.

We recommend to use prebuilt package of TensorFlow for C.

## Build on Linux

Edit TensorFlow for C path in `bootstrap-c.sh`, then

```
$ ./bootstrap-c.sh
$ cd build
$ make
```

Disable DLIB and GUI support in `bootstrap-c.sh` if you don't have dlib and/or X11 installed on your system.

## Build on Windows(Visual Studio)

We recommend to use prebuilt package from

https://github.com/Neargye/hello_tf_c_api

since their prebuilt package contains .lib(import library) which is required for linking in Visual Studio.

Edit TensorFlow for C path in `vcsetup.bat`, then

```
> vcsetup.bat
```

You can use following procedure if you use bash terminal(e.g. git for Windows)

```
$ cmd //c vcsetup.bat
```

### Use dlib

It can automatically detect and crop face region of input image when using dlib.

Clone dlib with git submodule.

```
$ git submodule update --init
```

Then enable `WITH_DLIB` in CMake option.

Face frontalization is only available with dlib build at the moment.


## Prepare freezed model of PRNet

We first need to dump a graph from PRNet.

At `PRnet` repo, in the function of `PosPrediction` in `predictor.py`, add the following code and run `run_basic.py` to get `trained_graph.pb`.

```py

    def predict(self, image):

        vars = {}
        with self.sess:
            for v in tf.trainable_variables():
                vars[v.value().name] = v.eval()

        g_1 = tf.get_default_graph()
        g_2 = tf.Graph()
        consts = {}
        with g_2.as_default():
            with tf.Session() as sess:
                for k in vars.keys():
                    consts[k] = tf.constant(vars[k])

                tf.import_graph_def(g_1.as_graph_def(), input_map=consts, name="")
                tf.train.write_graph(g_2.as_graph_def(), './', 'trained_graph.pb', as_text=False)
        exit()
```


Then freeze it with weight data like a following way.

```
$ bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=../PRNet/trained_graph.pb \
  --input_checkpoint=../PRNet/Data/net-data/256_256_resfcn256_weight \
  --input_binary=true --output_graph=../PRNet/prnet_frozen.pb \
  --output_node_names=resfcn256/Conv2d_transpose_16/Sigmoid
```

## Prepare input image

Copy `tensorflow.dll` to your path.

In `PRNet` repo, setup ASCII representation of `Data/uv-data/canonical_vertices.npy` and save it as `Data/uv-data/canonical_vertices.txt`

### Without dlib build

Prepare 256x256 input image. Input image must contain face area by manual cropping.

Here is an example of creating 256x256 pixel image using ImageMagick.
(Do not forget to append `!` to image extent)

```
$ convert input.png -resize 256x256! image-256x256.png
```

### With dlib build

Face are is automatically detected and cropped using dlib so you can use arbitrary sized image unless dlib can detect a face.

## Run

Run prnet infer like the following.

```
$ ./prnet --graph ../../PRNet/prnet_frozen.pb --data ../../PRNet/Data --image ../input.png
```

* `--image` specifies input image
* `--graph` specifies the freezed graph file.
* `--data` specifies `Data` folder of PRNet repository.

Wavefront .obj file will be written as `output.obj`.

If you build `prnet-infer` with GUI support(`WITH_GUI` in CMake option), you can view resulting mesh.


## TensorFlow lite(experimental)

You may run PRNetInfer on TensorFlow lite(and TensorFlow lite GPU) from `r1.12`.

### Convert forzen model to tflite model.

```
# Assume pip installed tensorflow.
$ tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=prnet_frozen.pb \
  --input_arrays=Placeholder \
  --output_arrays=resfcn256/Conv2d_transpose_16/Sigmoid
```

## TODO

* [x] Use dlib to automatically detect and crop face region.
* [x] Face frontalization(requires dlib)
* [x] Show landmark points.
* [ ] Faster inference using GPU.

## License

PRnetInfer source code is licensed under MIT license. Please see `LICENSE` for details.

* girl_with_earlings-256.jpg : Public domain. https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring

### Third party licenses

* PRNet : MIT license. https://github.com/YadiraF/PRNet
* dlib : Boost Software License. http://dlib.net/
* NanoRT : MIT license. https://github.com/lighttransport/nanort
* ImGui : MIT license. https://github.com/ocornut/imgui
* glfw : zlib/libpng license http://www.glfw.org/
* cxxopts : MIT license. https://github.com/jarro2783/cxxopts

