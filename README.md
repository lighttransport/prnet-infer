# PRNet inference in C++11

PRNetInfer is a C++11 port of PRNet https://github.com/YadiraF/PRNet in Tensorflow C++ API(Inference only).

![](images/earing-result.jpg)

## Dependences

* TensorFlow `r1.8` or later.
* C++11 compiler.
* CMake 3.5.1.

## Supported platforms

* [x] Ubuntu 16.04
* [ ] Windows and macOS may work

## Setup TensorFlow

Build TensorFlow with C++ API supoort.
TensorFlow Lite does not support enough functionality(e.g. `conv2d_transpose`) to run PRNet model at the moment(`r1.8`), so use ordinal TensorFlow.

Please check out tensorflow `r1.8`, then build `libtensorflow_cc.so` with Bazel.
(CMake and Makefile based build system does not work well)

Please don't forget to specify `monolithic` option. Other build configuration won't work when linked with an user C++ app.

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

## Run

Prepare 256x256 input image. Input image must contain face area by manual cropping(automatically crop face area using dlib is TODO)

Here is an example of creating 256x256 pixel image using ImageMagick.
(Do not forget to append `!` to image extent)

```
$ convert input.png -resize 256x256! image-256x256.png
```

Then, run prnet like the following.

```
$ ./prnet --graph ../../PRNet/prnet_frozen.pb --data ../../PRNet/Data --image ../input.png
```

* `--image` specifies input image(must be 256x256 pixels and contains face region by manual cropping)
* `--graph` specifies the freezed graph file.
* `--data` specifies `Data` folder of PRNet repository.

Wavefront .obj file will be written to `output.obj`.

If you build `prnet-infer` with GUI support(`WITH_GUI` in CMake option), you can view resulting mesh.

## TODO

* [ ] Use dlib to automatically detect and crop face region.
* [ ] Faster inference using GPU.
* [ ] Show landmark points.

## License

PRnetInfer source code is licensed under MIT license. Please see `LICENSE` for details.

* girl_with_earlings-256.jpg : Public domain. https://en.wikipedia.org/wiki/Girl_with_a_Pearl_Earring

### Third party licenses

* PRNet : MIT license. https://github.com/YadiraF/PRNet
* NanoRT : MIT license. https://github.com/lighttransport/nanort
* ImGui : MIT license. https://github.com/ocornut/imgui 
* glfw : zlib/libpng license http://www.glfw.org/
* cxxopts : MIT license. https://github.com/jarro2783/cxxopts

