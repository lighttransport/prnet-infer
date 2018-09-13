#include "tf_predictor.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "tensorflow/c/c_api.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <iostream>
#include <fstream>
#include <string>

namespace prnet {

namespace {

void free_buffer(void *data, size_t length) {
  free(data);
}

TF_Buffer *read_file(const std::string &filename) {

  FILE *f = fopen(filename.c_str(), "rb");
  if (!f) {
    std::cerr << "Failed to open file : " << filename << std::endl;
    return nullptr;
  }

  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (fsize < 16) {
    std::cerr << "Invalid data size : " << fsize << std::endl;
    return nullptr;
  }

  void *data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
bool LoadGraph(const std::string& graph_file_name, TF_Session *session) {
  
#if 0 // TODO
  //tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
#endif
  return false;
}

} // anonymous namespace

class TensorflowPredictor::Impl {
public:
  void init(int argc, char* argv[]) {
    std::cout << "TF C API. Version " << TF_Version() << std::endl;
  }

  bool load(const std::string& graph_filename, const std::string& inp_layer,
            const std::string& out_layer) {
    // First we load and initialize the model.
    bool load_graph_status = LoadGraph(graph_filename, session);
    if (!load_graph_status) {
      std::cerr << "Failed to load graph. " << std::endl;
      return false;
    }

    input_layer = inp_layer;
    output_layer = out_layer;

    return true;
  }

  bool predict(const Image<float>& inp_img, Image<float>& out_img) {
#if 0
    // Copy from input image
    Eigen::Index inp_width = static_cast<Eigen::Index>(inp_img.getWidth());
    Eigen::Index inp_height = static_cast<Eigen::Index>(inp_img.getHeight());
    Eigen::Index inp_channels = static_cast<Eigen::Index>(inp_img.getChannels());
    Tensor input_tensor(DT_FLOAT, {1, inp_height, inp_width, inp_channels});
    // TODO: No copy
    std::copy_n(inp_img.getData(), inp_width * inp_height * inp_channels,
                input_tensor.flat<float>().data());

    // Run
    std::vector<Tensor> output_tensors;
    Status run_status = session->Run({{input_layer, input_tensor}},
                                     {output_layer}, {}, &output_tensors);
    if (!run_status.ok()) {
      std::cerr << "Running model failed: " << run_status;
      return false;
    }
    const Tensor& output_tensor = output_tensors[0];

    // Copy to output image
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();
    assert(tensor.dimension(0) == 1);
    size_t out_height = static_cast<size_t>(tensor.dimension(1));
    size_t out_width = static_cast<size_t>(tensor.dimension(2));
    size_t out_channels = static_cast<size_t>(tensor.dimension(3));
    out_img.create(out_width, out_height, out_channels);
    out_img.foreach([&](int x, int y, int c, float& v) {
      v = tensor(0, y, x, c);
    });

    return true;
#else
    return false;
#endif
  }

private:
  //std::unique_ptr<tensorflow::Session> session;
  TF_Session *session;
  std::string input_layer, output_layer;
};

// PImpl pattern
TensorflowPredictor::TensorflowPredictor() : impl(new Impl()) {}
TensorflowPredictor::~TensorflowPredictor() {}
void TensorflowPredictor::init(int argc, char* argv[]) {
  impl->init(argc, argv);
}
bool TensorflowPredictor::load(const std::string& graph_filename,
                               const std::string& inp_layer,
                               const std::string& out_layer) {
  return impl->load(graph_filename, inp_layer, out_layer);
}
bool TensorflowPredictor::predict(const Image<float>& inp_img,
                                  Image<float>& out_img) {
  return impl->predict(inp_img, out_img);
}

} // namespace prnet
