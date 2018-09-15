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
#include <cstring>

namespace prnet {

namespace {

void free_buffer(void *data, size_t length) {
  free(data);
}

void nonfree_dealloc_tensor(void *data, size_t length, void *arg) {
  // No need to free memory 
  (void)data;
  (void)length;
  (void)arg;
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
  size_t n = fread(data, fsize, 1, f);
  fclose(f);

  if (n != 1) {
    std::cerr << "Fread error" << std::endl;
    return nullptr;
  }
    
  TF_Buffer *buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;

  return buf;
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
bool LoadGraph(const std::string& graph_file_name, TF_Status *status, TF_Graph **graph, TF_Session **session) {
  
  TF_Buffer *graph_def = read_file(graph_file_name);
  if (graph_def == nullptr) {
    std::cerr << "Failed to read graph file." << std::endl;
    return false;
  }

  TF_SessionOptions *sess_opts = nullptr;

  (*graph) = TF_NewGraph();
  TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef((*graph), graph_def, graph_opts, status);


  bool ret = false;

  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "ERROR: Unable to import graph : " << TF_Message(status) << std::endl;
    goto release;
  }

  std::cout << "Loaded graph file : " << graph_file_name << std::endl;

  sess_opts = TF_NewSessionOptions();
  (*session) = TF_NewSession((*graph), sess_opts, status);
  if (TF_GetCode(status) != TF_OK) {
    std::cerr << "Failed to create Session : " << TF_Message(status) << std::endl;

    goto release;
  }
  
  ret = true;

release:

  TF_DeleteSessionOptions(sess_opts);
  TF_DeleteBuffer(graph_def);
  TF_DeleteImportGraphDefOptions(graph_opts);

  return ret;
}

} // anonymous namespace

class TensorflowPredictor::Impl {
public:
  void init(int argc, char* argv[]) {
    std::cout << "TF C API. Version " << TF_Version() << std::endl;
  }

  void release() {
    if (session != nullptr) {
      TF_CloseSession(session, status);
      TF_DeleteSession(session, status);
      TF_DeleteGraph(graph);
      TF_DeleteStatus(status);
    }
  }

  bool load(const std::string& graph_filename, const std::string& inp_layer,
            const std::string& out_layer) {
    if (status == nullptr) {
      status = TF_NewStatus();
    }

    // First we load and initialize the model.
    bool load_graph_status = LoadGraph(graph_filename, status, &graph, &session);
    if (!load_graph_status) {
      std::cerr << "Failed to load graph from a file : " << graph_filename << std::endl;
      return false;
    }

    input_layer = inp_layer;
    output_layer = out_layer;

    return true;
  }

  bool predict(const Image<float>& inp_img, Image<float>& out_img) {

    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

    // Setup input tensor.

    size_t inp_width = inp_img.getWidth();
    size_t inp_height = inp_img.getHeight();
    size_t inp_channels = inp_img.getChannels();

    std::cout << "input height x width x channels = " << inp_height << " x " << inp_width << " x " << inp_channels << std::endl;

    int64_t input_dims[4] = {1, int64_t(inp_height), int64_t(inp_width), int64_t(inp_channels)};
    size_t input_len = 1 * inp_height * inp_width * inp_channels * sizeof(float);

    std::vector<float> input_buffer;
    input_buffer.resize(input_len / sizeof(float));
    memcpy(input_buffer.data(), inp_img.getData(), input_len);
    
    // Must provide deallocator otherwise null pointer exception will happen when deleting tensor.
    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 4, reinterpret_cast<void *>(const_cast<float *>(input_buffer.data())), input_len, nonfree_dealloc_tensor, /* dealloc_arg */nullptr);
    input_values.push_back(input_tensor);
    
    TF_Operation* input_op = TF_GraphOperationByName(graph, input_layer.c_str());
    TF_Output input_opout = {input_op, 0};

    inputs.push_back(input_opout);

    std::vector<TF_Output> outputs;
    TF_Operation *output_op = TF_GraphOperationByName(graph, output_layer.c_str());
    TF_Output output_opout = {output_op, 0};
    outputs.push_back(output_opout);

    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

    output_values.push_back(nullptr);

    TF_SessionRun(session,
      /* run_options */nullptr,
      /* const TF_Output* inputs */ &inputs[0],
      /* TF_Tensor* const* input_values */ &input_values[0],
      /* int ninputs */ inputs.size(),
      /* const TF_Output* outputs */ &outputs[0],
      /* TF_Tensor** output_values */ &output_values[0],
      /* int noutputs */ outputs.size(),
      /* target_opers */ nullptr,
      /* int ntargets */ 0,
      /* run_metadata */ nullptr,
      /* status */ status);

    if (TF_GetCode(status) != TF_OK) {
      std::cerr << "Failed to run session : " << TF_Message(status) << std::endl;
    }

    float *output_ptr = static_cast<float *>(TF_TensorData(output_values[0]));

    // Copy to output image
    out_img.create(inp_width, inp_height, inp_channels);
    out_img.foreach([&](int x, int y, int c, float& v) {
      v = output_ptr[inp_channels * (y * inp_width + x) + c];
    });

    TF_DeleteTensor(input_tensor);
    // TF_SessionRun will allocate TF_Tensor through TF_Run_Helper() called within TF_SessionRun().
    // So delete output tensor here.
    TF_DeleteTensor(output_values[0]); 


    return true;
  }

private:
  TF_Session *session = nullptr;
  TF_Status *status = nullptr;
  TF_Graph *graph = nullptr;
  TF_Tensor *input_tensor = nullptr;
  std::string input_layer, output_layer;
};

// PImpl pattern
TensorflowPredictor::TensorflowPredictor() : impl(new Impl()) {}
TensorflowPredictor::~TensorflowPredictor() {
  impl->release();
}

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
