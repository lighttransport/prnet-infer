#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

//#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/protobuf/meta_graph.pb.h"
//#include "tensorflow/core/public/session.h"
//#include "tensorflow/core/public/session_options.h"
//
//#include "tensorflow/cc/saved_model/loader.h"
//
//#include "tensorflow/core/platform/init_main.h"
//#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"


#include "cxxopts.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

using namespace tensorflow;

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

tensorflow::Status LoadModel(std::unique_ptr<tensorflow::Session> *sess, std::string graph_fn, std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  //tensorflow::MetaGraphDef graph_def;
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) {
    std::cerr << "ReadBinaryProto failed." << std::endl;
    std::cerr << status.ToString() << std::endl;
    return status;
  }

  int node_count = graph_def.node_size();
  std::cout << "node_cout " << node_count << std::endl;

#if 0
  // create the graph
  status = (*sess)->Create(graph_def);
  if (status != tensorflow::Status::OK()) {
    std::cerr << "Create the graph failed." << std::endl;
    return status;
  }
#endif

#if 0
  for (int i = 0 ; i < node_count; i++) {
    auto n = graph_def.node(i);
    //if (n.name().find("nWeights") != std::string::npos) {
      std::cout << "name : " << n.name() << std::endl;
    //}
  }
#endif
#if 0
  // restore model from checkpoint, iff checkpoint is given
  if (checkpoint_fn != "") {
    std::cout << "restore..." << std::endl;
    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpoint_fn;

    //tensor_dict feed_dict = {{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
    std::string restore_op_name = "resfcn256/Conv2d_transpose_16/Sigmond";
    tensor_dict feed_dict = {{/* tensor_name */"", checkpointPathTensor}};
    status = sess->Run(feed_dict, {}, {restore_op_name}, nullptr);
    if (status != tensorflow::Status::OK()) {
      std::cerr << "Restore model from checkpoint failed." << std::endl;
      std::cerr << status.ToString() << std::endl;
      return status;
    }
  } else {
  {   
    // virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
    //                  const std::vector<string>& output_tensor_names,
    //                  const std::vector<string>& target_node_names,
    //                  std::vector<Tensor>* outputs) = 0;
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK()) {
      std::cerr << "Run failed." << std::endl;
      return status;
    }
  }
#endif

  sess->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*sess)->Create(graph_def);
  if (!session_create_status.ok()) {
    std::cerr << session_create_status.ToString() << std::endl;
    return session_create_status;
  }

  std::cout << "got it!" << std::endl;

  return tensorflow::Status::OK();
}


#if 0
tensorflow::Status LoadGraph(tensorflow::string graph_filename,
                             std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_filename, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_filename, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return tensorflow::Status::OK();
}
#endif

int main(int argc, char** argv) {
  // std::string graph_filename = "trained_graph.pb";
  // const string checkpointPath = "models/my-model";

  std::string graph_filename;
  std::string checkpoint_filepath;
  bool verbose = false;

  std::vector<Flag> flag_list = {
    Flag("graph", &graph_filename, "graph to be executed")};

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << usage;
    return -1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

#if 0
  cxxopts::Options options("PRNetInfer",
                           "TensorFlow C++ version of PRNet inference");

  options.add_options()("v,verbose", "Verbose output")(
      "f,file", "Model filename", cxxopts::value<std::string>())(
      "g,graph", "Graph filename", cxxopts::value<std::string>())(
      "c,checkpoint", "Checkpoint path", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  if (!result.count("graph")) {
    std::cerr << "Please specify graph filename" << std::endl;
    return EXIT_FAILURE;
  }

  if (!result.count("checkpoint")) {
    std::cerr << "Please specify checkpoint path" << std::endl;
    return EXIT_FAILURE;
  }

  // const std::string model_name = result["file"].as<std::string>();
  const std::string graph_filename = result["graph"].as<std::string>();
  const std::string checkpoint_filepath =
      result["checkpoint"].as<std::string>();
  const bool verbose = result.count("verbose") ? true : false;
#endif

  // std::cout << "New session ..." << std::endl;
  // Session* session;
  // Status status = NewSession(SessionOptions(), &session);
  // if (!status.ok()) {
  //  std::cerr << "New session error : " << status.ToString() << std::endl;
  //  return -1;
  //}

  //{
  //  Status status = tensorflow::NewSession(sess_options, &sess);
  //  if (!status.ok()) {
  //    std::cerr << "New session error : " << status.ToString() << std::endl;
  //    return -1;
  //  }
  //}

  std::unique_ptr<tensorflow::Session> sess;
  //tensorflow::SessionOptions sess_options;
  //TF_CHECK_OK(tensorflow::NewSession(sess_options, &sess));
  Status status = LoadModel(&sess, graph_filename, checkpoint_filepath);

  std::cout << "Loaded graph. " << std::endl;

#if 0
  // Add the graph to the session
  std::cout << "Create session ..." << std::endl;
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
    std::cerr << "Create session error: " << status.ToString() << std::endl;
    return -1;
  }
#endif

  // Read weights from the saved checkpoint
  //std::cout << "Read weights from the saved checkpoint ..." << std::endl;
  //Tensor checkpointPathTensor(DT_STRING, TensorShape());
  //checkpointPathTensor.scalar<std::string>()() = checkpoint_filepath;
#if 0
  TF_CHECK_OK(session->Run(
      {
          {graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},
      },
      {}, {graph_def.saver_def().restore_op_name()}, nullptr));
#endif

  // TODO(syoyo): Implement
  // auto feedDict = ...
  // auto outputOps = ...
  // std::vector<Tensor> outputTensors;
  // status = session->Run(feedDict, outputOps, {}, &outputTensors);

  return 0;
}
