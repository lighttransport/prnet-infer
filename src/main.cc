#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

//#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/tensor.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include "tensorflow/core/public/session.h"

#include "cxxopts.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

using namespace tensorflow;

int main(int argc, char** argv) {
  const string checkpointPath = "models/my-model";

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

  std::cout << "New session ..." << std::endl;
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << "New session error : " << status.ToString() << std::endl;
    return -1;
  }

  // Read graph
  std::cout << "Read graph ..." << std::endl;
  MetaGraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), graph_filename.c_str(), &graph_def);
  if (!status.ok()) {
    std::cerr << "Read graph error: " << status.ToString() << std::endl;
    return -1;
  }

  // Add the graph to the session
  std::cout << "Create session ..." << std::endl;
  status = session->Create(graph_def.graph_def());
  if (!status.ok()) {
    std::cerr << "Create session error: " << status.ToString() << std::endl;
    return -1;
  }

  // Read weights from the saved checkpoint
  std::cout << "Read weights from the saved checkpoint ..." << std::endl;
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkpoint_filepath;
  status = session->Run(
      {
          {graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},
      },
      {}, {graph_def.saver_def().restore_op_name()}, nullptr);
  if (!status.ok()) {
    std::cerr << "Read checkpoint failed: " << status.ToString() << std::endl;
    return -1;
  }

  // TODO(syoyo): Implement
  // auto feedDict = ...
  // auto outputOps = ...
  // std::vector<Tensor> outputTensors;
  // status = session->Run(feedDict, outputOps, {}, &outputTensors);

  return 0;
}
