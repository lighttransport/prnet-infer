#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cxxopts.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "tf_predictor.h"

#ifdef USE_GUI
#include "ui.h"
#endif

#include "face-data.h"
#include "mesh.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace prnet;

bool LoadImage(const std::string& filename, Image<float>& image) {
  // Load image
  int width, height, channels;
  unsigned char *data =
    stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (!data) {
    std::cerr << "Failed to load image (" << filename << ")" << std::endl;
    return false;
  }

  // Cast
  image.create(width, height, channels);
  image.foreach([&](int x, int y, int c, float &v) {
      v = static_cast<float>(data[(y * width + x) * channels + c]) / 255.f;
  });

  // Free
  stbi_image_free(data);

  return true;
}

bool SaveImage(const std::string& filename, Image<float>& image) {
  const size_t height = image.getHeight();
  const size_t width = image.getWidth();
  const size_t channels = image.getChannels();

  // Cast
  std::vector<unsigned char> data(height * width * channels);
  image.foreach([&](int x, int y, int c, float &v) {
      data[(y * width + x) * channels + c] =
        static_cast<unsigned char>(v * 255.f);
  });


  // Save
  stbi_write_jpg(filename.c_str(), width, height, channels, &data.at(0), 0);

  return true;
}

// --------------------------------

// Convert 3D position map(Image) to mesh using FaceData.
bool ConvertToMesh(const Image<float> &image, const FaceData &face_data, Mesh *mesh) {
  if (image.getWidth() != 256) {
    std::cerr << "Invalid width for Image. " << std::endl;
    return false;
  }

  if (image.getHeight() != 256) {
    std::cerr << "Invalid height for Image. " << std::endl;
    return false;
  }

  if (image.getChannels() != 3) {
    std::cerr << "Invalid channels for Image. " << std::endl;
    return false;
  }

  // Look up vertex position from 3D position map(256x256x3)
  mesh->vertices.clear();
  for (size_t i = 0; i < face_data.face_indices.size(); i++) {

    size_t idx = face_data.face_indices[i];

    const float x = image.getData()[3 * idx + 0];
    const float y = image.getData()[3 * idx + 1];
    const float z = image.getData()[3 * idx + 2];

    mesh->vertices.push_back(x);
    mesh->vertices.push_back(y);
    mesh->vertices.push_back(z);
  }

  for (size_t i = 0; i < face_data.triangles.size(); i++) {
    // Triangle index starts with 1, so subtract -1
    uint32_t idx = face_data.triangles[i] - 1;
    if (idx >= (mesh->vertices.size() / 3)) {
      std::cerr << "??? : invalid triangle index. " << idx << " is greater or equal to " << (mesh->vertices.size() / 3) << std::endl;
      exit(-1);
    }
    mesh->faces.push_back(idx);
  }
  
  return true;
}

// Save as wavefront .obj mesh
bool SaveAsWObj(const std::string &filename, prnet::Mesh &mesh)
{
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open file to write : " << filename << std::endl;
    return false;
  }

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    ofs << "v " << mesh.vertices[3 * i + 0] << " " << mesh.vertices[3 * i + 1] << " " << mesh.vertices[3 * i + 2] << std::endl;
  }

  for (size_t i = 0; i < mesh.faces.size() / 3; i++) {
    // For .obj, face index starts with 1.
    int f0 = mesh.faces[3 * i + 0] + 1; 
    int f1 = mesh.faces[3 * i + 1] + 1; 
    int f2 = mesh.faces[3 * i + 2] + 1; 
    ofs << "f " << f0 << " " << f1 << " " << f2 << std::endl;
  }

  return true;
}

// --------------------------------

int main(int argc, char** argv) {
  
  cxxopts::Options options("prnet-infer", "PRNet infererence in C++");
  options.add_options()
    ("i,image", "Input image file", cxxopts::value<std::string>())
    ;

  auto result = options.parse(argc, argv);

  if (!result.count("image")) {
    std::cerr << "Please specify input image with -i or --image option." << std::endl;
    return -1;
  }

  std::string image_filename = result["image"].as<std::string>();

  // Load image
  std::cout << "Loading image \"" << image_filename << "\"" << std::endl;

  Image<float> inp_img;
  if (!LoadImage(image_filename, inp_img)) {
    return -1;
  }

  FaceData face_data;

  // Load face data
  if (!LoadFaceData("../Data/uv-data", &face_data)) {
    return -1;
  }


  // Predict
  TensorflowPredictor tf_predictor;
  tf_predictor.init(argc, argv);
  std::cout << "inited" << std::endl;
  tf_predictor.load("../prnet_frozen.pb", "Placeholder",
                    "resfcn256/Conv2d_transpose_16/Sigmoid");
  std::cout << "loaded" << std::endl;
  Image<float> out_img;


  std::cout << "Start running network... " << std::endl << std::flush;

  auto startT = std::chrono::system_clock::now();
  tf_predictor.predict(inp_img, out_img);
  auto endT = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = endT - startT;
  std::cout << "Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;

  Mesh mesh;
  if (!ConvertToMesh(out_img, face_data, &mesh)) {
    std::cerr << "failed to convert result image to mesh." << std::endl;
    return -1;
  }

  SaveAsWObj("output.obj", mesh);

#ifdef USE_GUI
  bool ret = RunUI(mesh, out_img);
  if (!ret) {
    std::cerr << "failed to run GUI." << std::endl;
  }
#else
  // Save image
  SaveImage("out.jpg", out_img);
#endif

  return 0;
}
