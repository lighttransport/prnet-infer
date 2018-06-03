#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
  for (size_t i = 0; i < face_data.face_indices.size() / 2; i++) {

    const float x = image.getData()[3 * i + 0];
    const float y = image.getData()[3 * i + 1];
    const float z = image.getData()[3 * i + 2];

    mesh->vertices.push_back(x);
    mesh->vertices.push_back(y);
    mesh->vertices.push_back(z);
  }

  for (size_t i = 0; i < face_data.triangles.size(); i++) {
    mesh->faces.push_back(i);
  }
  
}

// --------------------------------

int main(int argc, char** argv) {
  
  FaceData face_data;

  // Load face data
  if (!LoadFaceData("../Data/uv-data", &face_data)) {
    return -1;
  }

  // Load image
  Image<float> inp_img;
  if (!LoadImage("../cropped_img.jpg", inp_img)) {
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
    return -1;
  }

#ifdef USE_GUI
  RunUI(mesh, out_img);
#else
  // Save image
  SaveImage("out.jpg", out_img);
#endif

  return 0;
}
