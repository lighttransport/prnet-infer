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

#include <iostream>


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

int main(int argc, char** argv) {
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
  tf_predictor.predict(inp_img, out_img);
  std::cout << "ran" << std::endl;

  // Save image
  SaveImage("out.jpg", out_img);

  return 0;
}
