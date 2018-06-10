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

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace prnet;

template <typename T>
inline T clamp(T f, T fmin, T fmax) {
  return std::max(std::min(fmax, f), fmin);
}

static bool LoadImage(const std::string &filename, Image<float> &image) {
  // Load image
  int width, height, channels;
  unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels,
                                  /* required channels */ 3);
  if (!data) {
    std::cerr << "Failed to load image (" << filename << ")" << std::endl;
    return false;
  }

  // Cast
  image.create(size_t(width), size_t(height), size_t(channels));
  image.foreach ([&](int x, int y, int c, float &v) {
    v = static_cast<float>(data[(y * width + x) * channels + c]) / 255.f;
    // TODO(LTE): Do we really need degamma?
    v = std::pow(v, 2.2f);
  });

  // Free
  stbi_image_free(data);

  return true;
}

static bool SaveImage(const std::string &filename, Image<float> &image, const float scale = 1.0f) {
  const size_t height = image.getHeight();
  const size_t width = image.getWidth();
  const size_t channels = image.getChannels();

  // Cast
  std::vector<unsigned char> data(height * width * channels);
  image.foreach ([&](int x, int y, int c, float &v) {
    data[(size_t(y) * width + size_t(x)) * size_t(channels) + size_t(c)] =
        static_cast<unsigned char>(clamp(scale * v * 255.f, 0.0f, 255.0f));
  });

  // Save
  stbi_write_jpg(filename.c_str(), int(width), int(height), int(channels),
                 &data.at(0), 0);

  return true;
}

// --------------------------------

inline void FilterFloat(float *rgba, const float *image, int i00, int i10,
                        int i01, int i11,
                        float w[4],  // weight
                        int channels) {
  float texel[4][4];

  rgba[0] = rgba[1] = rgba[2] = 0.0f;

  // Filter in linear space
  for (int i = 0; i < channels; i++) {
    texel[0][i] = image[i00 + i];
    texel[1][i] = image[i10 + i];
    texel[2][i] = image[i01 + i];
    texel[3][i] = image[i11 + i];
  }

  for (int i = 0; i < channels; i++) {
    rgba[i] = w[0] * texel[0][i] + w[1] * texel[1][i] + w[2] * texel[2][i] +
              w[3] * texel[3][i];
  }

  if (channels < 4) {
    rgba[3] = 1.0;
  }
}

// Fetch texture with bilinear filtering.
static void FetchTexture(const float u, const float v, int width, int height,
                         int components, const float *image, float *rgba) {
  // clamp to edge
  if ((u < 0.0f) || (u >= 1.0f) || (v < 0.0f) || (v >= 1.0f)) {
    rgba[0] = 0.0f;
    rgba[1] = 0.0f;
    rgba[2] = 0.0f;
    rgba[3] = 0.0f;
    return;
  }

  float sx = std::floor(u);
  float sy = std::floor(v);

  float uu = u - sx;
  float vv = v - sy;

  // clamp
  uu = clamp(uu, 0.0f, 1.0f);
  vv = clamp(vv, 0.0f, 1.0f);

  float px = (width - 1) * uu;
  float py = (height - 1) * vv;

  int x0 = int(px);
  int y0 = int(py);
  int x1 = ((x0 + 1) >= width) ? (width - 1) : (x0 + 1);
  int y1 = ((y0 + 1) >= height) ? (height - 1) : (y0 + 1);

  float dx = px - float(x0);
  float dy = py - float(y0);

  float w[4];

  w[0] = (1.0f - dx) * (1.0f - dy);
  w[1] = (1.0f - dx) * (dy);
  w[2] = (dx) * (1.0f - dy);
  w[3] = (dx) * (dy);

  int i00 = components * (y0 * width + x0);
  int i01 = components * (y0 * width + x1);
  int i10 = components * (y1 * width + x0);
  int i11 = components * (y1 * width + x1);

  FilterFloat(rgba, image, i00, i10, i01, i11, w, components);
}

//
// Crop an image with bilinear filtering.
// pixel bounding box is defined in (xs, ys) - (xe, ye)
// bounding box range is in (0, 0) x (width-1, height-1)
//
static void CropImage(const Image<float> &in_img, int xs, int xe, int ys,
                      int ye, Image<float> *out_img) {
  size_t width = in_img.getWidth();
  size_t height = in_img.getHeight();
  size_t channels = in_img.getChannels();

  out_img->create(width, height, channels);

  if ((xs == xe) || (ys == ye)) {
    return;
  }

  const float *src = in_img.getData();
  float *dst = out_img->getData();

  for (size_t y = 0; y < height; y++) {
    float v = (ys + 0.5f + (y / float(height)) * (ye - ys + 1)) / float(height);
    for (size_t x = 0; x < width; x++) {
      float u = (xs + 0.5f + (x / float(width)) * (xe - xs + 1)) / float(width);

      float rgba[4];
      FetchTexture(u, v, int(width), int(height), int(channels), src, rgba);

      for (size_t c = 0; c < channels; c++) {
        dst[channels * (y * width + x) + c] = rgba[c];
      }
    }
  }
}

// Create texture map from 3D position map
static bool CreateTexture(const Image<float> &image, const Image<float> &posmap,
                          Image<float> *texture) {
  if (image.getWidth() != 256) {
    std::cerr << "Invalid width for Image. width must be 256 but has "
              << image.getWidth() << std::endl;
    return false;
  }

  if (image.getHeight() != 256) {
    std::cerr << "Invalid height for Image. height must be 256 but has "
              << image.getHeight() << std::endl;
    return false;
  }

  if (image.getChannels() != 3) {
    std::cerr << "Invalid channels for Image. channels must be 3 but has "
              << image.getChannels() << std::endl;
    return false;
  }

  if (posmap.getWidth() != 256) {
    std::cerr << "Invalid width for Position map. width must be 256 but has "
              << posmap.getWidth() << std::endl;
    return false;
  }

  if (posmap.getHeight() != 256) {
    std::cerr << "Invalid height for Position map. height must be 256 but has "
              << posmap.getHeight() << std::endl;
    return false;
  }

  if (posmap.getChannels() != 3) {
    std::cerr
        << "Invalid channels for Position map. channels must be 3 but has "
        << posmap.getChannels() << std::endl;
    return false;
  }

  size_t width = image.getWidth();
  size_t height = image.getHeight();

  texture->create(width, height, /* RGB */3);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      // Look up 2D position from 3D position map.
      float vx = posmap.getData()[3 * (y * width + x) + 0];
      float vy = posmap.getData()[3 * (y * width + x) + 1];
      // float b = image.getData()[3 * (y * width + x) + 2];

      // Fetch corresponding texel color
      // No filtering(nearest neighbor point sampling)
      int px = int(vx);
      int py = int(vy);

      if ((px < 0) || (py < 0) || (px >= int(width)) || (py >= int(height))) {
        // out-of-bounds
        texture->getData()[3 * (y * width + x) + 0] = 0.0f;
        texture->getData()[3 * (y * width + x) + 1] = 0.0f;
        texture->getData()[3 * (y * width + x) + 2] = 0.0f;
      } else {
        texture->getData()[3 * (y * width + x) + 0] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 0];
        texture->getData()[3 * (y * width + x) + 1] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 1];
        texture->getData()[3 * (y * width + x) + 2] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 2];
      }
    }
  }

  return true;
}

// Convert 3D position map(Image) to mesh using FaceData.
static bool ConvertToMesh(const Image<float> &image, const FaceData &face_data,
                          Mesh *mesh) {
  if (image.getWidth() != 256) {
    std::cerr << "Invalid width for Image. width must be 256 but has "
              << image.getWidth() << std::endl;
    return false;
  }

  if (image.getHeight() != 256) {
    std::cerr << "Invalid height for Image. height must be 256 but has "
              << image.getHeight() << std::endl;
    return false;
  }

  if (image.getChannels() != 3) {
    std::cerr << "Invalid channels for Image. channels must be 3 but has "
              << image.getChannels() << std::endl;
    return false;
  }

  //// DBG
  // for (size_t i = 0; i < 10; i++) {
  //  float x = image.getData()[3 * i + 0];
  //  float y = image.getData()[3 * i + 1];
  //  float z = image.getData()[3 * i + 2];
  //
  //  std::cout << "[" << i << "] + " << x << ", " << y << ", " << z <<
  //  std::endl;
  //}

  float bmin[3];
  float bmax[3];

  // Look up vertex position from 3D position map(256x256x3)
  mesh->vertices.clear();
  mesh->uvs.clear();
  for (size_t i = 0; i < face_data.face_indices.size(); i++) {
    size_t idx = face_data.face_indices[i];

    uint32_t px = uint32_t(idx % image.getWidth());
    uint32_t py = uint32_t(idx / image.getHeight());

    float x = image.getData()[3 * (py * image.getWidth() + px) + 0];
    float y = image.getData()[3 * (py * image.getWidth() + px) + 1];
    float z = image.getData()[3 * (py * image.getWidth() + px) + 2];

    mesh->vertices.push_back(x);
    mesh->vertices.push_back(y);
    mesh->vertices.push_back(z);

    if (i == 0) {
      bmin[0] = bmax[0] = x;
      bmin[1] = bmax[1] = y;
      bmin[2] = bmax[2] = z;
    } else {
      bmin[0] = std::min(bmin[0], x);
      bmin[1] = std::min(bmin[1], y);
      bmin[2] = std::min(bmin[2], z);

      bmax[0] = std::max(bmax[0], x);
      bmax[1] = std::max(bmax[1], y);
      bmax[2] = std::max(bmax[2], z);
    }

    //normalize uv
    float u = x / float(image.getWidth());
    float v = y / float(image.getHeight());

    mesh->uvs.push_back(u);
    mesh->uvs.push_back(v);
  }

  std::cout << "bmin " << bmin[0] << ", " << bmin[1] << ", " << bmin[2]
            << std::endl;
  std::cout << "bmax " << bmax[0] << ", " << bmax[1] << ", " << bmax[2]
            << std::endl;

#if 1
  // Centerize vertex position.
  {
    float bsize[3];
    bsize[0] = bmax[0] - bmin[0];
    bsize[1] = bmax[1] - bmin[1];
    bsize[2] = bmax[2] - bmin[2];
    for (size_t i = 0; i < mesh->vertices.size() / 3; i++) {
      mesh->vertices[3 * i + 0] -= (bmin[0] + 0.5f * bsize[0]);
      mesh->vertices[3 * i + 1] -= (bmin[1] + 0.5f * bsize[1]);
      mesh->vertices[3 * i + 2] -= (bmin[2] + 0.5f * bsize[2]);
    }
  }
#endif

  for (size_t i = 0; i < face_data.triangles.size(); i++) {
    // It looks triangle index starts with 1, but accepts it.
    uint32_t idx = face_data.triangles[i];
    if (idx >= (mesh->vertices.size() / 3)) {
      std::cerr << "??? : invalid triangle index. " << idx
                << " is greater or equal to " << (mesh->vertices.size() / 3)
                << std::endl;
      exit(-1);
    }
    mesh->faces.push_back(idx);
  }

  return true;
}

// Save as wavefront .obj mesh
static bool SaveAsWObj(const std::string &filename, prnet::Mesh &mesh) {
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open file to write : " << filename << std::endl;
    return false;
  }

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    ofs << "v " << 255.0f * mesh.vertices[3 * i + 0] << " "
        << 255.0f * mesh.vertices[3 * i + 1] << " "
        << 255.0f * mesh.vertices[3 * i + 2] << std::endl;
  }

  for (size_t i = 0; i < mesh.uvs.size() / 2; i++) {
    ofs << "vt " << mesh.uvs[2 * i + 0] << " " << mesh.uvs[2 * i + 1]
        << std::endl;
  }

  for (size_t i = 0; i < mesh.faces.size() / 3; i++) {
    // For .obj, face index starts with 1, so add +1.
    uint32_t f0 = mesh.faces[3 * i + 0] + 1;
    uint32_t f1 = mesh.faces[3 * i + 1] + 1;
    uint32_t f2 = mesh.faces[3 * i + 2] + 1;

    // Assume # of v == # of vt.
    ofs << "f " << f0 << "/" << f0 << " " << f1 << "/" << f1 << " " << f2 << "/"
        << f2 << std::endl;
  }

  // TODO(LTE): Output .mtl file.

  return true;
}

// Restore position coordinate.
static void RestorePositionMap(Image<float> *pos_img, const float crop_scaling_factor, const float xy_offset[2]) {

  size_t n = pos_img->getWidth() * pos_img->getHeight();

  // // DBG
  //  for (size_t i = 0; i < 10; i++) {
  //   float x = pos_img->getData()[3 * i + 0];
  //   float y = pos_img->getData()[3 * i + 1];
  //   float z = pos_img->getData()[3 * i + 2];
  // 
  //   std::cout << "Org [" << i << "] = " << x << ", " << y << ", " << z <<
  //   std::endl;
  // }

  for (size_t i = 0; i < n; i++) {
    float x = pos_img->getData()[3 * i + 0];
    float y = pos_img->getData()[3 * i + 1];
    float z = pos_img->getData()[3 * i + 2];

    pos_img->getData()[3 * i + 0] = x * crop_scaling_factor + xy_offset[0];
    pos_img->getData()[3 * i + 1] = y * crop_scaling_factor + xy_offset[1];
    pos_img->getData()[3 * i + 2] = z * crop_scaling_factor; // TODO(LTE): Do we need z offset?
  }
}

// --------------------------------

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif

int main(int argc, char **argv) {
  cxxopts::Options options("prnet-infer", "PRNet infererence in C++");
  options.add_options()("i,image", "Input image file",
                        cxxopts::value<std::string>())(
      "g,graph", "Input freezed graph file", cxxopts::value<std::string>())(
      "d,data", "Data folder of PRNet repo", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  if (!result.count("image")) {
    std::cerr << "Please specify input image with -i or --image option."
              << std::endl;
    return -1;
  }

  if (!result.count("graph")) {
    std::cerr << "Please specify freezed graph with -g or --graph option."
              << std::endl;
    return -1;
  }

  if (!result.count("data")) {
    std::cerr
        << "Please specify Data folder of PRNet repo with -d or --data option."
        << std::endl;
    return -1;
  }

  std::string image_filename = result["image"].as<std::string>();
  std::string graph_filename = result["graph"].as<std::string>();
  std::string data_dirname = result["data"].as<std::string>();

  // Load image
  std::cout << "Loading image \"" << image_filename << "\"" << std::endl;

  Image<float> inp_img;
  if (!LoadImage(image_filename, inp_img)) {
    return -1;
  }

  // Crop Image.
  Image<float> cropped_img;
  float crop_scaling_factor =
      -1.0f;  // should be filled in the following clause.
  {
    float width = float(inp_img.getWidth());
    float height = float(inp_img.getHeight());

    // In non dlib path, PRNet crops image from image center with x1.6 scaling
    // (minify) then revert it by (1/1.6) scaling.
    // (See PRNet's api.py::PRN::process for details)
    float scale = 1.6f;
    float center[2] = {width / 2.0f - 0.5f, height / 2.0f - 0.5f};

    int region[4];
    region[0] = int(center[0] - (width / 2.0f) / (1.0f / scale));
    region[1] = int(center[0] + (width / 2.0f) / (1.0f / scale));
    region[2] = int(center[1] - (height / 2.0f) / (1.0f / scale));
    region[3] = int(center[1] + (height / 2.0f) / (1.0f / scale));

    std::cout << "region " << region[0] << ", " << region[1] << ", "
              << region[2] << ", " << region[3] << std::endl;

    CropImage(inp_img, region[0], region[1], region[2], region[3],
              &cropped_img);

    SaveImage("dbg_cropped_img.jpg", cropped_img);

    crop_scaling_factor = scale;
  }
  assert(crop_scaling_factor > 0.0f);

  FaceData face_data;

  // Load face data
  if (!LoadFaceData("../Data/uv-data", &face_data)) {
    return -1;
  }

  // Predict
  TensorflowPredictor tf_predictor;
  tf_predictor.init(argc, argv);
  std::cout << "Initialized" << std::endl;
  tf_predictor.load(graph_filename, "Placeholder",
                    "resfcn256/Conv2d_transpose_16/Sigmoid");
  std::cout << "Loaded model" << std::endl;
  Image<float> pos_img;

  std::cout << "Start running network... " << std::endl << std::flush;

  auto startT = std::chrono::system_clock::now();
  tf_predictor.predict(cropped_img, pos_img);
  auto endT = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = endT - startT;
  std::cout << "Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;

  // Restore position map(compensation for cropping operaton)
  float xy_offset[2] = {
      -76.5f, -76.5f};  // HACK. 76.5 = crop factor = (127.5 - -127.5*1.6). This will be changed when using Dlib.

  // kMaxPos comes from `MaxPos` of PosPrediction class in PRNet repo.
  float kMaxPos = pos_img.getWidth() * 1.1f;
  RestorePositionMap(&pos_img, crop_scaling_factor * kMaxPos, xy_offset);

  // => Usually pixel value in `pos_img` now become mostly within [0, 255] range.

  // Save image
  SaveImage("pos.jpg", pos_img, 1.0f / 255.0f);

  Mesh mesh;
  if (!ConvertToMesh(pos_img, face_data, &mesh)) {
    std::cerr << "failed to convert result image to mesh." << std::endl;
    return -1;
  }

  Image<float> texture;
  bool has_texture = CreateTexture(inp_img, pos_img, &texture);
  if (has_texture) {
    SaveImage("texture.jpg", texture); // in linear space.
  }

  SaveAsWObj("output.obj", mesh);

#ifdef USE_GUI
  bool ret = RunUI(mesh, inp_img);
  if (!ret) {
    std::cerr << "failed to run GUI." << std::endl;
  }
#endif

  return 0;
}
