#include "face_cropper.h"

#include <cassert>
#include <cmath>
#include <iostream>

#ifdef USE_DLIB
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

namespace prnet {

namespace {

template <typename T>
inline T clamp(T f, T fmin, T fmax) {
  return std::max(std::min(fmax, f), fmin);
}

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
                      int ye, Image<float> *out_img, size_t dst_width,
                      size_t dst_height) {
  size_t width = in_img.getWidth();
  size_t height = in_img.getHeight();
  size_t channels = in_img.getChannels();

  out_img->create(dst_width, dst_height, channels);

  if ((xs == xe) || (ys == ye)) {
    return;
  }

  const float *src = in_img.getData();
  float *dst = out_img->getData();

  for (size_t y = 0; y < dst_height; y++) {
    float v = (ys + 0.5f + (y / float(dst_height)) * (ye - ys + 1)) / float(height);
    for (size_t x = 0; x < dst_width; x++) {
      float u = (xs + 0.5f + (x / float(dst_width)) * (xe - xs + 1)) / float(width);

      float rgba[4];
      FetchTexture(u, v, int(width), int(height), int(channels), src, rgba);

      for (size_t c = 0; c < channels; c++) {
        dst[channels * (y * dst_width + x) + c] = rgba[c];
      }
    }
  }
}

} // anonymous namespace

class FaceCropper::Impl {
public:
  bool crop_dlib(const Image<float>& inp_img, Image<float>& out_img,
                 float* scale, float *shift_x, float *shift_y) {
#ifdef USE_DLIB
    const int width = int(inp_img.getWidth());
    const int height = int(inp_img.getHeight());
    assert(inp_img.getChannels() == 3);

    // Create dlib image
    dlib::array2d<unsigned char> dlib_img(height, width);
    inp_img.foreach ([&](int x, int y, const float *v) {
      // Gray scale
      dlib_img[y][x] = static_cast<uint8_t>(clamp( (0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2]) * 255.0f, 0.0f, 255.0f));
    });

    // Detect
    const std::vector<dlib::rectangle> dets = detector(dlib_img);
    if (0 < dets.size()) {
      const dlib::rectangle &d = dets[0];

      // Crop
      const float left = float(d.left());
      const float right = float(d.right());
      const float top = float(d.top());
      const float bottom = float(d.bottom());
      const float old_size = (right - left + bottom - top) / 2.f;
      const float center[2] =
        {right - (right - left) / 2.f,
         bottom - (bottom - top) / 2.f + old_size * 0.14f};
      const float size = old_size * 1.58f;

      int region[4];
      region[0] = int(center[0] - (size / 2.0f));
      region[1] = int(center[0] + (size / 2.0f));
      region[2] = int(center[1] - (size / 2.0f));
      region[3] = int(center[1] + (size / 2.0f));

      CropImage(inp_img, region[0], region[1], region[2], region[3], &out_img,
                256, 256);

      *scale = size / float(width);
      *shift_x = center[0];
      *shift_y = center[1];

      return true;
    }
#else
    (void)inp_img;
    (void)out_img;
    (void)scale;
    (void)shift_x;
    (void)shift_y;
#endif
    return false;
  }

  bool crop_center(const Image<float>& inp_img, Image<float>& out_img,
                   float* scale, float *shift_x, float *shift_y) {
    const int width = int(inp_img.getWidth());
    const int height = int(inp_img.getHeight());

    // In non dlib path, PRNet crops image from image center with 1/1.6 scaling
    // (minify) then revert it by x1.6 scaling.
    // (See PRNet's api.py::PRN::process for details)
    const float SCALE = 1.6f;
    float center[2] = {width / 2.0f - 0.5f, height / 2.0f - 0.5f};

    int region[4];
    region[0] = int(center[0] - (width / 2.0f) * SCALE);
    region[1] = int(center[0] + (width / 2.0f) * SCALE);
    region[2] = int(center[1] - (height / 2.0f) * SCALE);
    region[3] = int(center[1] + (height / 2.0f) * SCALE);

    std::cout << "region = " << region[0] << ", " << region[1] << ", " << region[2] << ", " << region[3] << std::endl;

    CropImage(inp_img, region[0], region[1], region[2], region[3], &out_img,
              256, 256);

    *scale = SCALE;
    *shift_x = center[0] - ((256.0f / 2.0f) - 0.5f) * SCALE;
    *shift_y = center[1] - ((256.0f / 2.0f) - 0.5f) * SCALE;

    return true;
  }

private:
#ifdef USE_DLIB
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
#endif
};

// PImpl pattern
FaceCropper::FaceCropper() : impl(new Impl()) {}
FaceCropper::~FaceCropper() {}
bool FaceCropper::crop_dlib(const Image<float>& inp_img,
                            Image<float>& out_img, float* scale,
                            float *shift_x, float *shift_y) {
  return impl->crop_dlib(inp_img, out_img, scale, shift_x, shift_y);
}
bool FaceCropper::crop_center(const Image<float>& inp_img,
                              Image<float>& out_img, float* scale,
                              float *shift_x, float *shift_y) {
  return impl->crop_center(inp_img, out_img, scale, shift_x, shift_y);
}

} // namespace prnet
