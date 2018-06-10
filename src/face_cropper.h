#ifndef FACE_CROPPER_H_180610
#define FACE_CROPPER_H_180610

#include "image.h"

namespace prnet {

class FaceCropper {
public:
  FaceCropper();
  ~FaceCropper();
  bool crop_dlib(const Image<float>& inp_img, Image<float>& out_img,
                 float* scale, float *shift_x, float *shift_y);
  bool crop_center(const Image<float>& inp_img, Image<float>& out_img,
                   float* scale, float *shift_x, float *shift_y);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace prnet

#endif /* end of include guard */
