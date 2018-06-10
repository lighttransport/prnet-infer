#ifndef PRNET_INFER_UI_H_
#define PRNET_INFER_UI_H_

#include "image.h"
#include "mesh.h"

namespace prnet {

bool RunUI(const Mesh &mesh, const Image<float> &input_image,
           const std::vector<Image<float>> &debug_images);

};

#endif // PRNET_INFER_UI_H_
