#ifndef FACE_FRONTALIZER_H_180610
#define FACE_FRONTALIZER_H_180610

#include "image.h"
#include "face-data.h"
#include "mesh.h"

namespace prnet {

void FrontalizeFaceMesh(Mesh *front_mesh, const FaceData &face_data);

} // namespace prnet

#endif /* end of include guard */

