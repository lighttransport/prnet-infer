#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include "render-config.h"
#include "render-buffer.h"
#include "mesh.h"
#include "image.h"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Set mesh
  void SetMesh(const prnet::Mesh &mesh) {
    mesh_.vertices = mesh.vertices;
    mesh_.faces = mesh.faces;
    mesh_.uvs = mesh.uvs;
  }

  /// Set Image
  void SetImage(const prnet::Image<float> &image) {
    image_ = image;
  }
 
  /// Builds BVH
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(RenderBuffer *buffer, float quat[4],
              const RenderConfig& config);

 private:
  prnet::Mesh mesh_;
  prnet::Image<float> image_;
};
};

#endif  // EXAMPLE_RENDER_H_
