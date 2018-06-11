#ifndef PRNET_INFER_MESH_H_
#define PRNET_INFER_MESH_H_

#include <vector>
#include <cstdint>

namespace prnet {

///
/// Simple mesh representation.
///
class Mesh
{
  public:
    Mesh() {}
    Mesh(const Mesh &rhs) {
      vertices = rhs.vertices;
      faces = rhs.faces;
      uvs = rhs.uvs;
    }
    Mesh &operator=(const Mesh &rhs) {
      vertices = rhs.vertices;
      faces = rhs.faces;
      uvs = rhs.uvs;

      return (*this);
    }
    ~Mesh() {}

  std::vector<float> vertices;
  std::vector<uint32_t> faces;  // 3 * # of faces
  std::vector<float> uvs; // per vertex uv

}; 


}; 

#endif // PRNET_INFER_MESH_H_
