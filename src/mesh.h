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
    ~Mesh() {}

  std::vector<float> vertices;
  std::vector<uint32_t> faces;

}; 


}; 

#endif // PRNET_INFER_MESH_H_
