#include "sh_optimizer.h"

namespace prnet {

class ShOptimizer::Impl {
public:

  bool optimize(const Image<float>& rgb_img,
                const Image<float>& normal_img,
                std::array<float, 9> &out_sh_param) {
    (void) rgb_img;
    (void) normal_img;
    (void) out_sh_param;
    return true;
  }

private:
};

// PImpl pattern
ShOptimizer::ShOptimizer() : impl(new Impl()) {}
ShOptimizer::~ShOptimizer() {}
bool ShOptimizer::optimize(const Image<float>& rgb_img,
                           const Image<float>& normal_img,
                           std::array<float, 9> &out_sh_param) {
  return impl->optimize(rgb_img, normal_img, out_sh_param);
}

} // namespace prnet
