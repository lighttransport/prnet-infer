#ifndef SH_OPTIMIZER_180616
#define SH_OPTIMIZER_180616

#include "image.h"

#include <array>

namespace prnet {

class ShOptimizer {
public:
  ShOptimizer();
  ~ShOptimizer();

  bool optimize(const Image<float>& rgb_img,
                const Image<float>& pos_img,
                std::array<float, 9> &out_sh_param /* gray scale */);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace prnet


#endif /* end of include guard */
