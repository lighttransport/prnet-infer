#ifndef GUI_RENDER_BUFFER_H_
#define GUI_RENDER_BUFFER_H_

#include <vector>

namespace example {

enum ShowBufferMode {
  SHOW_BUFFER_COLOR,
  SHOW_BUFFER_NORMAL,
  SHOW_BUFFER_DEPTH,
  SHOW_BUFFER_POSITION,
  SHOW_BUFFER_TEXCOORD,
};

struct RenderBuffer {
  std::vector<float> rgba;  // 4
  std::vector<float> normal;  // 4 
  std::vector<float> depth;  // 4 
  std::vector<float> position;  // 4 
  std::vector<float> texcoords; // 4

  void resize(int width, int height) {
    rgba.resize(width * height * 4);
    normal.resize(width * height * 4);
    depth.resize(width * height * 4);
    position.resize(width * height * 4);
    texcoords.resize(width * height * 4);
  }
};

} // namespace example

#endif // GUI_RENDER_BUFFER_H_
