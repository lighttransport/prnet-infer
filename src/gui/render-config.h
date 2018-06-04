#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>

namespace example {

typedef struct {
  // framebuffer
  int width = 512;
  int height = 512;

  // camera
  float eye[3] = {0.0f, 0.0f, 100.0f};
  float up[3] = {0.0f, 1.0f, 0.0f};
  float look_at[3] = {0.0f, 0.0f, 0.0f};
  float fov = 45.0f;  // vertical fov in degree.

  // render pass
  int pass = 0;
  int max_passes = 1;

  // Scene info
  float scene_scale = 1.0f;

} RenderConfig;

/// Loads config from JSON file.
//bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
