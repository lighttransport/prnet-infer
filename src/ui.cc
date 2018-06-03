// UI loop

#include "ui.h"
#include "gui/render.h"
#include "gui/render-buffer.h"

#include "gui/glfw/include/GLFW/glfw3.h"
#include "gui/imgui/imgui.h"
#include "gui/imgui/imgui_impl_glfw_gl2.h"

#include <iostream>
#include <cmath>

namespace prnet {

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
};

struct UIParameters {
  float showDepthRange[2] = {0.0f, 100.0f};
  bool showDepthPeseudoColor = false;
};

UIParameters gUIParam;

static void error_callback(int error, const char *description) {
  std::cerr << "GLFW Error " << error << ", " << description << std::endl;
}

inline float pseudoColor(float v, int ch) {
  if (ch == 0) {  // red
    if (v <= 0.5f)
      return 0.f;
    else if (v < 0.75f)
      return (v - 0.5f) / 0.25f;
    else
      return 1.f;
  } else if (ch == 1) {  // green
    if (v <= 0.25f)
      return v / 0.25f;
    else if (v < 0.75f)
      return 1.f;
    else
      return 1.f - (v - 0.75f) / 0.25f;
  } else if (ch == 2) {  // blue
    if (v <= 0.25f)
      return 1.f;
    else if (v < 0.5f)
      return 1.f - (v - 0.25f) / 0.25f;
    else
      return 0.f;
  } else {  // alpha
    return 1.f;
  }
}

void Display(int width, int height, int buffer_mode, const RenderBuffer &buffer) {
  std::vector<float> buf(width * height * 4);
  if (buffer_mode == SHOW_BUFFER_COLOR) {
    // TODO: normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = buffer.rgba[4 * i + 0];
      buf[4 * i + 1] = buffer.rgba[4 * i + 1];
      buf[4 * i + 2] = buffer.rgba[4 * i + 2];
      buf[4 * i + 3] = buffer.rgba[4 * i + 3];
    }
  } else if (buffer_mode == SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.normal[i];
    }
  } else if (buffer_mode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.position[i];
    }
  } else if (buffer_mode == SHOW_BUFFER_DEPTH) {
    float d_min = std::min(gUIParam.showDepthRange[0], gUIParam.showDepthRange[1]);
    float d_diff = std::fabs(gUIParam.showDepthRange[1] - gUIParam.showDepthRange[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size(); i++) {
      float v = (buffer.depth[i] - d_min) / d_diff;
      if (gUIParam.showDepthPeseudoColor) {
        buf[i] = pseudoColor(v, i % 4);
      } else {
        buf[i] = v;
      }
    }
  } else if (buffer_mode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.texcoords[i];
    }
  }

  glRasterPos2i(-1, -1);
  glDrawPixels(width, height, GL_RGBA, GL_FLOAT,
               static_cast<const GLvoid*>(&buf.at(0)));

}

bool RunUI(const Mesh &mesh, const Image<float> &image)
{
  // Setup window
  glfwSetErrorCallback(error_callback);
  if (!glfwInit()) return false;
  GLFWwindow *window = glfwCreateWindow(1280, 720, "PRNet infer", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Setup ImGui binding
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  ImGui_ImplGlfwGL2_Init(window, true);

  // Load Dock
  // ImGui::LoadDock();

  io.Fonts->AddFontDefault();

  // Setup style
  ImGui::StyleColorsDark();

  // Setup raytrace renderer;
  


  // Main loop
  double mouse_x = 0, mouse_y = 0;
  ImVec2 layer_uv0(0, 0), layer_uv1(1, 1);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplGlfwGL2_NewFrame();

    // Ctrl + q to exit
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS &&
        glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
      // Exit application
      break;
    }

    //Display(
    
    // ImGui Display
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);
    // glUseProgram(0); // You may want this if using this code in an OpenGL 3+
    // context where shaders may be bound, but prefer using the GL3+ code.
    ImGui::Render();
    ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);

  }

  // Cleanup
  //ImGui::SaveDock();

  ImGui_ImplGlfwGL2_Shutdown();
  ImGui::DestroyContext();
  glfwTerminate();
  
  return true;
}


} // namespace prnet
