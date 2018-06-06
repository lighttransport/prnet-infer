// UI loop

#include "ui.h"
#include "gui/render-buffer.h"
#include "gui/render.h"
#include "gui/trackball.h"

#include "gui/glfw/include/GLFW/glfw3.h"
#include "gui/imgui/imgui.h"
#include "gui/imgui/imgui_impl_glfw_gl2.h"

#include <cmath>
#include <iostream>
#include <mutex>

namespace prnet {

struct UIParameters {
  float showDepthRange[2] = {0.0f, 3.0f};
  bool showDepthPeseudoColor = false;
  int showBufferMode = example::SHOW_BUFFER_COLOR;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

static example::RenderBuffer gRenderBuffer;
static UIParameters gUIParam;

static float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
static float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};

static example::Renderer gRenderer;

static std::atomic<bool> gRenderQuit;
static std::atomic<bool> gRenderRefresh;
static example::RenderConfig gRenderConfig;
static std::mutex gMutex;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

static void RequestRender() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  gRenderRefresh = true;
}

static void RenderThread() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  while (1) {
    if (gRenderQuit) {
      std::cout << "Quit render thread." << std::endl;
      return;
    }

    if (!gRenderRefresh || gRenderConfig.pass >= gRenderConfig.max_passes) {
      // Give some cycles to this thread.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    //auto startT = std::chrono::system_clock::now();

    // Initialize display buffer for the first pass.
    bool initial_pass = false;
    {
      std::lock_guard<std::mutex> guard(gMutex);
      if (gRenderConfig.pass == 0) {
        initial_pass = true;
      }
    }

    bool ret = gRenderer.Render(&gRenderBuffer, gCurrQuat, gRenderConfig);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);

      gRenderConfig.pass++;
    }

    //auto endT = std::chrono::system_clock::now();

    gRenderRefresh = false;

    //std::chrono::duration<double, std::milli> ms = endT - startT;
    // std::cout << ms.count() << " [ms]\n";
  }
}

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

static void Display(int width, int height, int buffer_mode,
             const example::RenderBuffer &buffer) {
  std::vector<float> buf(size_t(width * height * 4));
  if (buffer_mode == example::SHOW_BUFFER_COLOR) {
    // TODO: normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = buffer.rgba[4 * i + 0];
      buf[4 * i + 1] = buffer.rgba[4 * i + 1];
      buf[4 * i + 2] = buffer.rgba[4 * i + 2];
      buf[4 * i + 3] = buffer.rgba[4 * i + 3];
    }
  } else if (buffer_mode == example::SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.normal[i];
    }
  } else if (buffer_mode == example::SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.position[i];
    }
  } else if (buffer_mode == example::SHOW_BUFFER_DEPTH) {
    float d_min =
        std::min(gUIParam.showDepthRange[0], gUIParam.showDepthRange[1]);
    float d_diff =
        std::fabs(gUIParam.showDepthRange[1] - gUIParam.showDepthRange[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size(); i++) {
      float v = (buffer.depth[i] - d_min) / d_diff;
      if (gUIParam.showDepthPeseudoColor) {
        buf[i] = pseudoColor(v, i % 4);
      } else {
        buf[i] = v;
      }
    }
  } else if (buffer_mode == example::SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.texcoord[i];
    }
  } else if (buffer_mode == example::SHOW_BUFFER_DIFFUSE) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = buffer.diffuse[i];
    }
  }

  glRasterPos2i(-1, -1);
  glDrawPixels(width, height, GL_RGBA, GL_FLOAT,
               static_cast<const GLvoid *>(&buf.at(0)));
}

static void HandleUserInput(GLFWwindow *window, const double view_width,
                            const double view_height, double *prev_mouse_x,
                            double *prev_mouse_y) {
  // Handle mouse input
  double mouse_x, mouse_y;
  glfwGetCursorPos(window, &mouse_x, &mouse_y);
  if (int(mouse_x) == int(*prev_mouse_x) &&
      int(mouse_y) == int(*prev_mouse_y)) {
    return;
  }

  int window_width, window_height;
  glfwGetWindowSize(window, &window_width, &window_height);
  //const double width = static_cast<double>(window_width);
  const double height = static_cast<double>(window_height);

  const double kTransScale = 0.005;
  const double kZoomScale = 0.075;

  if (ImGui::IsMouseDown(0)) {  // left mouse button

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
      // T for translation

      gRenderConfig.eye[0] -= kTransScale * (mouse_x - (*prev_mouse_x));
      gRenderConfig.eye[1] -= kTransScale * (mouse_y - (*prev_mouse_y));
      gRenderConfig.look_at[0] -= kTransScale * (mouse_x - (*prev_mouse_x));
      gRenderConfig.look_at[1] -= kTransScale * (mouse_y - (*prev_mouse_y));

      RequestRender();

    } else if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
      // Z for zoom(dolly)

      gRenderConfig.eye[2] += kZoomScale * (mouse_y - (*prev_mouse_y));
      gRenderConfig.look_at[2] += kZoomScale * (mouse_y - (*prev_mouse_y));

      RequestRender();

    } else {
      // No key for rotation

      // Assume render view is located in lower-left.
      double offset_y = height - view_height;

      trackball(gPrevQuat, float((2.0 * (*prev_mouse_x) - view_width) / view_width),
                float((height - 2.0 * ((*prev_mouse_y) - offset_y)) / view_height),
                float((2.0 * mouse_x - view_width) / view_width),
                float((height - 2.0 * (mouse_y - offset_y)) / view_height));
      add_quats(gPrevQuat, gCurrQuat, gCurrQuat);

      RequestRender();
    }
  }

  // Update mouse coordinates
  *prev_mouse_x = mouse_x;
  *prev_mouse_y = mouse_y;
}

bool RunUI(const Mesh &mesh, const Image<float> &input_image) {
  // Setup window
  glfwSetErrorCallback(error_callback);
  if (!glfwInit()) {
    std::cerr << "Failed to initialize glfw" << std::endl;
    return false;
  }
  GLFWwindow *window = glfwCreateWindow(1280, 720, "PRNet infer", nullptr, nullptr);
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
  gRenderConfig.eye[0] = 0.0f;
  gRenderConfig.eye[1] = 0.0f;
  gRenderConfig.eye[2] = 2.0f;

  gRenderConfig.look_at[0] = 0.0f;
  gRenderConfig.look_at[1] = 0.0f;
  gRenderConfig.look_at[2] = 0.0f;

  gRenderConfig.up[0] = 0.0f;
  gRenderConfig.up[1] = 1.0f;
  gRenderConfig.up[2] = 0.0f;

  gRenderConfig.width = 512;
  gRenderConfig.height = 512;

  gRenderConfig.max_passes = 1;

  gRenderBuffer.resize(size_t(gRenderConfig.width), size_t(gRenderConfig.height));

  trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);

  // Setup renderer.
  gRenderer.SetMesh(mesh);
  gRenderer.SetImage(input_image);
  gRenderer.BuildBVH();

  // Launch render thread
  gRenderQuit = false;
  std::thread renderThread(RenderThread);

  // trigger first rendering
  RequestRender();

  // Main loop
  double mouse_x = 0, mouse_y = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplGlfwGL2_NewFrame();

    // Ctrl + q to exit
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS &&
        glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
      // Exit application
      break;
    }

    // space to reset rotation
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
      trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
      gRenderConfig.eye[0] = 0.0f;
      gRenderConfig.eye[1] = 0.0f;
      gRenderConfig.eye[2] = 2.0f;
      gRenderConfig.look_at[0] = 0.0f;
      gRenderConfig.look_at[1] = 0.0f;
      gRenderConfig.look_at[2] = 0.0f;
      gRenderConfig.up[0] = 0.0f;
      gRenderConfig.up[1] = 1.0f;
      gRenderConfig.up[2] = 0.0f;

      RequestRender();
    }

    // Handle user's mouse and key input
    HandleUserInput(window, double(gRenderConfig.width),
                    double(gRenderConfig.height), &mouse_x, &mouse_y);

    // ImGui
    ImGui::Begin("UI");
    {
      ImGui::RadioButton("color", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_COLOR);
      ImGui::SameLine();
      ImGui::RadioButton("normal", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("position", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_POSITION);
      ImGui::SameLine();
      ImGui::RadioButton("depth", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("texcoord", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_TEXCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("diffuse(texture)", &(gUIParam.showBufferMode),
                         example::SHOW_BUFFER_DIFFUSE);

      ImGui::InputFloat2("show depth range", gUIParam.showDepthRange);
      ImGui::Checkbox("show depth pesudo color",
                      &gUIParam.showDepthPeseudoColor);
    }
    ImGui::End();

    // Display rendered image.
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    Display(gRenderConfig.width, gRenderConfig.height, gUIParam.showBufferMode,
            gRenderBuffer);

    // ImGui Display
    // glUseProgram(0); // You may want this if using this code in an OpenGL 3+
    // context where shaders may be bound, but prefer using the GL3+ code.
    ImGui::Render();
    ImGui_ImplGlfwGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  // ImGui::SaveDock();

  gRenderQuit = true;
  renderThread.join();

  ImGui_ImplGlfwGL2_Shutdown();
  ImGui::DestroyContext();
  glfwTerminate();

  return true;
}

}  // namespace prnet
