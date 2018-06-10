#ifndef IMAGE_180602
#define IMAGE_180602

#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <functional>

namespace prnet {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

const static uint32_t DEFAULT_HW_CONCURRENCY =
  std::max(1U, std::thread::hardware_concurrency());

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <typename T>
class Image {
public:
  Image() {}

  void create(size_t w, size_t h, size_t c);
  void create(size_t w, size_t h, size_t c, const T* d);

  size_t getWidth() const { return width; }
  size_t getHeight() const { return height; }
  size_t getChannels() const { return channels; }
  const T* getData() const { return data.data(); }
  T* getData() { return data.data(); }

  const T& fetch(size_t x, size_t y, size_t c = 0) const;
  T& fetch(size_t x, size_t y, size_t c = 0);

  void foreach(const std::function<void(int x, int y ,T* v)> &func,
               uint32_t n_threads = DEFAULT_HW_CONCURRENCY);
  void foreach(const std::function<void(int x, int y, const T* v)> &func,
               uint32_t n_threads = DEFAULT_HW_CONCURRENCY) const;
  void foreach(const std::function<void(int x, int y , int c, T& v)> &func,
               uint32_t n_threads = DEFAULT_HW_CONCURRENCY);
  void foreach(const std::function<void(int x, int y, int c, const T& v)> &func,
               uint32_t n_threads = DEFAULT_HW_CONCURRENCY) const;

private:
  size_t width = 0;
  size_t height = 0;
  size_t channels = 0;
  std::vector<T> data;
};

#include "image_impl.h"

} // namsepace prnet

#endif /* end of include guard */
