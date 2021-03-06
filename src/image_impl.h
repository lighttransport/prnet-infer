template <typename T>
void Image<T>::create(size_t w, size_t h, size_t c) {
  width = w;
  height = h;
  channels = c;
  data.resize(w * h * c);
}

template <typename T>
void Image<T>::create(size_t w, size_t h, size_t c, const T* d) {
  create(w, h, c);
  std::copy(data.begin(), data.end(), d);
}

template <typename T>
const T& Image<T>::fetch(size_t x, size_t y, size_t c) const {
  return data[(y * width + x) * channels + c];
}

template <typename T>
T& Image<T>::fetch(size_t x, size_t y, size_t c) {
  return data[(y * width + x) * channels + c];
}

template <typename T>
void Image<T>::foreach(const std::function<void(size_t, size_t, T*)> &func,
                       uint32_t n_threads) {
  std::vector<std::thread> workers;
  std::atomic<size_t> i(0);
  for (uint32_t t = 0; t < n_threads; t++) {
    workers.emplace_back(std::thread([&]() {
      size_t y = 0;
      while ((y = i++) < height) {
        for (size_t x = 0; x < width; x++) {
          func(x, y, &data[(y * width + x) * channels]);
        }
      }
    }));
  }
  for (auto& t : workers) {
    t.join();
  }
}

template <typename T>
void Image<T>::foreach(
    const std::function<void(size_t, size_t, const T*)> &func,
    uint32_t n_threads) const {
  std::vector<std::thread> workers;
  std::atomic<size_t> i(0);
  for (uint32_t t = 0; t < n_threads; t++) {
    workers.emplace_back(std::thread([&]() {
      size_t y = 0;
      while ((y = i++) < height) {
        for (size_t x = 0; x < width; x++) {
          func(x, y, &data[(y * width + x) * channels]);
        }
      }
    }));
  }
  for (auto& t : workers) {
    t.join();
  }
}

template <typename T>
void Image<T>::foreach(
    const std::function<void(size_t, size_t, size_t, T&)> &func,
    uint32_t n_threads) {
  std::vector<std::thread> workers;
  std::atomic<size_t> i(0);
  for (uint32_t t = 0; t < n_threads; t++) {
    workers.emplace_back(std::thread([&]() {
      size_t y = 0;
      while ((y = i++) < height) {
        for (size_t x = 0; x < width; x++) {
          for (size_t c = 0; c < channels; c++) {
            func(x, y, c, data[(y * width + x) * channels + c]);
          }
        }
      }
    }));
  }
  for (auto& t : workers) {
    t.join();
  }
}

template <typename T>
void Image<T>::foreach(
    const std::function<void(size_t, size_t, size_t, const T&)> &func,
    uint32_t n_threads) const {
  std::vector<std::thread> workers;
  std::atomic<size_t> i(0);
  for (uint32_t t = 0; t < n_threads; t++) {
    workers.emplace_back(std::thread([&]() {
      size_t y = 0;
      while ((y = i++) < height) {
        for (size_t x = 0; x < width; x++) {
          for (size_t c = 0; c < channels; c++) {
            func(x, y, c, data[(y * width + x) * channels + c]);
          }
        }
      }
    }));
  }
  for (auto& t : workers) {
    t.join();
  }
}
