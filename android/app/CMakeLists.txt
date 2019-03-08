# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# dlib

if (WITH_DLIB)

  set(DLIB_JPEG_SUPPORT OFF CACHE INTERNAL "" FORCE)
  set(DLIB_PNG_SUPPORT OFF CACHE INTERNAL "" FORCE)
  set(DLIB_GIF_SUPPORT OFF CACHE INTERNAL "" FORCE)
  set(DLIB_USE_MKL_FFT OFF CACHE INTERNAL "" FORCE)
  set(DLIB_LINK_WITH_SQLITE3 OFF CACHE INTERNAL "" FORCE)
  set(DLIB_USE_BLAS OFF CACHE INTERNAL "" FORCE)
  set(DLIB_USE_LAPACK OFF CACHE INTERNAL "" FORCE)
  set(DLIB_USE_CUDA OFF CACHE INTERNAL "" FORCE)

  add_subdirectory(${CMAKE_SOURCE_DIR}/../../src/dlib dlib_build)

  # For some reason, `DLIB_NO_GUI_SUPPORT` is effective after `add_subdirectory`)
  set(DLIB_NO_GUI_SUPPORT ON CACHE INTERNAL "" FORCE)

  include_directories(${CMAKE_SOURCE_DIR}/../../src/dlib)
  add_definitions("-DUSE_DLIB=1")

endif (WITH_DLIB)



# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       dlib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )
