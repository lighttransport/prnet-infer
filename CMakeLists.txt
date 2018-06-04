cmake_minimum_required(VERSION 3.5.1)

project(PRNetInfer)

# [Build options] -------------------------------------------------------
option(WITH_DLIB "Build with dlib support" OFF)
option(WITH_GUI "Build with GUI support(for result visualization)" OFF)
# -----------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(Sanitizers) # Address sanitizer.

#add_definitions("-DEIGEN_AVOID_STL_ARRAY")

# threads
find_package(Threads)

if (WITH_DLIB)
  add_definitions("-DUSE_DLIB=1")
endif (WITH_DLIB)

if (WITH_GUI)
  add_definitions("-DUSE_GUI=1")

  set (PRNET_INFER_GUI_SOURCE
      ${CMAKE_SOURCE_DIR}/src/ui.cc
      ${CMAKE_SOURCE_DIR}/src/gui/render.cc
      ${CMAKE_SOURCE_DIR}/src/gui/trackball.cc
      ${CMAKE_SOURCE_DIR}/src/gui/matrix.cc
      ${CMAKE_SOURCE_DIR}/src/gui/imgui/imgui.cpp
      #${CMAKE_SOURCE_DIR}/src/gui/imgui/imgui_dock.cpp
      ${CMAKE_SOURCE_DIR}/src/gui/imgui/imgui_draw.cpp
      ${CMAKE_SOURCE_DIR}/src/gui/imgui/imgui_impl_glfw_gl2.cpp
      )

  set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "" FORCE)
  set(GLFW_INSTALL OFF CACHE INTERNAL "" FORCE)
  add_subdirectory(${CMAKE_SOURCE_DIR}/src/gui/glfw)
  include_directories(${CMAKE_SOURCE_DIR}/src/gui/glfw/include)

endif (WITH_GUI)

# Add custom build type DebugOpt
message("* Adding build types...")
IF (MSVC)
SET(CMAKE_CXX_FLAGS_DEBUGOPT
    "-DDEBUG /DEBUG /O2"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )
SET(CMAKE_C_FLAGS_DEBUGOPT
    "-DDEBUG /DEBUG /O2"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE )
ELSE () # Assume gcc
SET(CMAKE_CXX_FLAGS_DEBUGOPT
    "-g -O2 -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE )
SET(CMAKE_C_FLAGS_DEBUGOPT
    "-g -O2 -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C compiler during coverage builds."
    FORCE )
ENDIF()

SET(CMAKE_EXE_LINKER_FLAGS_DEBUGOPT
    ""
    CACHE STRING "Flags used for linking binaries during coverage builds."
    FORCE )
SET(CMAKE_SHARED_LINKER_FLAGS_DEBUGOPT
    ""
    CACHE STRING "Flags used by the shared libraries linker during coverage builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_DEBUGOPT
    CMAKE_C_FLAGS_DEBUGOPT
    CMAKE_EXE_LINKER_FLAGS_DEBUGOPT
    CMAKE_SHARED_LINKER_FLAGS_DEBUGOPT )

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel DebugOpt."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
message("* Current build type is : ${CMAKE_BUILD_TYPE}")

# C++11
set (CMAKE_CXX_STANDARD 11)

# PIC
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    
)

set (CORE_SOURCE
    ${CMAKE_SOURCE_DIR}/src/main.cc
    ${CMAKE_SOURCE_DIR}/src/tf_predictor.cc
    ${CMAKE_SOURCE_DIR}/src/face-data.cc
    )

link_directories(
    ${TENSORFLOW_BUILD_DIR}
    )

add_executable( prnet
    ${CORE_SOURCE}
    ${PRNET_INFER_GUI_SOURCE}
    )

target_include_directories(prnet
    PUBLIC ${TENSORFLOW_DIR}

    # for array_ops.h
    PUBLIC ${TENSORFLOW_DIR}/bazel-genfiles

    # headers for external packages
    PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/protobuf_archive/src
    PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/eigen_archive
    PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/nsync/public

    # this project
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

if (WITH_GUI)
  set(PRNET_INFER_EXT_LIBS glfw)

  # Required GL libraries on various platforms
  if (WIN32)
    list(APPEND PRNET_INFER_EXT_LIBS opengl32)
  elseif (APPLE)
    find_library(cocoa_library Cocoa)
    find_library(opengl_library OpenGL)
    find_library(corevideo_library CoreVideo)
    find_library(iokit_library IOKit)
    list(APPEND PRNET_INFER_EXT_LIBS ${cocoa_library} ${opengl_library} ${corevideo_library} ${iokit_library})
    #list(APPEND LIBNANOGUI_EXTRA_SOURCE src/darwin.mm)
  elseif(CMAKE_SYSTEM MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "BSD")
    list(APPEND PRNET_INFER_EXT_LIBS GL drm Xxf86vm Xrandr Xinerama Xcursor Xi X11 pthread )
    #if (NOT CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    #  list(APPEND PRNET_INFER_EXT_LIBS rt)
    #endif()
    #if(CMAKE_SYSTEM MATCHES "Linux")
    #  list(APPEND PRNET_INFER_EXT_LIBS dl)
    #endif()
  endif()
endif (WITH_GUI)

target_link_libraries( prnet
    tensorflow_cc
    ${PRNET_INFER_EXT_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    )


# Increase warning level for clang.
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(prnet PRIVATE -Weverything -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command)
ENDIF ()

add_sanitizers(prnet)

# # Tensorflow thing.
# # Fix for "No session factory registered for the given session" error in the runtime.
# if (UNIX AND NOT APPLE)
#   # https://github.com/tensorflow/tensorflow/issues/3308
#   # TARGET_LINK_LIBRARIES(prnet -Wl,--allow-multiple-definition -Wl,--whole-archive "${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so" -Wl,--no-whole-archive)
# endif ()
# 
# if (UNIX AND NOT APPLE)
#    # # Make sure that we don't strip global constructors on Linux.
#    # https://github.com/tensorflow/tensorflow/issues/3308
#    #SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--allow-multiple-definition -Wl,--whole-archive ${CMAKE_EXE_LINKER_FLAGS}")
# endif ()

# [VisualStudio]
if (WIN32)
  # Set `prnet` as a startup project for VS IDE
  if (CMAKE_VERSION VERSION_GREATER 3.6.0)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT prnet)
  endif ()
  
  # For easier debugging in VS IDE(cmake 3.8.0 or later required)
  if (CMAKE_VERSION VERSION_GREATER 3.8.0)
    # Set working directory to $lucia git repo root.
    set_target_properties(prnet PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif ()
endif ()
