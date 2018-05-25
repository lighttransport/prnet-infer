cmake_minimum_required(VERSION 3.5.1)

project(PRNetInfer)

# [Build options] -------------------------------------------------------
option(WITH_DLIB "Build with dlib support" OFF)
# -----------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(Sanitizers) # Address sanitizer.

add_definitions("-DEIGEN_AVOID_STL_ARRAY")

# threads
find_package(Threads)

if (WITH_DLIB)
  add_definitions("-DUSE_DLIB=1")
endif (WITH_DLIB)

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

# C++14
set (CMAKE_CXX_STANDARD 14)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    
)

set (CORE_SOURCE
    ${CMAKE_SOURCE_DIR}/src/main.cc
    )

link_directories(${TENSORFLOW_BUILD_DIR})

add_executable( prnet
    ${CORE_SOURCE}
    )

target_include_directories(prnet
    PUBLIC ${TENSORFLOW_DIR}
    # for array_ops.h
    PUBLIC ${TENSORFLOW_DIR}/bazel-genfiles
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    PUBLIC ${TENSORFLOW_BUILD_DIR}
    PUBLIC ${TENSORFLOW_BUILD_DIR}/external/zlib_archive
    PUBLIC ${TENSORFLOW_BUILD_DIR}/external/giflib-5.1.4
    PUBLIC ${TENSORFLOW_BUILD_DIR}/tools/pip_package/build_pip_package.runfiles/org_tensorflow/
    PUBLIC ${TENSORFLOW_BUILD_DIR}/tools/pip_package/build_pip_package.runfiles/eigen_archive
    PUBLIC ${TENSORFLOW_BUILD_DIR}/tools/pip_package/build_pip_package.runfiles/protobuf_archive/src
    PUBLIC ${TENSORFLOW_BUILD_DIR}/tools/pip_package/build_pip_package.runfiles/nsync/public
)

target_link_libraries( prnet
    tensorflow_cc
    tensorflow_framework
    tensorflow
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS})

# Increase warning level for clang.
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(prnet PRIVATE -Weverything -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command)
ENDIF ()

add_sanitizers(prnet)

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
