#include "face_frontalizer.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#ifdef USE_DLIB
#include <dlib/matrix.h>
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <iostream>

namespace prnet {

void FrontalizeFaceMesh(Mesh *front_mesh, const FaceData &face_data) {
#ifdef USE_DLIB
    const int N_VTX = 43867;  // Defined by PRNet template.

    assert(front_mesh->vertices.size() == N_VTX * 3);

    dlib::matrix<float, N_VTX, 4> vertices_homo;
    for (int i = 0; i < N_VTX; i++) {
        vertices_homo(i, 0) = front_mesh->vertices[3 * size_t(i) + 0];
        vertices_homo(i, 1) = front_mesh->vertices[3 * size_t(i) + 1];
        vertices_homo(i, 2) = front_mesh->vertices[3 * size_t(i) + 2];
        vertices_homo(i, 3) = 1.f;
    }

    dlib::matrix<float, N_VTX, 3> canonical_vertices;
    for (int i = 0; i < N_VTX; i++) {
        canonical_vertices(i, 0) = face_data.canonical_vertices[size_t(i)][0];
        canonical_vertices(i, 1) = face_data.canonical_vertices[size_t(i)][1];
        canonical_vertices(i, 2) = face_data.canonical_vertices[size_t(i)][2];
    }

    // c = P * v
    // c * vt * (v * vt)^-1 = P
    dlib::matrix<float, 4, 3> P = dlib::pinv(vertices_homo) * canonical_vertices;
    dlib::matrix<float, N_VTX, 3> front_vertices = vertices_homo * P;

    for (int i = 0; i < N_VTX; i++) {
        front_mesh->vertices[3 * size_t(i) + 0] = front_vertices(i, 0);
        front_mesh->vertices[3 * size_t(i) + 1] = front_vertices(i, 1);
        front_mesh->vertices[3 * size_t(i) + 2] = front_vertices(i, 2);
    }

#if 1
    // Centerize vertex position.
    {
        float bmin[3];
        float bmax[3];
        for (size_t i = 0; i < N_VTX; i++) {
            float x = front_mesh->vertices[3 * i + 0];
            float y = front_mesh->vertices[3 * i + 1];
            float z = front_mesh->vertices[3 * i + 2];
            if (i == 0) {
                bmin[0] = bmax[0] = x;
                bmin[1] = bmax[1] = y;
                bmin[2] = bmax[2] = z;
            } else {
                bmin[0] = std::min(bmin[0], x);
                bmin[1] = std::min(bmin[1], y);
                bmin[2] = std::min(bmin[2], z);

                bmax[0] = std::max(bmax[0], x);
                bmax[1] = std::max(bmax[1], y);
                bmax[2] = std::max(bmax[2], z);
            }
        }
        float bsize[3];
        bsize[0] = bmax[0] - bmin[0];
        bsize[1] = bmax[1] - bmin[1];
        bsize[2] = bmax[2] - bmin[2];
        for (size_t i = 0; i < front_mesh->vertices.size() / 3; i++) {
            front_mesh->vertices[3 * i + 0] -= (bmin[0] + 0.5f * bsize[0]);
            front_mesh->vertices[3 * i + 1] -= (bmin[1] + 0.5f * bsize[1]);
            front_mesh->vertices[3 * i + 2] -= (bmin[2] + 0.5f * bsize[2]);
        }
    }
#endif

#else
  (void)front_mesh;
  (void)face_data;

  std::cerr << "Face frontalization is not supported in non dlib buid at the momemnt." << std::endl;

#endif
}

} // namespace prnet
