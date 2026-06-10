#pragma once
#ifdef __CUDACC__
#define GLM_FORCE_CXX14         
#include <cuda.h>
#define GLM_FORCE_CUDA
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif
#include <cuda_runtime.h>
#include <glm/glm.hpp>

const int MAX_TREE_DEPTH = 20;

struct BoundingBox {
    glm::vec3 center;
    float halfWidth;
    __host__ __device__ BoundingBox() { center = glm::vec3(0.0f); halfWidth = 0.0f; }
    __host__ __device__ BoundingBox(glm::vec3 c, float hd) { center = c; halfWidth = hd; }
    __host__ __device__ bool contains(const glm::vec3& point) {
        return ((point.x >= center.x - halfWidth) && (point.x <= center.x + halfWidth) &&
            (point.y >= center.y - halfWidth) && (point.y <= center.y + halfWidth) &&
            (point.z >= center.z - halfWidth) && (point.z <= center.z + halfWidth));
    }
};

struct alignas(32) BvhNodeTraverse {
    float com_x, com_y, com_z;
    float mass;
    float halfWidth;
    int leftChild;
    int rightChild;
    int bodyIndex;
};

struct BvhNodeBuild {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    int parent;
};