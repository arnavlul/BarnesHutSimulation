#pragma once
#define GLM_FORCE_CXX14         
#define GLM_FORCE_CUDA
#include <cuda.h>
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

struct KarrasNode {
    int parent = -1;
    int leftChild = -1;
    int rightChild = -1;
    bool isLeaf = false;
    int bodyIndex = -1;
    float totalMass;
    glm::vec3 centerOfMass;
    BoundingBox bbox;
};