#pragma once
#include "CudaStructs.h"
#include <vector>           // safe here — only MSVC ever sees this

extern std::vector<BvhNodeTraverse> bvhTraversePool;
extern std::vector<BvhNodeBuild> bvhBuildPool;
extern std::vector<int> globalBodyIndices;

struct Universe {
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz;
    std::vector<float> ax, ay, az;
    std::vector<float> mass;
    std::vector<int> state;
};