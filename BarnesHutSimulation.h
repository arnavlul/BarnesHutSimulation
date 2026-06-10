#pragma once
#include "CudaStructs.h"
#include <vector>           // safe here — only MSVC ever sees this

extern std::vector<KarrasNode> karrasPool;
extern std::vector<int> globalBodyIndices;

struct Universe {
    std::vector<glm::vec3> position;
    std::vector<glm::vec3> velocity;
    std::vector<glm::vec3> acceleration;
    std::vector<float> mass;
    std::vector<int> state;
};