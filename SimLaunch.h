#pragma once
#include "BarnesHutSimulation.h"

extern "C" void launchKDKStep1(glm::vec3* d_positions, glm::vec3* d_velocities, glm::vec3* d_accelerations, int* d_states, float dt, int particleCount);
extern "C" void launchKDKStep2(glm::vec3* d_velocities, glm::vec3* d_accelerations, int* d_states, float dt, int particleCount);

extern "C" int generateMortonAndGetActiveCount(glm::vec3* d_positions, int* d_states, uint64_t* d_mortonCodes, int* d_bodyIndices, int* d_activeCount, BoundingBox bbox, int particleCount);

extern "C" void launchSimulationStep(
    glm::vec3* d_positions,
    glm::vec3* d_velocities,
    glm::vec3* d_accelerations,
    float* d_masses,
    int* d_states,
    uint64_t* d_mortonCodes,
    int* d_bodyIndices,
    uint64_t* d_mortonCodesTemp,
    int* d_bodyIndicesTemp,
    KarrasNode* d_karrasPool,
    int* d_activeCount,
    int* d_nodeFlags,
    BoundingBox bbox,
    int activeBodyCount,
    int particleCount,
    float dt, float theta, float G, float softeningSq
);
