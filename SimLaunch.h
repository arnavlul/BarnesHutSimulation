#pragma once
#include "BarnesHutSimulation.h"

extern "C" void launchKDKStep1(float* d_px, float* d_py, float* d_pz, float* d_vx, float* d_vy, float* d_vz, float* d_ax, float* d_ay, float* d_az, int* d_states, float dt, int particleCount);
extern "C" void launchKDKStep2(float* d_vx, float* d_vy, float* d_vz, float* d_ax, float* d_ay, float* d_az, int* d_states, float dt, int particleCount);

extern "C" int generateMortonAndGetActiveCount(float* d_px, float* d_py, float* d_pz, int* d_states, uint64_t* d_mortonCodes, int* d_bodyIndices, int* d_activeCount, BoundingBox bbox, int particleCount);

extern "C" void launchSimulationStep(
    float* d_px, float* d_py, float* d_pz,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_ax, float* d_ay, float* d_az,
    float* d_masses,
    int* d_states,
    uint64_t* d_mortonCodes,
    int* d_bodyIndices,
    uint64_t* d_mortonCodesTemp,
    int* d_bodyIndicesTemp,
    BvhNodeTraverse* d_trPool, BvhNodeBuild* d_bdPool,
    int* d_activeCount,
    int* d_nodeFlags,
    BoundingBox bbox,
    int activeBodyCount,
    int particleCount,
    float dt,
    float theta,
    float G,
    float softeningSq
);

// FP16 Conversion Kernel for disk saving
extern "C" void launchConvertToHalf(
    float* d_px_in, float* d_py_in, float* d_pz_in,
    float* d_vx_in, float* d_vy_in, float* d_vz_in,
    void* d_px_out, void* d_py_out, void* d_pz_out,
    void* d_vx_out, void* d_vy_out, void* d_vz_out,
    int particleCount, 
    void* stream
);
