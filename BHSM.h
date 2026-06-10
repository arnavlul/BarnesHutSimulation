#pragma once
#include <cstdint>

// Disable struct padding so it perfectly matches the binary file layout
#pragma pack(push, 1)

// The BHSM binary format header (24 bytes)
struct BhsimHeader {
    char magic[4];          // "BHSM"
    int32_t version;        // Format version (2 - SoA format)
    int32_t particleCount;  // Number of particles per frame
    int32_t totalFrames;    // Total frames in the file
    float dt;               // Physics dt used for calculation
    int32_t isFP16;         // 1 if data is FP16, 0 if FP32
};

// FP32 Frame Data Structure
struct BhsimFrameFP32 {
    // Dynamically sized arrays written directly to disk
    // float px[particleCount], py[particleCount], pz[particleCount]
    // float vx[particleCount], vy[particleCount], vz[particleCount]
};

// FP16 Frame Data Structure
struct BhsimFrameFP16 {
    // Dynamically sized arrays written directly to disk
    // half px[particleCount], py[particleCount], pz[particleCount]
    // half vx[particleCount], vy[particleCount], vz[particleCount]
};

#pragma pack(pop)
