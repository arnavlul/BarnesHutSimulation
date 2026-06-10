#include <cuda.h>
#define GLM_FORCE_CXX14         

#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <glm/glm.hpp>
#include "CudaStructs.h"
#include "SimLaunch.h"

// STEP 3: CUDA Kernels
// ==========================================

__device__ uint64_t expandBits64_d(uint32_t v) {
    uint64_t result = 0;
    for (int i = 0; i < 21; i++) {
        if (v & (1U << i)) result |= (1ull << (3 * i));
    }
    return result;
}

__device__ uint64_t getMortonCode64_d(glm::vec3 pos) {
    float x = fminf(fmaxf(pos.x * 2097152.0f, 0.0f), 2097151.0f);
    float y = fminf(fmaxf(pos.y * 2097152.0f, 0.0f), 2097151.0f);
    float z = fminf(fmaxf(pos.z * 2097152.0f, 0.0f), 2097151.0f);
    uint64_t xx = expandBits64_d((uint32_t)x);
    uint64_t yy = expandBits64_d((uint32_t)y);
    uint64_t zz = expandBits64_d((uint32_t)z);
    return xx * 1 + yy * 2 + zz * 4;
}

__device__ int longestCommonPrefix_d(const uint64_t* mortonCodes, int activeBodyCount, int i, int j) {
    if (j < 0 || j >= activeBodyCount) return -1;
    uint64_t code_i = mortonCodes[i];
    uint64_t code_j = mortonCodes[j];
    if (code_i == code_j) return 64 + __clzll((unsigned long long)i ^ (unsigned long long)j);
    return __clzll(code_i ^ code_j);
}

__global__ void kdkStep1Kernel(glm::vec3* pos, glm::vec3* vel, glm::vec3* acc, int* state, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vel[i].x += acc[i].x * (dt / 2.0f);
        vel[i].y += acc[i].y * (dt / 2.0f);
        vel[i].z += acc[i].z * (dt / 2.0f);
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
}

// ==========================================
// CUSTOM GPU MERGE SORT (O(N log N))
// Replaces Thrust to bypass MSVC CCCL 2.x bug
// ==========================================

__global__ void mergeSortKernel(uint64_t* keysIn, int* valsIn, uint64_t* keysOut, int* valsOut, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int segment = i / (2 * width);
    int start1 = segment * 2 * width;
    int end1 = (start1 + width < n) ? (start1 + width) : n;
    int start2 = end1;
    int end2 = (start2 + width < n) ? (start2 + width) : n;
    
    
    if (i < end1) {
        uint64_t myKey = keysIn[i];
        int l = start2, r = end2;
        while(l < r) {
            int mid = (l + r) / 2;
            if (keysIn[mid] < myKey) l = mid + 1;
            else r = mid;
        }
        int outIdx = (i - start1) + (l - start2) + start1;
        keysOut[outIdx] = myKey;
        valsOut[outIdx] = valsIn[i];
    } else if (i < end2) {
        uint64_t myKey = keysIn[i];
        int l = start1, r = end1;
        while(l < r) {
            int mid = (l + r) / 2;
            if (keysIn[mid] <= myKey) l = mid + 1;
            else r = mid;
        }
        int outIdx = (i - start2) + (l - start1) + start1;
        keysOut[outIdx] = myKey;
        valsOut[outIdx] = valsIn[i];
    }
}

__global__ void generateMortonCodesKernel(glm::vec3* pos, int* state, uint64_t* codes, int* indices, int* activeCount, BoundingBox bbox, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && state[i] == 0) {
        glm::vec3 p = pos[i];
        glm::vec3 norm = (p - bbox.center) / (bbox.halfWidth * 2.0f);
        norm += glm::vec3(0.5f);
        
        int idx = atomicAdd(activeCount, 1);
        codes[idx] = getMortonCode64_d(norm);
        indices[idx] = i;
    }
}

__global__ void initTreeKernel(KarrasNode* pool, int totalNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalNodes) {
        pool[i].parent = -1;
        pool[i].leftChild = -1;
        pool[i].rightChild = -1;
        pool[i].isLeaf = false;
        pool[i].bodyIndex = -1;
        pool[i].totalMass = 0.0f;
    }
}

__global__ void setupLeavesKernel(KarrasNode* pool, glm::vec3* pos, float* mass, int* globalIndices, int internalNodes, int activeCount) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < activeCount) {
        int leafIndex = internalNodes + j;
        pool[leafIndex].isLeaf = true;
        
        int b_idx = globalIndices[j];
        pool[leafIndex].bodyIndex = b_idx;
        pool[leafIndex].totalMass = mass[b_idx];
        pool[leafIndex].centerOfMass = pos[b_idx];
        pool[leafIndex].bbox = BoundingBox(pos[b_idx], 0.0f);
    }
}

__global__ void buildKarrasTreeKernel(uint64_t* codes, KarrasNode* pool, int activeCount, int internalNodes) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < internalNodes) {
        int d = (longestCommonPrefix_d(codes, activeCount, j, j + 1) > longestCommonPrefix_d(codes, activeCount, j, j - 1)) ? 1 : -1;
        int delta_min = longestCommonPrefix_d(codes, activeCount, j, j - d);
        
        int l_max = 2;
        while (longestCommonPrefix_d(codes, activeCount, j, j + l_max * d) > delta_min) l_max *= 2;
        
        int l = 0;
        for (int t = l_max / 2; t >= 1; t /= 2) {
            if (longestCommonPrefix_d(codes, activeCount, j, j + (l + t) * d) > delta_min) l += t;
        }
        int k = j + l * d;
        
        int delta_node = longestCommonPrefix_d(codes, activeCount, j, k);
        int s = 0;
        int t = l;
        do {
            t = (t + 1) >> 1;
            if (longestCommonPrefix_d(codes, activeCount, j, j + (s + t) * d) > delta_node) s += t;
        } while (t > 1);
        
        int min_d = (d < 0) ? d : 0;
        int min_jk = (j < k) ? j : k;
        int max_jk = (j > k) ? j : k;

        int gamma = j + s * d + min_d;
        int left = (min_jk == gamma) ? (internalNodes + gamma) : gamma;
        int right = (max_jk == gamma + 1) ? (internalNodes + gamma + 1) : (gamma + 1);
        
        pool[j].leftChild = left;
        pool[j].rightChild = right;
        pool[left].parent = j;
        pool[right].parent = j;
    }
}

__global__ void computeMassAndBboxKernel(KarrasNode* pool, int* nodeFlags, int activeCount, int internalNodes) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < activeCount) {
        int currentNode = internalNodes + j;
        int parent = pool[currentNode].parent;
        
        while (parent != -1) {
            __threadfence();
            int oldFlag = atomicAdd(&nodeFlags[parent], 1);
            if (oldFlag == 0) break;
            
            int left = pool[parent].leftChild;
            int right = pool[parent].rightChild;
            
            float massLeft = pool[left].totalMass;
            float massRight = pool[right].totalMass;
            pool[parent].totalMass = massLeft + massRight;
            
            if (pool[parent].totalMass > 0.0f) {
                pool[parent].centerOfMass = ((pool[left].centerOfMass * massLeft) + (pool[right].centerOfMass * massRight)) / pool[parent].totalMass;
            }
            
            BoundingBox bboxL = pool[left].bbox;
            BoundingBox bboxR = pool[right].bbox;
            
            float minX = fminf(bboxL.center.x - bboxL.halfWidth, bboxR.center.x - bboxR.halfWidth);
            float minY = fminf(bboxL.center.y - bboxL.halfWidth, bboxR.center.y - bboxR.halfWidth);
            float minZ = fminf(bboxL.center.z - bboxL.halfWidth, bboxR.center.z - bboxR.halfWidth);
            
            float maxX = fmaxf(bboxL.center.x + bboxL.halfWidth, bboxR.center.x + bboxR.halfWidth);
            float maxY = fmaxf(bboxL.center.y + bboxL.halfWidth, bboxR.center.y + bboxR.halfWidth);
            float maxZ = fmaxf(bboxL.center.z + bboxL.halfWidth, bboxR.center.z + bboxR.halfWidth);
            
            glm::vec3 newCenter = glm::vec3((minX + maxX) / 2.0f, (minY + maxY) / 2.0f, (minZ + maxZ) / 2.0f);
            float newHalfWidth = fmaxf(maxX - newCenter.x, fmaxf(maxY - newCenter.y, maxZ - newCenter.z));
            
            pool[parent].bbox = BoundingBox(newCenter, newHalfWidth);
            
            currentNode = parent;
            parent = pool[currentNode].parent;
        }
    }
}

__device__ void calculateForceDevice(int rootIndex, KarrasNode* pool, glm::vec3* pos, glm::vec3* acc, float* mass, int targetBodyIndex, float softeningSq, float theta, float G) {
    int stack[256];
    int stackptr = 0;
    stack[stackptr++] = rootIndex;
    
    glm::vec3 targetPos = pos[targetBodyIndex];
    glm::vec3 totalAcc = glm::vec3(0.0f);

    while (stackptr > 0) {
        int nodeIdx = stack[--stackptr];
        KarrasNode& node = pool[nodeIdx];
        if (node.totalMass == 0.0f) continue;
        
        if (node.isLeaf) {
            int bi = node.bodyIndex;
            if (bi != targetBodyIndex) {
                glm::vec3 direction = pos[bi] - targetPos;
                float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
                float invDistance = rsqrtf(distanceSq + softeningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                totalAcc += direction * (G * mass[bi] * invDistanceCubed);
            }
        } else {
            float width = node.bbox.halfWidth * 2.0f;
            glm::vec3 direction = node.centerOfMass - targetPos;
            float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
            
            if ((width * width) < (theta * theta * distanceSq)) {
                float invDistance = rsqrtf(distanceSq + softeningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                totalAcc += direction * (G * node.totalMass * invDistanceCubed);
            } else {
                if (node.leftChild != -1) stack[stackptr++] = node.leftChild;
                if (node.rightChild != -1) stack[stackptr++] = node.rightChild;
            }
        }
    }
    acc[targetBodyIndex] = totalAcc;
}

__device__ void calculateDriftersForceDevice(int rootIndex, KarrasNode* pool, glm::vec3* pos, glm::vec3* acc, float* mass, int drifterIndex, float softeningSq, float G) {
    if (mass[drifterIndex] == 0.0f) return;
    KarrasNode& root = pool[rootIndex];
    glm::vec3 direction = root.centerOfMass - pos[drifterIndex];
    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    float invDistance = rsqrtf(distanceSq + softeningSq);
    float invDistanceCubed = invDistance * invDistance * invDistance;
    acc[drifterIndex] = direction * (G * root.totalMass * invDistanceCubed);
}

__global__ void calculateForcesKernel(KarrasNode* pool, glm::vec3* pos, glm::vec3* acc, float* mass, int* state, int particleCount, float softeningSq, float theta, float G) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < particleCount) {
        acc[j] = glm::vec3(0.0f);
        if (state[j] == 0) {
            calculateForceDevice(0, pool, pos, acc, mass, j, softeningSq, theta, G);
        } else if (state[j] == 1) {
        calculateDriftersForceDevice(0, pool, pos, acc, mass, j, softeningSq, G);
        }
    }
}

__global__ void kdkStep2Kernel(glm::vec3* vel, glm::vec3* acc, int* state, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vel[i].x += acc[i].x * (dt / 2.0f);
        vel[i].y += acc[i].y * (dt / 2.0f);
        vel[i].z += acc[i].z * (dt / 2.0f);
    }
}

extern "C" void launchKDKStep1(glm::vec3* d_positions, glm::vec3* d_velocities, glm::vec3* d_accelerations, int* d_states, float dt, int particleCount) {
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    kdkStep1Kernel<<<blocks, threads>>>(d_positions, d_velocities, d_accelerations, d_states, dt, particleCount);
}

extern "C" void launchKDKStep2(glm::vec3* d_velocities, glm::vec3* d_accelerations, int* d_states, float dt, int particleCount) {
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    kdkStep2Kernel<<<blocks, threads>>>(d_velocities, d_accelerations, d_states, dt, particleCount);
}

extern "C" int generateMortonAndGetActiveCount(glm::vec3* d_positions, int* d_states, uint64_t* d_mortonCodes, int* d_bodyIndices, int* d_activeCount, BoundingBox bbox, int particleCount) {
    int zero = 0;
    cudaMemcpy(d_activeCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    generateMortonCodesKernel<<<blocks, threads>>>(d_positions, d_states, d_mortonCodes, d_bodyIndices, d_activeCount, bbox, particleCount);
    int activeBodyCount = 0;
    cudaMemcpy(&activeBodyCount, d_activeCount, sizeof(int), cudaMemcpyDeviceToHost);
    return activeBodyCount;
}

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
) {
    int threads = 256;
    if (activeBodyCount > 0) {
        uint64_t* inK = d_mortonCodes;
        int* inV = d_bodyIndices;
        uint64_t* outK = d_mortonCodesTemp;
        int* outV = d_bodyIndicesTemp;
        int sortBlocks = (activeBodyCount + threads - 1) / threads;

        for (int width = 1; width < activeBodyCount; width *= 2) {
            mergeSortKernel<<<sortBlocks, threads>>>(inK, inV, outK, outV, activeBodyCount, width);
            uint64_t* tK = inK; inK = outK; outK = tK;
            int* tV = inV; inV = outV; outV = tV;
        }
        uint64_t* sortedCodes = inK;
        int* sortedIndices = inV;

        int internalNodes = activeBodyCount - 1;
        int totalNodes = 2 * activeBodyCount - 1;

        int treeBlocks = (totalNodes + threads - 1) / threads;
        initTreeKernel<<<treeBlocks, threads>>>(d_karrasPool, totalNodes);

        int leafBlocks = (activeBodyCount + threads - 1) / threads;
        setupLeavesKernel<<<leafBlocks, threads>>>(d_karrasPool, d_positions, d_masses, sortedIndices, internalNodes, activeBodyCount);
        
        if (internalNodes > 0) {
            int internalBlocks = (internalNodes + threads - 1) / threads;
            buildKarrasTreeKernel<<<internalBlocks, threads>>>(sortedCodes, d_karrasPool, activeBodyCount, internalNodes);
            cudaMemset(d_nodeFlags, 0, internalNodes * sizeof(int));
            computeMassAndBboxKernel<<<leafBlocks, threads>>>(d_karrasPool, d_nodeFlags, activeBodyCount, internalNodes);
        }
    }

    int blocks = (particleCount + threads - 1) / threads;
    calculateForcesKernel<<<blocks, threads>>>(d_karrasPool, d_positions, d_accelerations, d_masses, d_states, particleCount, softeningSq, theta, G);
    kdkStep2Kernel<<<blocks, threads>>>(d_velocities, d_accelerations, d_states, dt, particleCount);
}
