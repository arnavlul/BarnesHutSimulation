#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#define CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cub/cub.cuh>
#include "BarnesHutSimulation.h"
__device__ inline uint64_t splitBy3_d(unsigned int a) {
    uint64_t x = a & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

__device__ uint64_t getMortonCode64_d(glm::vec3 pos) {
    pos.x = fminf(fmaxf(pos.x * 2097152.0f, 0.0f), 2097151.0f);
    pos.y = fminf(fmaxf(pos.y * 2097152.0f, 0.0f), 2097151.0f);
    pos.z = fminf(fmaxf(pos.z * 2097152.0f, 0.0f), 2097151.0f);

    uint64_t xx = splitBy3_d((unsigned int)pos.x);
    uint64_t yy = splitBy3_d((unsigned int)pos.y);
    uint64_t zz = splitBy3_d((unsigned int)pos.z);

    return xx * 4 + yy * 2 + zz;
}

__device__ int longestCommonPrefix_d(uint64_t* keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    uint64_t key1 = keys[i];
    uint64_t key2 = keys[j];
    if (key1 == key2) {
        return 64 + __clz(i ^ j);
    }
    return __clzll(key1 ^ key2);
}

__global__ void kdkStep1Kernel(float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* ax, float* ay, float* az, int* state, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (state[i] != -1) {
            vx[i] += ax[i] * (dt / 2.0f);
            vy[i] += ay[i] * (dt / 2.0f);
            vz[i] += az[i] * (dt / 2.0f);
            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;
            pz[i] += vz[i] * dt;
        }
    }
}

__global__ void generateMortonCodesKernel(float* px, float* py, float* pz, int* state, uint64_t* codes, int* indices, int* activeCount, BoundingBox bbox, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (state[i] != -1) {
            glm::vec3 p = glm::vec3(px[i], py[i], pz[i]);
            glm::vec3 norm = (p - bbox.center) / (bbox.halfWidth * 2.0f);
            norm += glm::vec3(0.5f);
            uint64_t code = getMortonCode64_d(norm);
            
            int idx = atomicAdd(activeCount, 1);
            codes[idx] = code;
            indices[idx] = i;
        }
    }
}

__global__ void initTreeKernel(BvhNodeTraverse* trPool, BvhNodeBuild* bdPool, int totalNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < totalNodes) {
        trPool[i].com_x = 0.0f; trPool[i].com_y = 0.0f; trPool[i].com_z = 0.0f;
        trPool[i].mass = 0.0f;
        trPool[i].halfWidth = 0.0f;
        trPool[i].leftChild = -1;
        trPool[i].rightChild = -1;
        trPool[i].bodyIndex = -1;
        bdPool[i].parent = -1;
        bdPool[i].min_x = 0.0f; bdPool[i].min_y = 0.0f; bdPool[i].min_z = 0.0f;
        bdPool[i].max_x = 0.0f; bdPool[i].max_y = 0.0f; bdPool[i].max_z = 0.0f;
    }
}

__global__ void setupLeavesKernel(BvhNodeTraverse* trPool, BvhNodeBuild* bdPool, float* px, float* py, float* pz, float* mass, int* globalIndices, int internalNodes, int activeCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < activeCount) {
        int b_idx = globalIndices[i];
        int leafIndex = internalNodes + i;
        trPool[leafIndex].bodyIndex = b_idx;
        trPool[leafIndex].mass = mass[b_idx];
        trPool[leafIndex].com_x = px[b_idx]; trPool[leafIndex].com_y = py[b_idx]; trPool[leafIndex].com_z = pz[b_idx];
        trPool[leafIndex].halfWidth = 0.0f;
        bdPool[leafIndex].min_x = px[b_idx]; bdPool[leafIndex].min_y = py[b_idx]; bdPool[leafIndex].min_z = pz[b_idx];
        bdPool[leafIndex].max_x = px[b_idx]; bdPool[leafIndex].max_y = py[b_idx]; bdPool[leafIndex].max_z = pz[b_idx];
    }
}

__global__ void buildKarrasTreeKernel(uint64_t* codes, BvhNodeTraverse* trPool, BvhNodeBuild* bdPool, int activeCount, int internalNodes) {
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

        trPool[j].leftChild = left;
        trPool[j].rightChild = right;
        bdPool[left].parent = j;
        bdPool[right].parent = j;
    }
}

__global__ void computeMassAndBboxKernel(BvhNodeTraverse* trPool, BvhNodeBuild* bdPool, int* nodeFlags, int activeCount, int internalNodes) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < activeCount) {
        int currentNode = internalNodes + j;
        int parent = bdPool[currentNode].parent;

        while (parent != -1) {
            __threadfence();
            int oldFlag = atomicAdd(&nodeFlags[parent], 1);
            if (oldFlag == 0) break;

            int left = trPool[parent].leftChild;
            int right = trPool[parent].rightChild;

            float massLeft = trPool[left].mass;
            float massRight = trPool[right].mass;
            trPool[parent].mass = massLeft + massRight;

            if (trPool[parent].mass > 0.0f) {
                trPool[parent].com_x = ((trPool[left].com_x * massLeft) + (trPool[right].com_x * massRight)) / trPool[parent].mass;
                trPool[parent].com_y = ((trPool[left].com_y * massLeft) + (trPool[right].com_y * massRight)) / trPool[parent].mass;
                trPool[parent].com_z = ((trPool[left].com_z * massLeft) + (trPool[right].com_z * massRight)) / trPool[parent].mass;
            }

            float minX = fminf(bdPool[left].min_x, bdPool[right].min_x);
            float minY = fminf(bdPool[left].min_y, bdPool[right].min_y);
            float minZ = fminf(bdPool[left].min_z, bdPool[right].min_z);

            float maxX = fmaxf(bdPool[left].max_x, bdPool[right].max_x);
            float maxY = fmaxf(bdPool[left].max_y, bdPool[right].max_y);
            float maxZ = fmaxf(bdPool[left].max_z, bdPool[right].max_z);

            bdPool[parent].min_x = minX; bdPool[parent].min_y = minY; bdPool[parent].min_z = minZ;
            bdPool[parent].max_x = maxX; bdPool[parent].max_y = maxY; bdPool[parent].max_z = maxZ;

            float newHalfWidth = fmaxf(fmaxf(maxX - minX, maxY - minY), maxZ - minZ) / 2.0f;
            trPool[parent].halfWidth = newHalfWidth;

            currentNode = parent;
            parent = bdPool[currentNode].parent;
        }
    }
}

__device__ void calculateForceDevice(int rootIndex, BvhNodeTraverse* pool, float* px, float* py, float* pz, float* ax, float* ay, float* az, float* mass, int targetBodyIndex, float softeningSq, float theta, float G) {
    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = rootIndex;

    float tpx = px[targetBodyIndex];
    float tpy = py[targetBodyIndex];
    float tpz = pz[targetBodyIndex];
    
    float tax = 0.0f;
    float tay = 0.0f;
    float taz = 0.0f;

    while (stackptr > 0) {
        int nodeIdx = stack[--stackptr];
        
        float4 data0 = __ldg((const float4*)&pool[nodeIdx]);
        float4 data1 = __ldg(((const float4*)&pool[nodeIdx]) + 1);

        float n_com_x = data0.x;
        float n_com_y = data0.y;
        float n_com_z = data0.z;
        float n_mass = data0.w;

        float n_halfWidth = data1.x;
        int n_leftChild = __float_as_int(data1.y);
        int n_rightChild = __float_as_int(data1.z);
        int n_bodyIndex = __float_as_int(data1.w);

        if (n_rightChild == -1) {
            int bi = n_bodyIndex;
            if (bi != targetBodyIndex && n_mass > 0.0f) {
                float dx = n_com_x - tpx;
                float dy = n_com_y - tpy;
                float dz = n_com_z - tpz;
                float distanceSq = dx * dx + dy * dy + dz * dz + softeningSq;
                float invDistance = rsqrtf(distanceSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float factor = G * n_mass * invDistanceCubed;

                tax += dx * factor;
                tay += dy * factor;
                taz += dz * factor;
            }
        } else {
            float dx = n_com_x - tpx;
            float dy = n_com_y - tpy;
            float dz = n_com_z - tpz;
            float distanceSq = dx * dx + dy * dy + dz * dz + softeningSq;
            float width = n_halfWidth * 2.0f;

            if ((width * width) / distanceSq < (theta * theta)) {
                float invDistance = rsqrtf(distanceSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float factor = G * n_mass * invDistanceCubed;

                tax += dx * factor;
                tay += dy * factor;
                taz += dz * factor;
            } else {
                if (n_leftChild != -1) stack[stackptr++] = n_leftChild;
                if (n_rightChild != -1) stack[stackptr++] = n_rightChild;
            }
        }
    }
    
    ax[targetBodyIndex] += tax;
    ay[targetBodyIndex] += tay;
    az[targetBodyIndex] += taz;
}

__device__ void calculateDriftersForceDevice(int rootIndex, BvhNodeTraverse* pool, float* px, float* py, float* pz, float* ax, float* ay, float* az, float* mass, int drifterIndex, float softeningSq, float theta, float G) {
    if (mass[drifterIndex] == 0.0f) return;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = rootIndex;

    float tpx = px[drifterIndex];
    float tpy = py[drifterIndex];
    float tpz = pz[drifterIndex];

    float tax = 0.0f;
    float tay = 0.0f;
    float taz = 0.0f;

    while (stackptr > 0) {
        int nodeIdx = stack[--stackptr];
        
        float4 data0 = __ldg((const float4*)&pool[nodeIdx]);
        float4 data1 = __ldg(((const float4*)&pool[nodeIdx]) + 1);

        float n_com_x = data0.x;
        float n_com_y = data0.y;
        float n_com_z = data0.z;
        float n_mass = data0.w;

        float n_halfWidth = data1.x;
        int n_leftChild = __float_as_int(data1.y);
        int n_rightChild = __float_as_int(data1.z);
        int n_bodyIndex = __float_as_int(data1.w);

        if (n_rightChild == -1) {
            int bi = n_bodyIndex;
            if (bi != drifterIndex && n_mass > 0.0f) {
                float dx = n_com_x - tpx;
                float dy = n_com_y - tpy;
                float dz = n_com_z - tpz;
                float distanceSq = dx * dx + dy * dy + dz * dz + softeningSq;
                float invDistance = rsqrtf(distanceSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float factor = G * n_mass * invDistanceCubed;

                tax += dx * factor;
                tay += dy * factor;
                taz += dz * factor;
            }
        } else {
            float dx = n_com_x - tpx;
            float dy = n_com_y - tpy;
            float dz = n_com_z - tpz;
            float distanceSq = dx * dx + dy * dy + dz * dz + softeningSq;
            float width = n_halfWidth * 2.0f;

            if ((width * width) / distanceSq < (theta * theta)) {
                float invDistance = rsqrtf(distanceSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float factor = G * n_mass * invDistanceCubed;

                tax += dx * factor;
                tay += dy * factor;
                taz += dz * factor;
            } else {
                if (n_leftChild != -1) stack[stackptr++] = n_leftChild;
                if (n_rightChild != -1) stack[stackptr++] = n_rightChild;
            }
        }
    }

    ax[drifterIndex] += tax;
    ay[drifterIndex] += tay;
    az[drifterIndex] += taz;
}

__global__ void calculateForcesKernel(BvhNodeTraverse* pool, float* px, float* py, float* pz, float* ax, float* ay, float* az, float* mass, int* state, int* sortedIndices, int activeCount, float softeningSq, float theta, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < activeCount) {
        int j = sortedIndices[i];
        if (state[j] == 0) {
            calculateForceDevice(0, pool, px, py, pz, ax, ay, az, mass, j, softeningSq, theta, G);
        } else if (state[j] == 1) {
            calculateDriftersForceDevice(0, pool, px, py, pz, ax, ay, az, mass, j, softeningSq, theta, G);
        }
    }
}

__global__ void kdkStep2Kernel(float* vx, float* vy, float* vz, float* ax, float* ay, float* az, int* state, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (state[i] != -1) {
            vx[i] += ax[i] * (dt / 2.0f);
            vy[i] += ay[i] * (dt / 2.0f);
            vz[i] += az[i] * (dt / 2.0f);
        }
    }
}

extern "C" void launchKDKStep1(float* d_px, float* d_py, float* d_pz, float* d_vx, float* d_vy, float* d_vz, float* d_ax, float* d_ay, float* d_az, int* d_states, float dt, int particleCount) {
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    kdkStep1Kernel<<<blocks, threads>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_states, dt, particleCount);
}

extern "C" void launchKDKStep2(float* d_vx, float* d_vy, float* d_vz, float* d_ax, float* d_ay, float* d_az, int* d_states, float dt, int particleCount) {
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    kdkStep2Kernel<<<blocks, threads>>>(d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_states, dt, particleCount);
}

extern "C" int generateMortonAndGetActiveCount(float* d_px, float* d_py, float* d_pz, int* d_states, uint64_t* d_mortonCodes, int* d_bodyIndices, int* d_activeCount, BoundingBox bbox, int particleCount) {
    int zero = 0;
    cudaMemcpy(d_activeCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    generateMortonCodesKernel<<<blocks, threads>>>(d_px, d_py, d_pz, d_states, d_mortonCodes, d_bodyIndices, d_activeCount, bbox, particleCount);
    int activeBodyCount = 0;
    cudaMemcpy(&activeBodyCount, d_activeCount, sizeof(int), cudaMemcpyDeviceToHost);
    return activeBodyCount;
}


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
) {
    int threads = 256;

    if (activeBodyCount > 0) {
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_mortonCodes, d_mortonCodesTemp, d_bodyIndices, d_bodyIndicesTemp, activeBodyCount);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_mortonCodes, d_mortonCodesTemp, d_bodyIndices, d_bodyIndicesTemp, activeBodyCount);
        cudaFree(d_temp_storage);

        uint64_t* sortedCodes = d_mortonCodesTemp;
        int* sortedIndices = d_bodyIndicesTemp;

        int internalNodes = activeBodyCount - 1;
        int totalNodes = 2 * activeBodyCount - 1;

        int treeBlocks = (totalNodes + threads - 1) / threads;
        initTreeKernel<<<treeBlocks, threads>>>(d_trPool, d_bdPool, totalNodes);

        int leafBlocks = (activeBodyCount + threads - 1) / threads;
        setupLeavesKernel<<<leafBlocks, threads>>>(d_trPool, d_bdPool, d_px, d_py, d_pz, d_masses, sortedIndices, internalNodes, activeBodyCount);
        
        if (internalNodes > 0) {
            int internalBlocks = (internalNodes + threads - 1) / threads;
            buildKarrasTreeKernel<<<internalBlocks, threads>>>(sortedCodes, d_trPool, d_bdPool, activeBodyCount, internalNodes);
            cudaMemset(d_nodeFlags, 0, internalNodes * sizeof(int));
            computeMassAndBboxKernel<<<leafBlocks, threads>>>(d_trPool, d_bdPool, d_nodeFlags, activeBodyCount, internalNodes);
        }

        cudaMemset(d_ax, 0, particleCount * sizeof(float));
        cudaMemset(d_ay, 0, particleCount * sizeof(float));
        cudaMemset(d_az, 0, particleCount * sizeof(float));

        int blocks = (activeBodyCount + threads - 1) / threads;
        calculateForcesKernel<<<blocks, threads>>>(d_trPool, d_px, d_py, d_pz, d_ax, d_ay, d_az, d_masses, d_states, sortedIndices, activeBodyCount, softeningSq, theta, G);
    }
    
    int kdkBlocks = (particleCount + threads - 1) / threads;
    kdkStep2Kernel<<<kdkBlocks, threads>>>(d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_states, dt, particleCount);
}

// ==========================================
// FP16 Conversion Kernel
// ==========================================

__global__ void convertToHalfKernel(float* px_in, float* py_in, float* pz_in, float* vx_in, float* vy_in, float* vz_in, half* px_out, half* py_out, half* pz_out, half* vx_out, half* vy_out, half* vz_out, int particleCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < particleCount) {
        px_out[i] = __float2half(px_in[i]);
        py_out[i] = __float2half(py_in[i]);
        pz_out[i] = __float2half(pz_in[i]);
        
        vx_out[i] = __float2half(vx_in[i]);
        vy_out[i] = __float2half(vy_in[i]);
        vz_out[i] = __float2half(vz_in[i]);
    }
}

extern "C" void launchConvertToHalf(float* px_in, float* py_in, float* pz_in, float* vx_in, float* vy_in, float* vz_in, void* px_out, void* py_out, void* pz_out, void* vx_out, void* vy_out, void* vz_out, int particleCount, void* stream) {
    int threads = 256;
    int blocks = (particleCount + threads - 1) / threads;
    cudaStream_t s = (cudaStream_t)stream;
    convertToHalfKernel<<<blocks, threads, 0, s>>>(px_in, py_in, pz_in, vx_in, vy_in, vz_in, (half*)px_out, (half*)py_out, (half*)pz_out, (half*)vx_out, (half*)vy_out, (half*)vz_out, particleCount);
}
