#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef near
#undef far

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstddef>
#include <omp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <math.h>
#include <cstdint>
#include <algorithm>
#include <string>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "BarnesHutSimulation.h"
#include "BHSM.h"
#include <intrin.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "SimLaunch.h"

// At the top of main.cpp, after CUDA includes
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

extern "C" {
    __declspec(dllexport) uint32_t NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

using namespace std;

float THETA = 0.5f;
float SofteningSq = 0.05f * 0.05f;

bool g_enableBloom = true;
float g_bloomExposure = 1.0f;
bool g_enableMotionBlur = true;
float g_motionBlurStrength = 0.85f;

const float G = 1.0f;
float dt = 0.01f;
int particleCount = 50000;

// KarrasNode Pool
vector<BvhNodeTraverse> bvhTraversePool;
vector<BvhNodeBuild> bvhBuildPool;

// Global Indices array
vector<int> globalBodyIndices;

// Camera Variables
float cameraRadius = 600.0f;
float cameraYaw = 45.0f;
float cameraPitch = 30.0f;

bool isDragging = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;

// Morton Code Part
// Spreads 21 bits out across 63 bits
uint64_t expandBits64(uint32_t v) {
    uint64_t result = 0;
    for (int i = 0; i < 21; i++) {
        if (v & (1U << i)) {
            result |= (1ull << (3 * i));
        }
    }
    return result;
}

// Converts 3D position into 64 bit Morton code
uint64_t getMortonCode64(glm::vec3 pos) {
    // 21 Bits can hold values from 0 to 2,097,151
    float x = min(max(pos.x * 2097152.0f, 0.0f), 2097151.0f);
    float y = min(max(pos.y * 2097152.0f, 0.0f), 2097151.0f);
    float z = min(max(pos.z * 2097152.0f, 0.0f), 2097151.0f);

    // Expand
    uint64_t xx = expandBits64((uint32_t)x);
    uint64_t yy = expandBits64((uint32_t)y);
    uint64_t zz = expandBits64((uint32_t)z);

    // Interleave them (Z bit 2, Y bit 1, X bit 0)
    return xx * 1 + yy * 2 + zz * 4;
}

int longestCommonPrefix(const vector<uint64_t>& mortonCodes, int activeBodyCount, int i, int j) {
    if (j < 0 || j >= activeBodyCount) return -1;

    uint64_t code_i = mortonCodes[i];
    uint64_t code_j = mortonCodes[j];

    // If same codes, use key as a fallback
    if (code_i == code_j) return 64 + (int)__lzcnt64((uint64_t)i ^ (uint64_t)j);

    return (int)__lzcnt64(code_i ^ code_j);

}


void frame_buffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // Won't scroll if mouse on UI
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    cameraRadius -= (float)yoffset * 20.0f;
    if (cameraRadius < 10.0f) cameraRadius = 10.0f;
    else if (cameraRadius > 5000.0f) cameraRadius = 5000.0f;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    // Won't click if mouse on UI
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    // Rotate cam when RMB held down
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        float xoffset = (float)(xpos - lastMouseX);
        float yoffset = (float)(lastMouseY - ypos); // Y coordinates go from bottom to top

        lastMouseX = xpos;
        lastMouseY = ypos;

        float sensitivity = 0.3f;
        cameraYaw += xoffset * sensitivity;
        cameraPitch += yoffset * sensitivity;

        // Gimbal lock
        if (cameraPitch > 89.0f) cameraPitch = 89.0f;
        else if (cameraPitch < -89.0f) cameraPitch = -89.0f;
    }
}

void generateUniformUniverse(Universe &universe, float spawnRange) {

    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> posGen(-spawnRange, spawnRange);
    uniform_real_distribution<float> velGen(-1.0, 1.0);
    uniform_real_distribution<float> massGen(1.0, 5.0);

    for (int i = 0; i < particleCount; i++) {
        glm::vec3 position = glm::vec3(posGen(gen), posGen(gen), posGen(gen));
        glm::vec3 velocity = glm::vec3(velGen(gen), velGen(gen), velGen(gen));
        glm::vec3 acceleration = glm::vec3(0.0f);
        float mass = massGen(gen);
       
        universe.px[i] = position.x; universe.py[i] = position.y; universe.pz[i] = position.z;
        universe.vx[i] = velocity.x; universe.vy[i] = velocity.y; universe.vz[i] = velocity.z;
        universe.ax[i] = acceleration.x; universe.ay[i] = acceleration.y; universe.az[i] = acceleration.z;
        universe.mass[i] = mass;
        universe.state[i] = 0;
    }
}

void generateGalaxyUniverse(Universe &universe, float spawnRange) {
    
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);


    random_device rd;
    mt19937 gen(rd());

  
    exponential_distribution<float> radiusDist(3.0f / spawnRange);
    uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);

    normal_distribution<float> heightDist(-spawnRange * 0.02f, spawnRange * 0.02f);

    uniform_real_distribution<float> massDist(1.0f, 5.0f);
    float smbhMass = 100000.0f;


    glm::vec3 smbh_position = glm::vec3(0.0f);
    glm::vec3 smbh_velocity = glm::vec3(0.0f);
    glm::vec3 smbh_acceleration = glm::vec3(0.0f);
    float smbh_mass = smbhMass;
    
    universe.px[0] = smbh_position.x; universe.py[0] = smbh_position.y; universe.pz[0] = smbh_position.z;
        universe.vx[0] = smbh_velocity.x; universe.vy[0] = smbh_velocity.y; universe.vz[0] = smbh_velocity.z;
        universe.ax[0] = smbh_acceleration.x; universe.ay[0] = smbh_acceleration.y; universe.az[0] = smbh_acceleration.z;
    universe.mass[0] = smbhMass;
    universe.state[0] = 0;


    for (int i = 1; i < particleCount; i++) {


        float r = radiusDist(gen);


        float theta = angleDist(gen);
        float y = heightDist(gen);

        glm::vec3 position = glm::vec3(r * cos(theta), y, r * sin(theta));


        // v = sqrt(G * M / r) orbital velocity
        float orbitalVelocity = sqrt((G * smbhMass) / r);

        // Calculate the Tangent Vector (perpendicular to the radius in the X-Z plane)
        // If position is (x, z), tangent is (-z, x)
        glm::vec3 tangent = glm::normalize(glm::vec3(-position.z, 0.0f, position.x));

        // Add 5% random velocity noise (Velocity Dispersion)
        // This prevents the galaxy from looking like a perfect, artificial DVD ring 
        // and gives the Hermite solver organic clumping to work with.
        normal_distribution<float> noiseDist(0.0f, orbitalVelocity * 0.05f);
        glm::vec3 velocityNoise = glm::vec3(noiseDist(gen), noiseDist(gen), noiseDist(gen));

        glm::vec3 velocity = (tangent * orbitalVelocity) + velocityNoise;

        // 3. Zero out ALL integration memory for GPU safety
        glm::vec3 acceleration = glm::vec3(0.0f);

        float mass = massDist(gen);

        universe.px[i] = position.x; universe.py[i] = position.y; universe.pz[i] = position.z;
        universe.vx[i] = velocity.x; universe.vy[i] = velocity.y; universe.vz[i] = velocity.z;
        universe.ax[i] = acceleration.x; universe.ay[i] = acceleration.y; universe.az[i] = acceleration.z;
        universe.mass[i] = mass;
        universe.state[i] = 0;

    }
}

void generateClusterThreeBody(Universe &universe, float spawnRange) {
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    int third = particleCount / 3;
    normal_distribution<float> clusterDist(0.0f, spawnRange * 0.15f); // tight clusters
    uniform_real_distribution<float> massDist(1.0f, 5.0f);

    float smbhMass = 50000.0f;

    glm::vec3 centers[3] = {
        glm::vec3(200.0f, 0.0f, 0.0f),
        glm::vec3(-100.0f, 173.2f, 0.0f),
        glm::vec3(-100.0f, -173.2f, 0.0f)
    };

    glm::vec3 velocities[3] = {
        glm::vec3(0.0f, 15.0f, 0.0f),
        glm::vec3(-13.0f, -7.5f, 0.0f),
        glm::vec3(13.0f, -7.5f, 0.0f)
    };

    for (int c = 0; c < 3; c++) {
        int startIdx = c * third;
        int endIdx = (c == 2) ? particleCount : (c + 1) * third;

        universe.px[startIdx] = centers[c].x; universe.py[startIdx] = centers[c].y; universe.pz[startIdx] = centers[c].z;
        universe.vx[startIdx] = velocities[c].x; universe.vy[startIdx] = velocities[c].y; universe.vz[startIdx] = velocities[c].z;
        universe.ax[startIdx] = 0.0f; universe.ay[startIdx] = 0.0f; universe.az[startIdx] = 0.0f;
        universe.mass[startIdx] = smbhMass;
        universe.state[startIdx] = 0;

        for (int i = startIdx + 1; i < endIdx; i++) {
            glm::vec3 offset(clusterDist(gen), clusterDist(gen), clusterDist(gen));
            glm::vec3 pos = centers[c] + offset;
            
            float dist = glm::length(offset);
            float orbitalVelocity = sqrt((G * smbhMass) / max(0.1f, dist));
            
            glm::vec3 tangent;
            if (dist > 0.1f) {
                glm::vec3 randVec(clusterDist(gen), clusterDist(gen), clusterDist(gen));
                tangent = glm::normalize(glm::cross(offset, randVec));
            } else {
                tangent = glm::vec3(1.0f, 0.0f, 0.0f);
            }

            glm::vec3 vel = velocities[c] + tangent * orbitalVelocity * 0.7f; // 0.7 bound

            universe.px[i] = pos.x; universe.py[i] = pos.y; universe.pz[i] = pos.z;
            universe.vx[i] = vel.x; universe.vy[i] = vel.y; universe.vz[i] = vel.z;
            universe.ax[i] = 0.0f; universe.ay[i] = 0.0f; universe.az[i] = 0.0f;
            universe.mass[i] = massDist(gen);
            universe.state[i] = 0;
        }
    }
}

void generateGalaxyCollision(Universe &universe, float spawnRange) {
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float halfParticles = particleCount / 2.0f;
    exponential_distribution<float> radiusDist(3.0f / (spawnRange * 0.7f));
    uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    normal_distribution<float> heightDist(-spawnRange * 0.02f, spawnRange * 0.02f);
    uniform_real_distribution<float> massDist(1.0f, 5.0f);
    
    float smbhMass = 80000.0f;

    // Galaxy 1 (Left)
    glm::vec3 center1(-250.0f, 0.0f, 0.0f);
    glm::vec3 vel1(20.0f, 0.0f, 5.0f); // Moving right and slightly "forward"
    universe.px[0] = center1.x; universe.py[0] = center1.y; universe.pz[0] = center1.z;
        universe.vx[0] = vel1.x; universe.vy[0] = vel1.y; universe.vz[0] = vel1.z;
        universe.ax[0] = glm::vec3(0.0f).x; universe.ay[0] = glm::vec3(0.0f).y; universe.az[0] = glm::vec3(0.0f).z;
    universe.mass[0] = smbhMass;
    universe.state[0] = 0;

    // Galaxy 2 (Right)
    glm::vec3 center2(250.0f, 50.0f, 0.0f);
    glm::vec3 vel2(-20.0f, 0.0f, -5.0f); // Moving left
    universe.px[1] = center2.x; universe.py[1] = center2.y; universe.pz[1] = center2.z;
        universe.vx[1] = vel2.x; universe.vy[1] = vel2.y; universe.vz[1] = vel2.z;
        universe.ax[1] = glm::vec3(0.0f).x; universe.ay[1] = glm::vec3(0.0f).y; universe.az[1] = glm::vec3(0.0f).z;
    universe.mass[1] = smbhMass;
    universe.state[1] = 0;

    for (int i = 2; i < particleCount; i++) {
        bool isGal1 = (i < halfParticles);
        glm::vec3 center = isGal1 ? center1 : center2;
        glm::vec3 galVel = isGal1 ? vel1 : vel2;

        float r = radiusDist(gen);
        float theta = angleDist(gen);
        float y = heightDist(gen);

        glm::vec3 localPos = glm::vec3(r * cos(theta), y, r * sin(theta));
        
        // Tilt Galaxy 2 slightly
        if (!isGal1) {
            float tilt = 0.5f;
            float newY = localPos.y * cos(tilt) - localPos.z * sin(tilt);
            float newZ = localPos.y * sin(tilt) + localPos.z * cos(tilt);
            localPos.y = newY;
            localPos.z = newZ;
        }

        glm::vec3 position = center + localPos;

        float orbitalVelocity = sqrt((G * smbhMass) / max(r, 0.1f));
        glm::vec3 tangent = glm::normalize(glm::vec3(-localPos.z, 0.0f, localPos.x));
        
        if (!isGal1) {
            float tilt = 0.5f;
            float newY = tangent.y * cos(tilt) - tangent.z * sin(tilt);
            float newZ = tangent.y * sin(tilt) + tangent.z * cos(tilt);
            tangent.y = newY;
            tangent.z = newZ;
        }

        normal_distribution<float> noiseDist(0.0f, orbitalVelocity * 0.05f);
        glm::vec3 velocityNoise = glm::vec3(noiseDist(gen), noiseDist(gen), noiseDist(gen));

        glm::vec3 velocity = galVel + (tangent * orbitalVelocity) + velocityNoise;

        universe.px[i] = position.x; universe.py[i] = position.y; universe.pz[i] = position.z;
        universe.vx[i] = velocity.x; universe.vy[i] = velocity.y; universe.vz[i] = velocity.z;
        universe.ax[i] = glm::vec3(0.0f).x; universe.ay[i] = glm::vec3(0.0f).y; universe.az[i] = glm::vec3(0.0f).z;
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateBinaryAccretion(Universe &universe, float spawnRange) {
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float bhMass = 150000.0f;
    float distance = 150.0f;
    float orbitVel = sqrt((G * bhMass) / (distance * 2.0f));

    universe.px[0] = -distance; universe.py[0] = 0.0f; universe.pz[0] = 0.0f;
    universe.vx[0] = 0.0f; universe.vy[0] = 0.0f; universe.vz[0] = orbitVel;
    universe.mass[0] = bhMass;
    universe.state[0] = 0;
    universe.ax[0] = 0.0f; universe.ay[0] = 0.0f; universe.az[0] = 0.0f;

    universe.px[1] = distance; universe.py[1] = 0.0f; universe.pz[1] = 0.0f;
    universe.vx[1] = 0.0f; universe.vy[1] = 0.0f; universe.vz[1] = -orbitVel;
    universe.mass[1] = bhMass;
    universe.state[1] = 0;
    universe.ax[1] = 0.0f; universe.ay[1] = 0.0f; universe.az[1] = 0.0f;

    uniform_real_distribution<float> radiusDist(distance * 1.5f, spawnRange * 1.5f);
    uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    normal_distribution<float> heightDist(0.0f, 20.0f);
    uniform_real_distribution<float> massDist(0.1f, 2.0f);

    for (int i = 2; i < particleCount; i++) {
        float r = radiusDist(gen);
        float theta = angleDist(gen);
        glm::vec3 pos(r * cos(theta), heightDist(gen), r * sin(theta));
        
        float v = sqrt((G * (bhMass * 2.0f)) / r);
        glm::vec3 tangent = glm::normalize(glm::vec3(-pos.z, 0.0f, pos.x));
        
        universe.px[i] = pos.x; universe.py[i] = pos.y; universe.pz[i] = pos.z;
        universe.vx[i] = tangent.x * v; universe.vy[i] = tangent.y * v; universe.vz[i] = tangent.z * v;
        universe.ax[i] = glm::vec3(0.0f).x; universe.ay[i] = glm::vec3(0.0f).y; universe.az[i] = glm::vec3(0.0f).z;
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateCosmologicalWeb(Universe &universe, float spawnRange) {
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float boxSize = spawnRange * 1.5f;
    uniform_real_distribution<float> posDist(-boxSize, boxSize);
    normal_distribution<float> velDist(0.0f, 1.0f); // very cold
    uniform_real_distribution<float> massDist(1.0f, 10.0f);

    // Create a few "dark matter halos" (attractors)
    vector<glm::vec3> halos;
    for(int i = 0; i < 20; i++) {
        halos.push_back(glm::vec3(posDist(gen), posDist(gen), posDist(gen)));
    }

    for (int i = 0; i < particleCount; i++) {
        glm::vec3 pos(posDist(gen), posDist(gen), posDist(gen));
        
        // Slightly bias positions towards nearest halo
        glm::vec3 nearestHalo = halos[0];
        float minDist = glm::length(pos - halos[0]);
        for(int j = 1; j < 20; j++) {
            float d = glm::length(pos - halos[j]);
            if(d < minDist) { minDist = d; nearestHalo = halos[j]; }
        }
        
        pos = glm::mix(pos, nearestHalo, 0.3f); // Clump 30% towards nodes

        universe.px[i] = pos.x; universe.py[i] = pos.y; universe.pz[i] = pos.z;
        universe.vx[i] = glm::vec3(velDist(gen), velDist(gen), velDist(gen)).x; universe.vy[i] = glm::vec3(velDist(gen), velDist(gen), velDist(gen)).y; universe.vz[i] = glm::vec3(velDist(gen), velDist(gen), velDist(gen)).z;
        universe.ax[i] = glm::vec3(0.0f).x; universe.ay[i] = glm::vec3(0.0f).y; universe.az[i] = glm::vec3(0.0f).z;
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateDisruptedRing(Universe &universe, float spawnRange) {
    universe.px.resize(particleCount); universe.py.resize(particleCount); universe.pz.resize(particleCount);
    universe.vx.resize(particleCount); universe.vy.resize(particleCount); universe.vz.resize(particleCount);
    universe.ax.resize(particleCount); universe.ay.resize(particleCount); universe.az.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float centralMass = 200000.0f;
    universe.px[0] = glm::vec3(0.0f).x; universe.py[0] = glm::vec3(0.0f).y; universe.pz[0] = glm::vec3(0.0f).z;
        universe.vx[0] = glm::vec3(0.0f).x; universe.vy[0] = glm::vec3(0.0f).y; universe.vz[0] = glm::vec3(0.0f).z;
        universe.ax[0] = glm::vec3(0.0f).x; universe.ay[0] = glm::vec3(0.0f).y; universe.az[0] = glm::vec3(0.0f).z;
    universe.mass[0] = centralMass;
    universe.state[0] = 0;

    // Rogue planet
    float rogueMass = 50000.0f;
    universe.px[1] = glm::vec3(-spawnRange * 1.5f, 0.0f, spawnRange * 0.5f).x; universe.py[1] = glm::vec3(-spawnRange * 1.5f, 0.0f, spawnRange * 0.5f).y; universe.pz[1] = glm::vec3(-spawnRange * 1.5f, 0.0f, spawnRange * 0.5f).z;
        universe.vx[1] = glm::vec3(80.0f, 0.0f, -20.0f).x; universe.vy[1] = glm::vec3(80.0f, 0.0f, -20.0f).y; universe.vz[1] = glm::vec3(80.0f, 0.0f, -20.0f).z;
        universe.ax[1] = glm::vec3(0.0f).x; universe.ay[1] = glm::vec3(0.0f).y; universe.az[1] = glm::vec3(0.0f).z;
    universe.mass[1] = rogueMass;
    universe.state[1] = 0;

    normal_distribution<float> radiusDist(spawnRange * 0.6f, 15.0f); // Thin ring
    uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    normal_distribution<float> heightDist(0.0f, 2.0f); // Very flat
    uniform_real_distribution<float> massDist(0.1f, 1.0f);

    for (int i = 2; i < particleCount; i++) {
        float r = radiusDist(gen);
        float theta = angleDist(gen);
        glm::vec3 pos(r * cos(theta), heightDist(gen), r * sin(theta));
        
        float v = sqrt((G * centralMass) / r);
        glm::vec3 tangent = glm::normalize(glm::vec3(-pos.z, 0.0f, pos.x));
        
        universe.px[i] = pos.x; universe.py[i] = pos.y; universe.pz[i] = pos.z;
        universe.vx[i] = tangent.x * v; universe.vy[i] = tangent.y * v; universe.vz[i] = tangent.z * v;
        universe.ax[i] = glm::vec3(0.0f).x; universe.ay[i] = glm::vec3(0.0f).y; universe.az[i] = glm::vec3(0.0f).z;
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

// ==========================================
// CPU Force calculation (Unused, kept for reference)
void calculateForce(int rootIndex, Universe &universe, int targetBodyIndex) {
    
    int stack[64];
    int stackptr = 0;

    stack[stackptr++] = rootIndex;

    while (stackptr > 0) {

        int nodeIdx = stack[--stackptr];
        BvhNodeTraverse* node = &bvhTraversePool[nodeIdx];

        if (node == nullptr || node->mass == 0.0f) continue;

        // Base Case: Leaf Node
        if (node->rightChild == -1) {
            int bi = node->bodyIndex;
            if (bi != targetBodyIndex) {
                glm::vec3 direction = glm::vec3(universe.px[bi] - universe.px[targetBodyIndex], universe.py[bi] - universe.py[targetBodyIndex], universe.pz[bi] - universe.pz[targetBodyIndex]);
                float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
                float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;

                float accelMag = (G * universe.mass[bi]) * invDistanceCubed;
                universe.ax[targetBodyIndex] += direction.x*accelMag; universe.ay[targetBodyIndex] += direction.y*accelMag; universe.az[targetBodyIndex] += direction.z*accelMag;
            }
        }
        else {
            float width = node->halfWidth * 2.0f;
            glm::vec3 direction = glm::vec3(node->com_x, node->com_y, node->com_z) - glm::vec3(universe.px[targetBodyIndex], universe.py[targetBodyIndex], universe.pz[targetBodyIndex]);
            float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;

            if ((width * width) < (THETA * THETA * distanceSq)) { // Treat node as one body
                float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float accelMag = (G * node->mass) * invDistanceCubed;
                universe.ax[targetBodyIndex] += direction.x*accelMag; universe.ay[targetBodyIndex] += direction.y*accelMag; universe.az[targetBodyIndex] += direction.z*accelMag;
            }
            else { // Recurse
                if (node->leftChild != -1) stack[stackptr++] = node->leftChild;
                if (node->rightChild != -1) stack[stackptr++] = node->rightChild;
            }

        }
    }

}

void calculateDriftersForce(int rootIndex, Universe &universe, int drifterIndex) {

    if (universe.mass[drifterIndex] == 0.0) return;

    BvhNodeTraverse* root = &bvhTraversePool[rootIndex];

    glm::vec3 direction = glm::vec3(root->com_x, root->com_y, root->com_z) - glm::vec3(universe.px[drifterIndex], universe.py[drifterIndex], universe.pz[drifterIndex]);
    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
    float invDistanceCubed = invDistance * invDistance * invDistance;

    float accelMag = (G * root->mass) * invDistanceCubed;
    universe.ax[drifterIndex] += direction.x*accelMag; universe.ay[drifterIndex] += direction.y*accelMag; universe.az[drifterIndex] += direction.z*accelMag;

}


const char* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in float px;\n"
    "layout (location = 1) in float py;\n"
    "layout (location = 2) in float pz;\n"
    "layout (location = 3) in float vx;\n"
    "layout (location = 4) in float vy;\n"
    "layout (location = 5) in float vz;\n"
    "out vec3 vVel;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main() {\n"
    "   vVel = vec3(vx, vy, vz);\n"
    "   gl_Position = projection * view * vec4(px, py, pz, 1.0);\n"
    "}\0";

const char* fragmentShaderSource = "#version 330 core\n"
    "layout (location = 0) out vec4 FragColor;\n"
    "layout (location = 1) out vec4 BrightColor;\n"
    "in vec3 vVel;\n"
    "uniform float maxSpeed;\n"
    "void main() {\n"
    "   float speed = length(vVel);\n"
    "   float normalizedSpeed = clamp(speed / maxSpeed, 0.0, 1.0);\n"
    "   vec3 slowColor = vec3(0.8, 0.1, 0.0);\n"
    "   vec3 midColor = vec3(0.9, 0.6, 0.1);\n"
    "   vec3 fastColor = vec3(0.7, 0.9, 1.0);\n"
    "   vec3 color = mix(slowColor, midColor, clamp(normalizedSpeed * 2.0, 0.0, 1.0));\n"
    "   color = mix(color, fastColor, clamp((normalizedSpeed - 0.5) * 2.0, 0.0, 1.0));\n"
    "   FragColor = vec4(color, 1.0);\n"
    "   float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));\n"
    "   if(brightness > 0.4) BrightColor = vec4(FragColor.rgb, 1.0);\n"
    "   else BrightColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "}\n\0";

const char* postVertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "layout (location = 1) in vec2 aTexCoords;\n"
    "out vec2 TexCoords;\n"
    "void main() {\n"
    "   TexCoords = aTexCoords;\n"
    "   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
    "}\0";

const char* blurFragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoords;\n"
    "uniform sampler2D image;\n"
    "uniform bool horizontal;\n"
    "uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);\n"
    "void main() {\n"
    "    vec2 tex_offset = 1.0 / textureSize(image, 0);\n"
    "    vec3 result = texture(image, TexCoords).rgb * weight[0];\n"
    "    if(horizontal) {\n"
    "        for(int i = 1; i < 5; ++i) {\n"
    "            result += texture(image, TexCoords + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];\n"
    "            result += texture(image, TexCoords - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];\n"
    "        }\n"
    "    } else {\n"
    "        for(int i = 1; i < 5; ++i) {\n"
    "            result += texture(image, TexCoords + vec2(0.0, tex_offset.y * i)).rgb * weight[i];\n"
    "            result += texture(image, TexCoords - vec2(0.0, tex_offset.y * i)).rgb * weight[i];\n"
    "        }\n"
    "    }\n"
    "    FragColor = vec4(result, 1.0);\n"
    "}\0";

const char* finalCompositeShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec2 TexCoords;\n"
"uniform sampler2D scene;\n"
"uniform sampler2D bloomBlur;\n"
"uniform bool bloom;\n"
"uniform float exposure;\n"
"void main() {\n"
"    vec3 hdrColor = texture(scene, TexCoords).rgb;\n"
"    if(bloom) {\n"
"        vec3 bloomColor = texture(bloomBlur, TexCoords).rgb;\n"
"        hdrColor += bloomColor;\n"
"        vec3 result = vec3(1.0) - exp(-hdrColor * exposure);\n"
"        FragColor = vec4(result, 1.0);\n"
"    } else {\n"
"        FragColor = vec4(hdrColor, 1.0);\n"
"    }\n"
"}\0";

const char* fadeFragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform float fadeAmount;\n"
    "void main() {\n"
    "    FragColor = vec4(0.0, 0.0, 0.0, fadeAmount);\n"
    "}\0";

const char* copyFragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoords;\n"
    "uniform sampler2D screenTexture;\n"
    "void main() {\n"
    "    FragColor = texture(screenTexture, TexCoords);\n"
    "}\0";

struct SimFileInfo {
    std::string filename;
    std::string displayString;
    std::filesystem::file_time_type writeTime;
};

std::vector<SimFileInfo> getAvailableSimFiles() {
    std::vector<SimFileInfo> files;
    std::string dir = "simulation files";
    std::filesystem::create_directories(dir);
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bhsim") {
                SimFileInfo info;
                info.filename = dir + "/" + entry.path().filename().string();
                info.writeTime = std::filesystem::last_write_time(entry);
                
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(info.writeTime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
                std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
                std::tm* gmt = std::localtime(&tt);
                std::stringstream buffer;
                buffer << std::put_time(gmt, "%Y-%m-%d %H:%M:%S");
                info.displayString = entry.path().filename().string() + " (" + buffer.str() + ")";
                
                files.push_back(info);
            }
        }
        std::sort(files.begin(), files.end(), [](const SimFileInfo& a, const SimFileInfo& b) {
            return a.writeTime > b.writeTime;
        });
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    return files;
}

float cpuHalfToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) return 0.0f;
    exp = exp - 15 + 127;
    uint32_t f = (sign << 31) | (exp << 23) | (mant << 13);
    float res;
    memcpy(&res, &f, 4);
    return res;
}

struct CameraWaypoint {
    int frameIndex;
    glm::vec3 position;
    glm::vec3 lookAt;
};

void interpolateCamera(const std::vector<CameraWaypoint>& waypoints, int frame, glm::vec3& outPos, glm::vec3& outLook);
void runMP4Export(GLFWwindow* window, const char* filename, const char* mp4Filename, bool useCinematic, float radius, float pitch, float yaw, const std::vector<CameraWaypoint>& customPath = {});

void runPlayback(GLFWwindow* window, const char* filename) {
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Failed to open %s for playback.\n", filename);
        return;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        printf("Failed to create file mapping.\n");
        CloseHandle(hFile);
        return;
    }

    void* mappedData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mappedData) {
        printf("Failed to map view of file.\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return;
    }

    BhsimHeader* header = (BhsimHeader*)mappedData;
    if (strncmp(header->magic, "BHSM", 4) != 0) {
        printf("Invalid BHSM file!\n");
        UnmapViewOfFile(mappedData);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return;
    }

    int pCount = header->particleCount;
    int totalFrames = header->totalFrames;
    bool isFP16 = header->isFP16 != 0;

    auto compileProgram = [](const char* vSrc, const char* fSrc) -> unsigned int {
        unsigned int v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vSrc, NULL);
        glCompileShader(v);
        unsigned int f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fSrc, NULL);
        glCompileShader(f);
        unsigned int p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
        return p;
    };

    unsigned int shaderProgram = compileProgram(vertexShaderSource, fragmentShaderSource);
    unsigned int blurProgram = compileProgram(postVertexShaderSource, blurFragmentShaderSource);
    unsigned int finalProgram = compileProgram(postVertexShaderSource, finalCompositeShaderSource);
    unsigned int fadeProgram = compileProgram(postVertexShaderSource, fadeFragmentShaderSource);
    unsigned int copyProgram = compileProgram(postVertexShaderSource, copyFragmentShaderSource);


    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    if (isFP16) {
        glVertexAttribPointer(0, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 0));
        glVertexAttribPointer(1, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 1));
        glVertexAttribPointer(2, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 2));
        glVertexAttribPointer(3, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 3));
        glVertexAttribPointer(4, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 4));
        glVertexAttribPointer(5, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 5));
    } else {
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 0));
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 1));
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 2));
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 3));
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 4));
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 5));
    }
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);

    // Quad for post-processing
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // FBO Setup
    int scrWidth, scrHeight;
    glfwGetFramebufferSize(window, &scrWidth, &scrHeight);
    
    unsigned int hdrFBO;
    glGenFramebuffers(1, &hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    unsigned int colorBuffers[2];
    glGenTextures(2, colorBuffers);
    for (unsigned int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0);
    }
    unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);
    unsigned int rboDepth;
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, scrWidth, scrHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("Framebuffer not complete!\n");

    // Ping-pong FBOs for blur
    unsigned int pingpongFBO[2];
    unsigned int pingpongColorbuffers[2];
    glGenFramebuffers(2, pingpongFBO);
    glGenTextures(2, pingpongColorbuffers);
    for (unsigned int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongColorbuffers[i], 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Accumulation FBO for Motion Blur
    unsigned int accumFBO;
    unsigned int accumTexture;
    glGenFramebuffers(1, &accumFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
    glGenTextures(1, &accumTexture);
    glBindTexture(GL_TEXTURE_2D, accumTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTexture, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Initial Shader Configuration
    glUseProgram(shaderProgram);
    glUseProgram(blurProgram);
    glUniform1i(glGetUniformLocation(blurProgram, "image"), 0);
    glUseProgram(finalProgram);
    glUniform1i(glGetUniformLocation(finalProgram, "scene"), 0);
    glUniform1i(glGetUniformLocation(finalProgram, "bloomBlur"), 1);
    glUniform1i(glGetUniformLocation(finalProgram, "accum"), 2);

    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0f);

    int currentFrame = 0;
    bool playing = true;
    int playbackSpeed = 1;
    bool lockToCenter = false;
    float maxHeatmapSpeed = 100.0f;


    size_t posSize = isFP16 ? ((size_t)pCount * 6) : ((size_t)pCount * 12);
    size_t velSize = posSize;
    size_t frameSize = posSize + velSize;
    uint8_t* frameDataStart = (uint8_t*)mappedData + sizeof(BhsimHeader);

    std::vector<CameraWaypoint> customKeyframes;
    char mp4Output[256] = "exported videos/custom_export.mp4";
    std::filesystem::create_directories("exported videos");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(500, 500));
        ImGui::Begin("Playback Controls", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
        
        ImGui::SliderInt("Frame", &currentFrame, 0, totalFrames - 1);
        if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
        ImGui::SameLine();
        ImGui::SliderInt("Speed", &playbackSpeed, 1, 15);
        ImGui::Checkbox("Lock Camera to Center of Mass", &lockToCenter);
        ImGui::SliderFloat("Heatmap Max Speed", &maxHeatmapSpeed, 1.0f, 200.0f);
        
        ImGui::Separator();
        ImGui::Text("Post-Processing");
        ImGui::Checkbox("Enable Bloom", &g_enableBloom);
        if (g_enableBloom) ImGui::SliderFloat("Bloom Exposure", &g_bloomExposure, 0.1f, 5.0f);
        ImGui::Checkbox("Enable Motion Blur", &g_enableMotionBlur);
        if (g_enableMotionBlur) ImGui::SliderFloat("Motion Blur Strength", &g_motionBlurStrength, 0.1f, 0.99f);

        ImGui::Separator();
        ImGui::Text("Interactive MP4 Exporter");
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Mode 1: Static Tripod Camera");
        if (ImGui::Button("SET STATIC ANGLE HERE")) {
            customKeyframes.clear();
            float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
            float camY = cameraRadius * sin(glm::radians(cameraPitch));
            float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
            customKeyframes.push_back({0, glm::vec3(camX, camY, camZ), glm::vec3(0.0f, 0.0f, 0.0f)});
        }
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Mode 2: Moving Cinematic Spline");
        if (ImGui::Button("Add Spline Keyframe")) {
            float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
            float camY = cameraRadius * sin(glm::radians(cameraPitch));
            float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
            customKeyframes.push_back({currentFrame, glm::vec3(camX, camY, camZ), glm::vec3(0.0f, 0.0f, 0.0f)});
            std::sort(customKeyframes.begin(), customKeyframes.end(), [](const CameraWaypoint& a, const CameraWaypoint& b){
                return a.frameIndex < b.frameIndex;
            });
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear All Keyframes")) customKeyframes.clear();
        
        ImGui::Spacing();
        ImGui::Text("Active Keyframes: %d", (int)customKeyframes.size());
        for (size_t i = 0; i < customKeyframes.size(); i++) {
            ImGui::Text("  - Keyframe %d at Frame %d", (int)i + 1, customKeyframes[i].frameIndex);
        }
        
        if (customKeyframes.size() == 1) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[READY] Static Camera angle confirmed.");
        } else if (customKeyframes.size() > 1) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[READY] Cinematic Spline path confirmed.");
        }
        ImGui::Spacing();
        
        ImGui::InputText("Export .mp4", mp4Output, IM_ARRAYSIZE(mp4Output));
        if (ImGui::Button("RENDER MP4 NOW", ImVec2(-1, 30))) {
            bool playing_bak = playing;
            playing = false; // Pause playback
            
            // Generate missing keyframes if they only added 1, or 0
            if (customKeyframes.empty()) {
                // If they don't add any keyframes, just record the current camera as a static shot
                float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
                float camY = cameraRadius * sin(glm::radians(cameraPitch));
                float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
                customKeyframes.push_back({0, glm::vec3(camX, camY, camZ), glm::vec3(0.0f, 0.0f, 0.0f)});
            }
            
            runMP4Export(window, filename, mp4Output, false, 0, 0, 0, customKeyframes);
            
            playing = playing_bak;
        }

        if (ImGui::Button("Save Sliders & Return to Launcher", ImVec2(-1, 30))) {
            return; // Will return to main loop
        }
        ImGui::Spacing();
        if (ImGui::Button("Exit to Desktop", ImVec2(-1, 0))) {
            glfwSetWindowShouldClose(window, true);
        }
        ImGui::End();

        if (playing) {
            currentFrame += playbackSpeed;
            if (currentFrame >= totalFrames) currentFrame = totalFrames - 1;
        }

        uint8_t* currentFramePtr = frameDataStart + (currentFrame * frameSize);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, frameSize, currentFramePtr, GL_DYNAMIC_DRAW);
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glm::vec3 center(0.0f, 0.0f, 0.0f);
        if (lockToCenter) {
            double cx = 0, cy = 0, cz = 0;
            if (isFP16) {
                uint16_t* ptr = (uint16_t*)currentFramePtr;
                for (int i = 0; i < pCount; i++) {
                    cx += cpuHalfToFloat(ptr[i*3 + 0]);
                    cy += cpuHalfToFloat(ptr[i*3 + 1]);
                    cz += cpuHalfToFloat(ptr[i*3 + 2]);
                }
            } else {
                float* ptr = (float*)currentFramePtr;
                for (int i = 0; i < pCount; i++) {
                    cx += (double)ptr[i*3 + 0];
                    cy += (double)ptr[i*3 + 1];
                    cz += (double)ptr[i*3 + 2];
                }
            }
            center = glm::vec3((float)(cx / pCount), (float)(cy / pCount), (float)(cz / pCount));
        }

        float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        float camY = cameraRadius * sin(glm::radians(cameraPitch));
        float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));
        glm::vec3 cameraPos = center + glm::vec3(camX, camY, camZ);

        glm::mat4 view = glm::lookAt(cameraPos, center, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)display_w / (float)display_h, 0.1f, 10000.0f);

        // 1. Draw scene into HDR FBO
        glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // MUST be pure black for accumulation!
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
        unsigned int projLoc = glGetUniformLocation(shaderProgram, "projection");
        unsigned int speedLoc = glGetUniformLocation(shaderProgram, "maxSpeed");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(speedLoc, maxHeatmapSpeed);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, pCount);
        
        // 2. Gaussian Blur on BrightColor buffer (ping-pong)
        bool horizontal = true;
        if (g_enableBloom) {
            bool first_iteration = true;
            int amount = 10;
            glUseProgram(blurProgram);
            for (unsigned int i = 0; i < amount; i++) {
                glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[horizontal]);
                glUniform1i(glGetUniformLocation(blurProgram, "horizontal"), horizontal);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, first_iteration ? colorBuffers[1] : pingpongColorbuffers[!horizontal]);
                
                glBindVertexArray(quadVAO);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                
                horizontal = !horizontal;
                if (first_iteration) first_iteration = false;
            }
        }

        // 3. Motion Blur Accumulation
        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        if (g_enableMotionBlur) {
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(fadeProgram); glUniform1f(glGetUniformLocation(fadeProgram, "fadeAmount"), 1.0f - g_motionBlurStrength);
            glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glDisable(GL_BLEND);
        } else {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
        }

        // Add current HDR scene to accumulation buffer
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE); // Additive
        glUseProgram(finalProgram); 
        glUniform1i(glGetUniformLocation(finalProgram, "bloom"), g_enableBloom ? 1 : 0);
        glUniform1f(glGetUniformLocation(finalProgram, "exposure"), g_bloomExposure);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorBuffers[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[!horizontal]);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);

        // 4. Render to Screen (Default Framebuffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f); // Deep space blue background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(copyProgram);
        glUniform1i(glGetUniformLocation(copyProgram, "screenTexture"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accumTexture);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_BLEND);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    UnmapViewOfFile(mappedData);
    CloseHandle(hMapping);
    CloseHandle(hFile);
}

void runOfflineSimulation(GLFWwindow* window, int choice, float targetDt, int durationSeconds, int targetFPS, bool useFP16, const char* filename, bool chunkFiles, int chunkSizeGB) {
    Universe universe;
    if (choice == 2) generateGalaxyCollision(universe, 300);
    else if (choice == 3) generateBinaryAccretion(universe, 300);
    else if (choice == 4) generateCosmologicalWeb(universe, 300);
    else if (choice == 5) generateDisruptedRing(universe, 300);
    else if (choice == 6) generateClusterThreeBody(universe, 300);
    else generateGalaxyUniverse(universe, 300);

    // CUDA Allocations
    float *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz, *d_ax, *d_ay, *d_az;
    float *d_masses;
    int *d_states;
    CUDA_CHECK(cudaMalloc(&d_px, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ax, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ay, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_az, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_masses, particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, particleCount * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_px, universe.px.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, universe.py.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, universe.pz.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, universe.vx.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, universe.vy.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, universe.vz.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ax, universe.ax.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ay, universe.ay.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az, universe.az.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses, universe.mass.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_states, universe.state.data(), particleCount * sizeof(int), cudaMemcpyHostToDevice));

    uint64_t *d_mortonCodes, *d_mortonCodesTemp;
    int *d_bodyIndices, *d_bodyIndicesTemp, *d_activeCount, *d_nodeFlags;
    int maxNodes = 2 * particleCount - 1;
    BvhNodeTraverse* d_trPool;
    BvhNodeBuild* d_bdPool;

    CUDA_CHECK(cudaMalloc(&d_mortonCodes, particleCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_mortonCodesTemp, particleCount * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bodyIndices, particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bodyIndicesTemp, particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_activeCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nodeFlags, maxNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_trPool, maxNodes * sizeof(BvhNodeTraverse)));
    CUDA_CHECK(cudaMalloc(&d_bdPool, maxNodes * sizeof(BvhNodeBuild)));

    // Pinned memory setup
    size_t singleArraySize = useFP16 ? (particleCount * 2) : (particleCount * 4);
    void *h_px[2], *h_py[2], *h_pz[2], *h_vx[2], *h_vy[2], *h_vz[2];
    for (int i=0; i<2; ++i) {
        CUDA_CHECK(cudaHostAlloc(&h_px[i], singleArraySize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_py[i], singleArraySize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_pz[i], singleArraySize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_vx[i], singleArraySize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_vy[i], singleArraySize, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_vz[i], singleArraySize, cudaHostAllocDefault));
    }

    void *d_fp16_px = nullptr, *d_fp16_py = nullptr, *d_fp16_pz = nullptr;
    void *d_fp16_vx = nullptr, *d_fp16_vy = nullptr, *d_fp16_vz = nullptr;
    if (useFP16) {
        CUDA_CHECK(cudaMalloc(&d_fp16_px, singleArraySize));
        CUDA_CHECK(cudaMalloc(&d_fp16_py, singleArraySize));
        CUDA_CHECK(cudaMalloc(&d_fp16_pz, singleArraySize));
        CUDA_CHECK(cudaMalloc(&d_fp16_vx, singleArraySize));
        CUDA_CHECK(cudaMalloc(&d_fp16_vy, singleArraySize));
        CUDA_CHECK(cudaMalloc(&d_fp16_vz, singleArraySize));
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Failed to open file for writing.\n");
        return;
    }
    BhsimHeader header;
    memcpy(header.magic, "BHSM", 4);
    header.version = 2;
    header.particleCount = particleCount;
    header.totalFrames = durationSeconds * targetFPS;
    header.dt = targetDt;
    header.isFP16 = useFP16 ? 1 : 0;
    fwrite(&header, sizeof(BhsimHeader), 1, file);

    cudaStream_t computeStream, copyStream;
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&copyStream);

    int totalFrames = header.totalFrames;
    int currentFrame = 0;
    bool paused = false;
    bool finished = false;

    // Thread synchronization
    std::mutex mtx;
    std::condition_variable cv;
    bool writeReady = false;
    bool threadExit = false;
    int writeBufferIdx = 0;

    std::thread writerThread([&]() {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return writeReady || threadExit; });
            if (threadExit && !writeReady) break;
            
            fwrite(h_px[writeBufferIdx], 1, singleArraySize, file);
            fwrite(h_py[writeBufferIdx], 1, singleArraySize, file);
            fwrite(h_pz[writeBufferIdx], 1, singleArraySize, file);
            fwrite(h_vx[writeBufferIdx], 1, singleArraySize, file);
            fwrite(h_vy[writeBufferIdx], 1, singleArraySize, file);
            fwrite(h_vz[writeBufferIdx], 1, singleArraySize, file);
            
            writeReady = false;
            lock.unlock();
            cv.notify_one();
        }
    });

    float simulatedTime = 0.0f;
    float timeBetweenFrames = 1.0f / (float)targetFPS;
    float nextSnapshotTime = 0.0f;
    int bufferIdx = 0;

    while (!glfwWindowShouldClose(window) && !finished && currentFrame < totalFrames) {
        glfwPollEvents();

        // UI Rendering
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(1280 / 2.0f, 720 / 2.0f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(400, 200));
        ImGui::Begin("Offline Calculation", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

        float progress = (float)currentFrame / (float)totalFrames;
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f));
        ImGui::Text("Frame: %d / %d", currentFrame, totalFrames);
        ImGui::Spacing();
        if (ImGui::Button(paused ? "Resume" : "Pause")) {
            paused = !paused;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save & Exit Early")) {
            finished = true;
        }
        ImGui::End();

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();

        if (paused) {
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
            continue;
        }

        // --- Physics Ticks ---
        while (simulatedTime < nextSnapshotTime && !finished) {
            BoundingBox universeBoundingBox(glm::vec3(0.0f), 2000.0f);
            
            launchKDKStep1(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_states, targetDt, particleCount);

            int activeBodyCount = generateMortonAndGetActiveCount(d_px, d_py, d_pz, d_states, d_mortonCodes, d_bodyIndices, d_activeCount, universeBoundingBox, particleCount);

            launchSimulationStep(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_masses, d_states,
                d_mortonCodes, d_bodyIndices, d_mortonCodesTemp, d_bodyIndicesTemp,
                d_trPool, d_bdPool, d_activeCount, d_nodeFlags, universeBoundingBox,
                activeBodyCount, particleCount, targetDt, THETA, G, SofteningSq
            );
            simulatedTime += targetDt;
        }

        // We reached snapshot time! Let's download data to h_pos[bufferIdx].
        // First wait for previous write to finish.
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return !writeReady; });
        }

        // Issue D2H copy
        if (useFP16) {
            launchConvertToHalf(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_fp16_px, d_fp16_py, d_fp16_pz, d_fp16_vx, d_fp16_vy, d_fp16_vz, particleCount, computeStream);
            cudaMemcpyAsync(h_px[bufferIdx], d_fp16_px, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_py[bufferIdx], d_fp16_py, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_pz[bufferIdx], d_fp16_pz, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vx[bufferIdx], d_fp16_vx, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vy[bufferIdx], d_fp16_vy, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vz[bufferIdx], d_fp16_vz, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
        } else {
            cudaMemcpyAsync(h_px[bufferIdx], d_px, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_py[bufferIdx], d_py, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_pz[bufferIdx], d_pz, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vx[bufferIdx], d_vx, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vy[bufferIdx], d_vy, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
            cudaMemcpyAsync(h_vz[bufferIdx], d_vz, singleArraySize, cudaMemcpyDeviceToHost, computeStream);
        }

        cudaStreamSynchronize(computeStream);

        // Tell writer thread to write
        {
            std::lock_guard<std::mutex> lock(mtx);
            writeBufferIdx = bufferIdx;
            writeReady = true;
        }
        cv.notify_one();

        // Swap buffer
        bufferIdx = 1 - bufferIdx;
        currentFrame++;
        nextSnapshotTime += timeBetweenFrames;

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    {
        std::lock_guard<std::mutex> lock(mtx);
        threadExit = true;
    }
    cv.notify_all();
    writerThread.join();

    // Overwrite the header with the actual number of frames generated
    // This fixes the bug where "Save & Exit Early" leaves corrupted totalFrames
    header.totalFrames = currentFrame;
    fseek(file, 0, SEEK_SET);
    fwrite(&header, sizeof(BhsimHeader), 1, file);
    fclose(file);
    cudaFreeHost(h_px[0]); cudaFreeHost(h_py[0]); cudaFreeHost(h_pz[0]);
    cudaFreeHost(h_px[1]); cudaFreeHost(h_py[1]); cudaFreeHost(h_pz[1]);
    cudaFreeHost(h_vx[0]); cudaFreeHost(h_vy[0]); cudaFreeHost(h_vz[0]);
    cudaFreeHost(h_vx[1]); cudaFreeHost(h_vy[1]); cudaFreeHost(h_vz[1]);
}

void interpolateCamera(const std::vector<CameraWaypoint>& waypoints, int frame, glm::vec3& outPos, glm::vec3& outLook) {
    if (waypoints.empty()) return;
    if (frame <= waypoints.front().frameIndex) {
        outPos = waypoints.front().position;
        outLook = waypoints.front().lookAt;
        return;
    }
    if (frame >= waypoints.back().frameIndex) {
        outPos = waypoints.back().position;
        outLook = waypoints.back().lookAt;
        return;
    }
    for (size_t i = 0; i < waypoints.size() - 1; i++) {
        if (frame >= waypoints[i].frameIndex && frame < waypoints[i+1].frameIndex) {
            int diff = waypoints[i+1].frameIndex - waypoints[i].frameIndex;
            if (diff == 0) {
                outPos = waypoints[i].position;
                outLook = waypoints[i].lookAt;
                return;
            }
            float t = (float)(frame - waypoints[i].frameIndex) / diff;
            float t2 = (1.0f - cosf(t * 3.14159265f)) / 2.0f; // Smooth interpolation
            outPos = glm::mix(waypoints[i].position, waypoints[i+1].position, t2);
            outLook = glm::mix(waypoints[i].lookAt, waypoints[i+1].lookAt, t2);
            return;
        }
    }
}

void runMP4Export(GLFWwindow* window, const char* filename, const char* mp4Filename, bool useCinematic, float radius, float pitch, float yaw, const std::vector<CameraWaypoint>& customPath) {
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("Failed to open %s for export.\n", filename);
        return;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) { CloseHandle(hFile); return; }
    void* mappedData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mappedData) { CloseHandle(hMapping); CloseHandle(hFile); return; }

    BhsimHeader* header = (BhsimHeader*)mappedData;
    if (strncmp(header->magic, "BHSM", 4) != 0) {
        printf("Invalid BHSM file!\n");
        UnmapViewOfFile(mappedData); CloseHandle(hMapping); CloseHandle(hFile); return;
    }

    int pCount = header->particleCount;
    int totalFrames = header->totalFrames;
    bool isFP16 = header->isFP16 != 0;

    auto compileProgram = [](const char* vSrc, const char* fSrc) -> unsigned int {
        unsigned int v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vSrc, NULL);
        glCompileShader(v);
        unsigned int f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fSrc, NULL);
        glCompileShader(f);
        unsigned int p = glCreateProgram();
        glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
        glDeleteShader(v); glDeleteShader(f); return p;
    };

    unsigned int shaderProgram = compileProgram(vertexShaderSource, fragmentShaderSource);
    unsigned int blurProgram = compileProgram(postVertexShaderSource, blurFragmentShaderSource);
    unsigned int finalProgram = compileProgram(postVertexShaderSource, finalCompositeShaderSource);
    unsigned int fadeProgram = compileProgram(postVertexShaderSource, fadeFragmentShaderSource);
    unsigned int copyProgram = compileProgram(postVertexShaderSource, copyFragmentShaderSource);

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO);
    glBindVertexArray(VAO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
    if (isFP16) {
        glVertexAttribPointer(0, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 0));
        glVertexAttribPointer(1, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 1));
        glVertexAttribPointer(2, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 2));
        glVertexAttribPointer(3, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 3));
        glVertexAttribPointer(4, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 4));
        glVertexAttribPointer(5, 1, GL_HALF_FLOAT, GL_FALSE, 2, (void*)((size_t)pCount * 2 * 5));
    } else {
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 0));
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 1));
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 2));
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 3));
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 4));
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 4, (void*)((size_t)pCount * 4 * 5));
    }
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);

    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f, -1.0f, -1.0f,  0.0f, 0.0f,  1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,  1.0f, -1.0f,  1.0f, 0.0f,  1.0f,  1.0f,  1.0f, 1.0f
    };
    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO); glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO); glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    int scrWidth = 1920, scrHeight = 1080;
    
    unsigned int hdrFBO, colorBuffers[2], rboDepth;
    glGenFramebuffers(1, &hdrFBO); glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glGenTextures(2, colorBuffers);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0);
    }
    unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);
    glGenRenderbuffers(1, &rboDepth); glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, scrWidth, scrHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

    unsigned int pingpongFBO[2], pingpongColorbuffers[2];
    glGenFramebuffers(2, pingpongFBO); glGenTextures(2, pingpongColorbuffers);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongColorbuffers[i], 0);
    }

    unsigned int accumFBO, accumTexture;
    glGenFramebuffers(1, &accumFBO); glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
    glGenTextures(1, &accumTexture); glBindTexture(GL_TEXTURE_2D, accumTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, scrWidth, scrHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, accumTexture, 0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);

    unsigned int outputFBO, outputTexture;
    glGenFramebuffers(1, &outputFBO); glBindFramebuffer(GL_FRAMEBUFFER, outputFBO);
    glGenTextures(1, &outputTexture); glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, scrWidth, scrHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTexture, 0);
    
    glUseProgram(shaderProgram);
    glUseProgram(blurProgram); glUniform1i(glGetUniformLocation(blurProgram, "image"), 0);
    glUseProgram(finalProgram); glUniform1i(glGetUniformLocation(finalProgram, "scene"), 0); glUniform1i(glGetUniformLocation(finalProgram, "bloomBlur"), 1);
    
    glEnable(GL_DEPTH_TEST); glPointSize(2.0f);

    size_t posSize = isFP16 ? ((size_t)pCount * 6) : ((size_t)pCount * 12);
    size_t frameSize = posSize * 2;
    uint8_t* frameDataStart = (uint8_t*)mappedData + sizeof(BhsimHeader);

    std::vector<CameraWaypoint> cameraPath;
    if (!customPath.empty()) {
        cameraPath = customPath;
    } else {
        cameraPath = {
            {0, glm::vec3(0, 800, 800), glm::vec3(0,0,0)},
            {totalFrames / 4, glm::vec3(500, 200, 500), glm::vec3(0,0,0)},
            {totalFrames / 2, glm::vec3(0, 50, 400), glm::vec3(0,0,0)},
            {(totalFrames * 3) / 4, glm::vec3(-400, 200, 200), glm::vec3(0,0,0)},
            {totalFrames - 1, glm::vec3(0, 600, 100), glm::vec3(0,0,0)}
        };
    }

    uint8_t* pixels = new uint8_t[scrWidth * scrHeight * 3];

    char cmd[512];
    sprintf(cmd, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s 1920x1080 -r 60 -i - -vf vflip -c:v libx264 -preset slow -crf 18 \"%s\"", mp4Filename);
    FILE* ffmpeg = _popen(cmd, "wb");
    if (!ffmpeg) {
        printf("Failed to launch FFmpeg!\n");
        return;
    }

    printf("Starting MP4 Export (1920x1080 @ 60fps) with %zu keyframes...\n", customPath.size());

    for (int frame = 0; frame < totalFrames; frame++) {
        if (frame % 60 == 0) {
            printf("\rExported frame %d / %d                                     ", frame, totalFrames);
            fflush(stdout);
        }
        glfwPollEvents();
        if (glfwWindowShouldClose(window)) break;
        
        uint8_t* currentFramePtr = frameDataStart + (frame * frameSize);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, frameSize, currentFramePtr, GL_DYNAMIC_DRAW);

        glm::vec3 camPos, camLook;
        if (useCinematic || !customPath.empty()) {
            interpolateCamera(cameraPath, frame, camPos, camLook);
        } else {
            float camX = radius * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
            float camY = radius * sin(glm::radians(pitch));
            float camZ = radius * cos(glm::radians(pitch)) * sin(glm::radians(yaw));
            camPos = glm::vec3(camX, camY, camZ);
            camLook = glm::vec3(0.0f, 0.0f, 0.0f);
        }
        glm::mat4 view = glm::lookAt(camPos, camLook, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)scrWidth / (float)scrHeight, 0.1f, 10000.0f);

        glViewport(0, 0, scrWidth, scrHeight);

        glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(shaderProgram, "maxSpeed"), 50.0f);
        glBindVertexArray(VAO); glDrawArrays(GL_POINTS, 0, pCount);

        bool horizontal = true, first_iteration = true;
        if (g_enableBloom) {
            glUseProgram(blurProgram);
            for (int i = 0; i < 10; i++) {
                glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[horizontal]);
                glUniform1i(glGetUniformLocation(blurProgram, "horizontal"), horizontal);
                glBindTexture(GL_TEXTURE_2D, first_iteration ? colorBuffers[1] : pingpongColorbuffers[!horizontal]);
                glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6);
                horizontal = !horizontal; if (first_iteration) first_iteration = false;
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, accumFBO);
        if (g_enableMotionBlur) {
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(fadeProgram); glUniform1f(glGetUniformLocation(fadeProgram, "fadeAmount"), 1.0f - g_motionBlurStrength);
            glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glDisable(GL_BLEND);
        } else {
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
        }

        glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
        glUseProgram(finalProgram);
        glUniform1i(glGetUniformLocation(finalProgram, "bloom"), g_enableBloom ? 1 : 0);
        glUniform1f(glGetUniformLocation(finalProgram, "exposure"), g_bloomExposure);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, colorBuffers[0]);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, pingpongColorbuffers[!horizontal]);
        glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glDisable(GL_BLEND);

        glBindFramebuffer(GL_FRAMEBUFFER, outputFBO);
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(copyProgram); glUniform1i(glGetUniformLocation(copyProgram, "screenTexture"), 0);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, accumTexture);
        glBindVertexArray(quadVAO); glDrawArrays(GL_TRIANGLES, 0, 6); glDisable(GL_BLEND);

        glReadPixels(0, 0, scrWidth, scrHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        fwrite(pixels, 1, scrWidth * scrHeight * 3, ffmpeg);

        if (frame % 60 == 0) printf("Exported frame %d / %d\n", frame, totalFrames);
    }

    _pclose(ffmpeg);
    delete[] pixels;
    UnmapViewOfFile(mappedData); CloseHandle(hMapping); CloseHandle(hFile);
    printf("Export complete!\n");
}

int main() {

    if (!glfwInit()) {
        cerr << "Failed to load GLFW" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Barnes-Hut Simulation", NULL, NULL);

    if (!window) {
        cerr << "Failed to create GLFWwindow" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, frame_buffer_size_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cerr << "Failed to initialise GLAD" << endl;
        return -1;
    }

    int cudaDeviceCount = 0;
    unsigned int glDeviceCount = 0;
    int glDevices[4];

    // Ask CUDA which device(s) are driving the current OpenGL context
    cudaGLGetDevices(&glDeviceCount, glDevices, 4, cudaGLDeviceListCurrentFrame);

    if (glDeviceCount == 0) {
        fprintf(stderr, "No CUDA device found for the current OpenGL context!\n");
        return -1;
    }

    // Force CUDA to use that device
    cudaSetDevice(glDevices[0]);
    fprintf(stdout, "Using CUDA device %d for GL interop\n", glDevices[0]);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ==========================================
    // PHASE 1: ImGui Launcher Menu
    // ==========================================
    enum AppMode { MODE_REALTIME, MODE_PRECALC, MODE_PLAYBACK, MODE_MP4_EXPORT };
    AppMode currentMode = MODE_REALTIME;
    int initialConditionChoice = 1;
    float targetDt = 0.01f;
    int targetFPS = 60;
    int durationSeconds = 60;
    bool useFP16 = true;
    bool chunkFiles = false;
    int chunkSizeGB = 10;
    std::filesystem::create_directories("simulation files");
    std::filesystem::create_directories("exported videos");
    char saveFilename[256] = "simulation files/simulation.bhsim";
    char playbackFilename[256] = "simulation files/simulation.bhsim";
    char mp4OutputFilename[256] = "exported videos/output.mp4";
    bool launcherFinished = false;
    bool shouldExit = false;
    bool vSync = true;
    
    bool useCinematicCamera = true;
    float exportRadius = 800.0f;
    float exportPitch = 45.0f;
    float exportYaw = 0.0f;
    
    std::vector<SimFileInfo> simFiles = getAvailableSimFiles();
    int selectedSimFileIdx = 0;
    if (!simFiles.empty()) {
        strncpy(playbackFilename, simFiles[0].filename.c_str(), sizeof(playbackFilename));
    }

    while (true) {
        launcherFinished = false;
        
        while (!launcherFinished && !glfwWindowShouldClose(window)) {
            glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(1280 / 2.0f, 720 / 2.0f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(600, 500));
        ImGui::Begin("Barnes-Hut Engine Launcher", NULL, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);

        if (ImGui::BeginTabBar("ModeTabs")) {
            if (ImGui::BeginTabItem("Real-Time Engine")) {
                currentMode = MODE_REALTIME;
                ImGui::Text("Run physics and rendering simultaneously.");
                ImGui::Spacing();
                int rtConditionIdx = initialConditionChoice - 1;
                ImGui::Combo("Initial Condition", &rtConditionIdx, 
                             "Standard Spiral Galaxy\0Galaxy Collision (The Milky Way & Andromeda)\0Binary Accretion System\0Cosmological Web\0The Disrupted Ring\0The Cluster 3-Body Problem\0\0");
                initialConditionChoice = rtConditionIdx + 1;
                
                ImGui::InputInt("Particle Count", &particleCount, 1000, 10000);
                if (particleCount < 100) particleCount = 100;
                
                ImGui::SliderFloat("Barnes-Hut Theta", &THETA, 0.1f, 1.0f, "%.2f");
                ImGui::Checkbox("Enable V-Sync", &vSync);
                
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Pre-Calculate Offline")) {
                currentMode = MODE_PRECALC;
                ImGui::Text("Run headless calculation at maximum speed and save to disk.");
                ImGui::Spacing();
                
                int conditionIdx = initialConditionChoice - 1;
                ImGui::Combo("Initial Condition", &conditionIdx, 
                             "Standard Spiral Galaxy\0Galaxy Collision (The Milky Way & Andromeda)\0Binary Accretion System\0Cosmological Web\0The Disrupted Ring\0The Cluster 3-Body Problem\0\0");
                initialConditionChoice = conditionIdx + 1;

                ImGui::InputInt("Particle Count", &particleCount, 1000, 10000);
                if (particleCount < 100) particleCount = 100;

                ImGui::SliderFloat("Barnes-Hut Theta", &THETA, 0.1f, 1.0f, "%.2f");
                ImGui::InputFloat("Physics dt", &targetDt, 0.001f, 0.01f, "%.4f");
                ImGui::SliderInt("Simulation Duration (s)", &durationSeconds, 10, 600);
                ImGui::SliderInt("Target Playback FPS", &targetFPS, 24, 240);
                
                ImGui::Separator();
                ImGui::Text("Storage Configuration");
                ImGui::Checkbox("Use FP16 Storage (50%% smaller, extremely fast I/O)", &useFP16);
                ImGui::InputText("Save Filename", saveFilename, IM_ARRAYSIZE(saveFilename));

                // Calculate file size
                long long totalFrames = durationSeconds * targetFPS;
                long long bytesPerParticle = useFP16 ? 6 : 12; // half3 vs float3
                long long bytesPerFrame = particleCount * bytesPerParticle;
                long long totalBytes = totalFrames * bytesPerFrame;
                float totalGB = (float)totalBytes / (1024.0f * 1024.0f * 1024.0f);

                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Estimated File Size: %.2f GB", totalGB);
                
                if (totalGB > 5.0f) {
                    ImGui::Checkbox("Auto-chunk files?", &chunkFiles);
                    if (chunkFiles) ImGui::SliderInt("Chunk Size (GB)", &chunkSizeGB, 1, 20);
                }

                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Playback Renderer")) {
                currentMode = MODE_PLAYBACK;
                ImGui::Text("Stream pre-calculated simulations directly from the SSD.");
                ImGui::Spacing();
                
                if (ImGui::Button("Refresh Files")) {
                    simFiles = getAvailableSimFiles();
                }
                ImGui::SameLine();
                if (ImGui::BeginCombo("Select Simulation", simFiles.empty() ? "No files found" : simFiles[selectedSimFileIdx].displayString.c_str())) {
                    for (size_t i = 0; i < simFiles.size(); i++) {
                        bool is_selected = (selectedSimFileIdx == i);
                        if (ImGui::Selectable(simFiles[i].displayString.c_str(), is_selected)) {
                            selectedSimFileIdx = i;
                            strncpy(playbackFilename, simFiles[i].filename.c_str(), sizeof(playbackFilename));
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Cinematic MP4 Export")) {
                currentMode = MODE_MP4_EXPORT;
                ImGui::Text("Headless rendering to MP4 via FFmpeg using keyframed splines.");
                ImGui::Spacing();
                
                if (ImGui::Button("Refresh Files##mp4")) {
                    simFiles = getAvailableSimFiles();
                }
                ImGui::SameLine();
                if (ImGui::BeginCombo("Select Simulation##mp4", simFiles.empty() ? "No files found" : simFiles[selectedSimFileIdx].displayString.c_str())) {
                    for (size_t i = 0; i < simFiles.size(); i++) {
                        bool is_selected = (selectedSimFileIdx == i);
                        if (ImGui::Selectable(simFiles[i].displayString.c_str(), is_selected)) {
                            selectedSimFileIdx = i;
                            strncpy(playbackFilename, simFiles[i].filename.c_str(), sizeof(playbackFilename));
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Interactive Custom Spline Editor & Post-Processing:");
                if (ImGui::Button("OPEN INTERACTIVE CINEMATIC DIRECTOR", ImVec2(-1, 30))) {
                    currentMode = MODE_PLAYBACK;
                    launcherFinished = true;
                }
                if (ImGui::Button("CONFIGURE BLOOM & MOTION BLUR", ImVec2(-1, 30))) {
                    currentMode = MODE_PLAYBACK;
                    launcherFinished = true;
                }
                ImGui::Spacing();

                ImGui::Separator();
                ImGui::Checkbox("Use Pre-Programmed Demo Camera Spline", &useCinematicCamera);
                if (!useCinematicCamera) {
                    ImGui::SliderFloat("Camera Radius", &exportRadius, 100.0f, 5000.0f);
                    ImGui::SliderFloat("Camera Pitch", &exportPitch, -89.0f, 89.0f);
                    ImGui::SliderFloat("Camera Yaw", &exportYaw, 0.0f, 360.0f);
                }
                ImGui::Separator();
                
                ImGui::InputText("Output .mp4", mp4OutputFilename, IM_ARRAYSIZE(mp4OutputFilename));
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        
        if (ImGui::Button("LAUNCH ENGINE", ImVec2(-1, 50))) {
            launcherFinished = true;
        }

        ImGui::End();

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
        if (glfwWindowShouldClose(window)) {
            glfwTerminate();
            return 0;
        }

        if (currentMode == MODE_PRECALC) {
            printf("Routing to Offline Calculation...\n");
            runOfflineSimulation(window, initialConditionChoice, targetDt, durationSeconds, targetFPS, useFP16, saveFilename, chunkFiles, chunkSizeGB);
            if (!glfwWindowShouldClose(window)) continue;
            glfwTerminate();
            return 0;
        }

        if (currentMode == MODE_PLAYBACK) {
            printf("Routing to Playback Renderer...\n");
            runPlayback(window, playbackFilename);
            if (!glfwWindowShouldClose(window)) {
                currentMode = MODE_MP4_EXPORT; // Go back to export tab!
                continue; 
            }
            glfwTerminate();
            return 0;
        }

        if (currentMode == MODE_MP4_EXPORT) {
            printf("Routing to Headless MP4 Export...\n");
            runMP4Export(window, playbackFilename, mp4OutputFilename, useCinematicCamera, exportRadius, exportPitch, exportYaw);
            if (!glfwWindowShouldClose(window)) continue;
            glfwTerminate();
            return 0;
        }

        if (currentMode == MODE_REALTIME) {
            break;
        }
    }

    // Apply V-Sync setting
    glfwSwapInterval(vSync ? 1 : 0);

    // ==========================================
    // Original Real-Time Engine Setup Below
    // ==========================================

    // Shader Compilation
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // VAO and VBO Setup

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Allocate GPU Memory for bodies

    Universe universe;
    
    if (initialConditionChoice == 2) generateGalaxyCollision(universe, 300);
    else if (initialConditionChoice == 3) generateBinaryAccretion(universe, 300);
    else if (initialConditionChoice == 4) generateCosmologicalWeb(universe, 300);
    else if (initialConditionChoice == 5) generateDisruptedRing(universe, 300);
    else if (initialConditionChoice == 6) generateClusterThreeBody(universe, 300);
    else generateGalaxyUniverse(universe, 300);
    
    glBufferData(GL_ARRAY_BUFFER, particleCount * 4 * 6, NULL, GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, particleCount * 4, universe.px.data());
        glBufferSubData(GL_ARRAY_BUFFER, particleCount * 4 * 1, particleCount * 4, universe.py.data());
        glBufferSubData(GL_ARRAY_BUFFER, particleCount * 4 * 2, particleCount * 4, universe.pz.data());
        glBufferSubData(GL_ARRAY_BUFFER, particleCount * 4 * 3, particleCount * 4, universe.vx.data());
        glBufferSubData(GL_ARRAY_BUFFER, particleCount * 4 * 4, particleCount * 4, universe.vy.data());
        glBufferSubData(GL_ARRAY_BUFFER, particleCount * 4 * 5, particleCount * 4, universe.vz.data());

    // Tell OpenGL how to read position from body struct
    // Stride = sizeof(Body), offset is where 'position' starts
    
    size_t pCount = particleCount;
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 0));
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 1));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 2));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 3));
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 4));
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 0, (void*)((size_t)pCount * 4 * 5));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);

    // Allow z-depth sortin and change point size
    glEnable(GL_DEPTH_TEST);

    glPointSize(2.0f);

    // ==========================================
    // STEP 2: CUDA GPU Memory Allocation
    // ==========================================

    float *d_vx, *d_vy, *d_vz;
    float *d_ax, *d_ay, *d_az;
    float* d_masses;
    int* d_states;

    cudaMalloc(&d_vx, particleCount * sizeof(float));
    cudaMalloc(&d_vy, particleCount * sizeof(float));
    cudaMalloc(&d_vz, particleCount * sizeof(float));
    cudaMalloc(&d_ax, particleCount * sizeof(float));
    cudaMalloc(&d_ay, particleCount * sizeof(float));
    cudaMalloc(&d_az, particleCount * sizeof(float));
    cudaMalloc(&d_masses, particleCount * sizeof(float));
    cudaMalloc(&d_states, particleCount * sizeof(int));

    // Copy initial CPU state to the GPU
    cudaMemcpy(d_vx, universe.vx.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_vy, universe.vy.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_vz, universe.vz.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ax, universe.ax.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_ay, universe.ay.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice); cudaMemcpy(d_az, universe.az.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, universe.mass.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, universe.state.data(), particleCount * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate Device memory for the Karras Tree
    uint64_t* d_mortonCodes;
    int* d_bodyIndices;
    int maxNodes = 2 * particleCount - 1;
    BvhNodeTraverse* d_trPool;
    BvhNodeBuild* d_bdPool;
    
    cudaMalloc(&d_mortonCodes, particleCount * sizeof(uint64_t));
    cudaMalloc(&d_bodyIndices, particleCount * sizeof(int));
    
    // Temp buffers for custom Merge Sort
    uint64_t* d_mortonCodesTemp;
    int* d_bodyIndicesTemp;
    cudaMalloc(&d_mortonCodesTemp, particleCount * sizeof(uint64_t));
    cudaMalloc(&d_bodyIndicesTemp, particleCount * sizeof(int));

    cudaMalloc(&d_trPool, maxNodes * sizeof(BvhNodeTraverse));
    cudaMalloc(&d_bdPool, maxNodes * sizeof(BvhNodeBuild));
    
    int* d_activeCount;
    cudaMalloc(&d_activeCount, sizeof(int));
    int* d_nodeFlags;
    cudaMalloc(&d_nodeFlags, maxNodes * sizeof(int));

    // OpenGL Interop: Map the VBO directly to CUDA to avoid CPU round-trips!
    cudaGraphicsResource* vbo_resource;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vbo_resource, VBO, cudaGraphicsMapFlagsNone));
    float *d_px, *d_py, *d_pz;

  


    bool pauseSimulation = false;
    int simulationSpeed = 1;
    int maxThreads = omp_get_max_threads();
    int activeThreads = 1;
    glm::vec3 systemCOM = glm::vec3(0.0f);

    globalBodyIndices.resize(particleCount);
    
    vector<uint64_t> mortonCodes(particleCount);
    vector<int> bodyIndices(particleCount);
    float maxHeatmapSpeed = 50.0f;
    bool lockToCenter = false;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        // Menu
        {
            ImGui::Begin("Simulation Controls");

            ImGui::Text("Barnes-Hut Engine Status: Active");
                
            ImGui::Checkbox("Pause Simulation", &pauseSimulation);
            ImGui::SliderFloat("Theta (Accuracy)", &THETA, 0.0, 1.5);
            ImGui::SliderFloat("Heatmap Max Speed", &maxHeatmapSpeed, 1.0f, 200.0f);
            ImGui::Checkbox("Lock Camera to Center of Mass", &lockToCenter);

            ImGui::SliderInt("Fast Forward", &simulationSpeed, 1, 20);

            ImGui::SliderInt("CPU Threads", &activeThreads, 1, maxThreads);

            ImGui::Text("Application average: %.3f ms/frame; FPS: (%.1f FPS), COM: {%.3f, %.3f, %.3f}", 1000.f / io.Framerate, io.Framerate, systemCOM.x, systemCOM.y, systemCOM.z);


            ImGui::End();
        }

        if (!pauseSimulation) {
            for (int i = 0; i < simulationSpeed; i++) {


                int threads = 256;
                int blocks = (particleCount + threads - 1) / threads;

                // 1. Map VBO to get d_positions
        CUDA_CHECK(cudaGraphicsMapResources(1, &vbo_resource, 0));
        size_t num_bytes;
        void* mapped_ptr;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&mapped_ptr, &num_bytes, vbo_resource));
        d_px = (float*)mapped_ptr;
        d_py = d_px + particleCount;
        d_pz = d_py + particleCount;

                // 2. KDK Step 1
                launchKDKStep1(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_states, dt, particleCount);

                // 3. Drifter analysis and Bounding Box on CPU (Copy positions back)
                // 120KB copy is virtually instant
                cudaMemcpy(universe.px.data(), d_px, particleCount * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(universe.py.data(), d_py, particleCount * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(universe.pz.data(), d_pz, particleCount * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(universe.vx.data(), d_vx, particleCount * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(universe.vy.data(), d_vy, particleCount * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(universe.vz.data(), d_vz, particleCount * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(universe.state.data(), d_states, particleCount * sizeof(int), cudaMemcpyDeviceToHost);



                // Calculating Standard Deviation
                float sum_r = 0.0;
                int valid_count = 0;
#pragma omp parallel for reduction(+:sum_r, valid_count)  num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.mass[j] > 0.0f && universe.state[j] == 0) {
                        sum_r += sqrt(universe.px[j]*universe.px[j] + universe.py[j]*universe.py[j] + universe.pz[j]*universe.pz[j]);
                        valid_count++;
                    }
                }

                float mu = (valid_count > 0) ? (sum_r / valid_count) : 0.0f;
                float sum_variance = 0.0f;
#pragma omp parallel for reduction(+: sum_variance) num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.mass[j] > 0.0f && universe.state[j] == 0) {
                        float r = sqrt(universe.px[j]*universe.px[j] + universe.py[j]*universe.py[j] + universe.pz[j]*universe.pz[j]);
                        sum_variance += (r - mu) * (r - mu);
                    }
                }

                float sigma = (valid_count > 0) ? sqrt(sum_variance / valid_count) : 0.0f;
                float MAX_BOUND = (valid_count > 0) ? max(300.0f, mu + (5 * sigma)) : 800.0f;

                // Calculating mass for escape velocity
                float totalMass = 0.0f;
#pragma omp parallel for reduction(+:totalMass) num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.state[j] == 0) totalMass += universe.mass[j];
                }

                float escVelTerm = 2 * G * totalMass;

                // Checking state
#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.mass[j] <= 0) continue;
                    float distance = sqrt(universe.px[j]*universe.px[j] + universe.py[j]*universe.py[j] + universe.pz[j]*universe.pz[j]);

                    if (distance <= MAX_BOUND) {
                        if (universe.state[j] == 1) universe.state[j] = 0;
                    }
                    else {
                        glm::vec3 vel = glm::vec3(universe.vx[j], universe.vy[j], universe.vz[j]);
                        float velsq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

                        if (velsq >= (escVelTerm / max(0.1f, sqrt(universe.px[j]*universe.px[j] + universe.py[j]*universe.py[j] + universe.pz[j]*universe.pz[j])))) universe.state[j] = 2; 
                        else universe.state[j] = 1; 
                    }
                }

                float xMin = FLT_MAX, xMax = -FLT_MAX;
                float yMin = FLT_MAX, yMax = -FLT_MAX;
                float zMin = FLT_MAX, zMax = -FLT_MAX;

                // Dynamic Sizing 
                for (int j = 0; j < particleCount; j++) {
                    if (universe.state[j] == 1 || universe.state[j] == 2) continue;
                    if (universe.px[j] < xMin) xMin = universe.px[j];
                    if (universe.px[j] > xMax) xMax = universe.px[j];
                    if (universe.py[j] < yMin) yMin = universe.py[j];
                    if (universe.py[j] > yMax) yMax = universe.py[j];
                    if (universe.pz[j] < zMin) zMin = universe.pz[j];
                    if (universe.pz[j] > zMax) zMax = universe.pz[j];
                }

                glm::vec3 universeCenter = glm::vec3((xMin + xMax) / 2.0, (yMin + yMax) / 2.0, (zMin + zMax) / 2.0);
                float universeHalfWidth = std::max({ (xMax - universeCenter.x), (yMax - universeCenter.y), (zMax - universeCenter.z) });
                BoundingBox universeBoundingBox(universeCenter, universeHalfWidth + 1.0f);

                // Copy states back (since drifter logic updated them)
                cudaMemcpy(d_states, universe.state.data(), particleCount * sizeof(int), cudaMemcpyHostToDevice);

                // 4. Generate Morton Codes
                int activeBodyCount = generateMortonAndGetActiveCount(d_px, d_py, d_pz, d_states, d_mortonCodes, d_bodyIndices, d_activeCount, universeBoundingBox, particleCount);

                launchSimulationStep(d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az, d_masses, d_states,
                    d_mortonCodes, d_bodyIndices, d_mortonCodesTemp, d_bodyIndicesTemp,
                    d_trPool, d_bdPool, d_activeCount, d_nodeFlags, universeBoundingBox,
                    activeBodyCount, particleCount, dt, THETA, G, SofteningSq
                );

                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
                    


            }
        }

        // Rendering Section

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glUseProgram(shaderProgram);

        // Dynamic Cam

        if (lockToCenter) {
            BvhNodeTraverse rootNode;
            cudaMemcpy(&rootNode, d_trPool, sizeof(BvhNodeTraverse), cudaMemcpyDeviceToHost);
            systemCOM = glm::vec3(rootNode.com_x, rootNode.com_y, rootNode.com_z);
        } else {
            systemCOM = glm::vec3(0.0f);
        }

        float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        float camY = cameraRadius * sin(glm::radians(cameraPitch));
        float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));

        glm::vec3 cameraPos = systemCOM + glm::vec3(camX, camY, camZ);

        // Cam looks at center
        glm::mat4 view = glm::lookAt(cameraPos, systemCOM, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)display_w / (float)display_h, 0.1f, 10000.0f);

        unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
        unsigned int projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(shaderProgram, "maxSpeed"), maxHeatmapSpeed);


        // Draw
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, particleCount);

        ImGui::Render();

        glViewport(0, 0, display_w, display_h);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}