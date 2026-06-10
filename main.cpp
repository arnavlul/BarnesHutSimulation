#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <iostream>
#include "BarnesHutSimulation.h"
#include <random>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstddef>
#include <omp.h>
#include <math.h>
#include <cstdint>
#include <algorithm>
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

const float G = 1.0f;
float dt = 0.01f;
float THETA = 0.5f;
const int particleCount = 50000;
const float SofteningSq = 0.2f;

// KarrasNode Pool
vector<KarrasNode> karrasPool;

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

    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
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
       
        universe.position[i] = position;
        universe.velocity[i] = velocity;
        universe.acceleration[i] = acceleration;
        universe.mass[i] = mass;
        universe.state[i] = 0;
    }
}

void generateGalaxyUniverse(Universe &universe, float spawnRange) {
    
    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
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
    
    universe.position[0] = smbh_position;
    universe.velocity[0] = smbh_velocity;
    universe.acceleration[0] = smbh_acceleration;
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

        universe.position[i] = position;
        universe.velocity[i] = velocity;
        universe.acceleration[i] = acceleration;
        universe.mass[i] = mass;
        universe.state[i] = 0;

    }
}

void generateGalaxyCollision(Universe &universe, float spawnRange) {
    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
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
    universe.position[0] = center1;
    universe.velocity[0] = vel1;
    universe.acceleration[0] = glm::vec3(0.0f);
    universe.mass[0] = smbhMass;
    universe.state[0] = 0;

    // Galaxy 2 (Right)
    glm::vec3 center2(250.0f, 50.0f, 0.0f);
    glm::vec3 vel2(-20.0f, 0.0f, -5.0f); // Moving left
    universe.position[1] = center2;
    universe.velocity[1] = vel2;
    universe.acceleration[1] = glm::vec3(0.0f);
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

        universe.position[i] = position;
        universe.velocity[i] = velocity;
        universe.acceleration[i] = glm::vec3(0.0f);
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateBinaryAccretion(Universe &universe, float spawnRange) {
    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float bhMass = 150000.0f;
    float distance = 150.0f;
    float orbitVel = sqrt((G * bhMass) / (distance * 2.0f));

    universe.position[0] = glm::vec3(-distance, 0.0f, 0.0f);
    universe.velocity[0] = glm::vec3(0.0f, 0.0f, orbitVel);
    universe.mass[0] = bhMass;
    universe.state[0] = 0;
    universe.acceleration[0] = glm::vec3(0.0f);

    universe.position[1] = glm::vec3(distance, 0.0f, 0.0f);
    universe.velocity[1] = glm::vec3(0.0f, 0.0f, -orbitVel);
    universe.mass[1] = bhMass;
    universe.state[1] = 0;
    universe.acceleration[1] = glm::vec3(0.0f);

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
        
        universe.position[i] = pos;
        universe.velocity[i] = tangent * v;
        universe.acceleration[i] = glm::vec3(0.0f);
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateCosmologicalWeb(Universe &universe, float spawnRange) {
    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
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

        universe.position[i] = pos;
        universe.velocity[i] = glm::vec3(velDist(gen), velDist(gen), velDist(gen));
        universe.acceleration[i] = glm::vec3(0.0f);
        universe.mass[i] = massDist(gen);
        universe.state[i] = 0;
    }
}

void generateDisruptedRing(Universe &universe, float spawnRange) {
    universe.position.resize(particleCount);
    universe.velocity.resize(particleCount);
    universe.acceleration.resize(particleCount);
    universe.mass.resize(particleCount);
    universe.state.resize(particleCount);

    random_device rd;
    mt19937 gen(rd());

    float centralMass = 200000.0f;
    universe.position[0] = glm::vec3(0.0f);
    universe.velocity[0] = glm::vec3(0.0f);
    universe.acceleration[0] = glm::vec3(0.0f);
    universe.mass[0] = centralMass;
    universe.state[0] = 0;

    // Rogue planet
    float rogueMass = 50000.0f;
    universe.position[1] = glm::vec3(-spawnRange * 1.5f, 0.0f, spawnRange * 0.5f);
    universe.velocity[1] = glm::vec3(80.0f, 0.0f, -20.0f);
    universe.acceleration[1] = glm::vec3(0.0f);
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
        
        universe.position[i] = pos;
        universe.velocity[i] = tangent * v;
        universe.acceleration[i] = glm::vec3(0.0f);
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
        KarrasNode* node = &karrasPool[nodeIdx];

        if (node == nullptr || node->totalMass == 0.0f) continue;

        // Base Case: Leaf Node
        if (node->isLeaf) {
            int bi = node->bodyIndex;
            if (bi != targetBodyIndex) {
                glm::vec3 direction = universe.position[bi] - universe.position[targetBodyIndex];
                float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
                float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;

                float accelMag = (G * universe.mass[bi]) * invDistanceCubed;
                universe.acceleration[targetBodyIndex] += (direction)*accelMag;
            }
        }
        else {
            float width = node->bbox.halfWidth * 2.0f;
            glm::vec3 direction = node->centerOfMass - universe.position[targetBodyIndex];
            float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;

            if ((width * width) < (THETA * THETA * distanceSq)) { // Treat node as one body
                float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
                float invDistanceCubed = invDistance * invDistance * invDistance;
                float accelMag = (G * node->totalMass) * invDistanceCubed;
                universe.acceleration[targetBodyIndex] += direction * accelMag;
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

    KarrasNode* root = &karrasPool[rootIndex];

    glm::vec3 direction = root->centerOfMass - universe.position[drifterIndex];
    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    float invDistance = 1.0 / sqrt(distanceSq + SofteningSq);
    float invDistanceCubed = invDistance * invDistance * invDistance;

    float accelMag = (G * root->totalMass) * invDistanceCubed;
    universe.acceleration[drifterIndex] += direction * accelMag;

}


const char* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"
    "void main() {\n"
    "   gl_Position = projection * view * vec4(aPos, 1.0);\n"
    "}\0";

const char* fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main() {\n"
    "   FragColor = vec4(0.5, 0.8, 1.0, 1.0);\n"
    "}\n\0";



int main() {
    int initialConditionChoice = 1;
    cout << "\n============================================\n";
    cout << "  BARNES-HUT N-BODY SIMULATION STARTUP      \n";
    cout << "============================================\n";
    cout << "Select Initial Condition:\n";
    cout << "1: Standard Spiral Galaxy\n";
    cout << "2: Galaxy Collision (The Milky Way & Andromeda)\n";
    cout << "3: Binary Accretion System\n";
    cout << "4: Cosmological Web\n";
    cout << "5: The Disrupted Ring\n";
    cout << "Choice (1-5): ";
    cin >> initialConditionChoice;

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
    else generateGalaxyUniverse(universe, 300);
    
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(glm::vec3), universe.position.data(), GL_DYNAMIC_DRAW);

    // Tell OpenGL how to read position from body struct
    // Stride = sizeof(Body), offset is where 'position' starts
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Allow z-depth sortin and change point size
    glEnable(GL_DEPTH_TEST);

    glPointSize(2.0f);

    // ==========================================
    // STEP 2: CUDA GPU Memory Allocation
    // ==========================================

    glm::vec3* d_velocities;
    glm::vec3* d_accelerations;
    float* d_masses;
    int* d_states;

    cudaMalloc(&d_velocities, particleCount * sizeof(glm::vec3));
    cudaMalloc(&d_accelerations, particleCount * sizeof(glm::vec3));
    cudaMalloc(&d_masses, particleCount * sizeof(float));
    cudaMalloc(&d_states, particleCount * sizeof(int));

    // Copy initial CPU state to the GPU
    cudaMemcpy(d_velocities, universe.velocity.data(), particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accelerations, universe.acceleration.data(), particleCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, universe.mass.data(), particleCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, universe.state.data(), particleCount * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate Device memory for the Karras Tree
    uint64_t* d_mortonCodes;
    int* d_bodyIndices;
    int maxNodes = 2 * particleCount - 1;
    KarrasNode* d_karrasPool;
    
    cudaMalloc(&d_mortonCodes, particleCount * sizeof(uint64_t));
    cudaMalloc(&d_bodyIndices, particleCount * sizeof(int));
    
    // Temp buffers for custom Merge Sort
    uint64_t* d_mortonCodesTemp;
    int* d_bodyIndicesTemp;
    cudaMalloc(&d_mortonCodesTemp, particleCount * sizeof(uint64_t));
    cudaMalloc(&d_bodyIndicesTemp, particleCount * sizeof(int));

    cudaMalloc(&d_karrasPool, maxNodes * sizeof(KarrasNode));
    
    int* d_activeCount;
    cudaMalloc(&d_activeCount, sizeof(int));
    int* d_nodeFlags;
    cudaMalloc(&d_nodeFlags, maxNodes * sizeof(int));

    // OpenGL Interop: Map the VBO directly to CUDA to avoid CPU round-trips!
    cudaGraphicsResource* vbo_resource;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vbo_resource, VBO, cudaGraphicsMapFlagsNone));
    glm::vec3* d_positions; // We will extract this pointer from the VBO during the render loop

  


    bool pauseSimulation = false;
    int simulationSpeed = 1;
    int maxThreads = omp_get_max_threads();
    int activeThreads = 1;
    glm::vec3 systemCOM = glm::vec3(0.0f);

    globalBodyIndices.resize(particleCount);
    
    vector<uint64_t> mortonCodes(particleCount);
    vector<int> bodyIndices(particleCount);


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
                CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &num_bytes, vbo_resource));

                // 2. KDK Step 1
                launchKDKStep1(d_positions, d_velocities, d_accelerations, d_states, dt, particleCount);

                // 3. Drifter analysis and Bounding Box on CPU (Copy positions back)
                // 120KB copy is virtually instant
                cudaMemcpy(universe.position.data(), d_positions, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
                cudaMemcpy(universe.velocity.data(), d_velocities, particleCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
                cudaMemcpy(universe.state.data(), d_states, particleCount * sizeof(int), cudaMemcpyDeviceToHost);



                // Calculating Standard Deviation
                float sum_r = 0.0;
                int valid_count = 0;
#pragma omp parallel for reduction(+:sum_r, valid_count)  num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.mass[j] > 0.0f && universe.state[j] == 0) {
                        sum_r += glm::length(universe.position[j]);
                        valid_count++;
                    }
                }

                float mu = (valid_count > 0) ? (sum_r / valid_count) : 0.0f;
                float sum_variance = 0.0f;
#pragma omp parallel for reduction(+: sum_variance) num_threads(activeThreads)
                for (int j = 0; j < particleCount; j++) {
                    if (universe.mass[j] > 0.0f && universe.state[j] == 0) {
                        float r = glm::length(universe.position[j]);
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
                    float distance = glm::length(universe.position[j]);

                    if (distance <= MAX_BOUND) {
                        if (universe.state[j] == 1) universe.state[j] = 0;
                    }
                    else {
                        glm::vec3 vel = universe.velocity[j];
                        float velsq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

                        if (velsq >= (escVelTerm / max(0.1f, glm::length(universe.position[j])))) universe.state[j] = 2; 
                        else universe.state[j] = 1; 
                    }
                }

                float xMin = FLT_MAX, xMax = -FLT_MAX;
                float yMin = FLT_MAX, yMax = -FLT_MAX;
                float zMin = FLT_MAX, zMax = -FLT_MAX;

                // Dynamic Sizing 
                for (int j = 0; j < particleCount; j++) {
                    if (universe.state[j] == 1 || universe.state[j] == 2) continue;
                    if (universe.position[j].x < xMin) xMin = universe.position[j].x;
                    if (universe.position[j].x > xMax) xMax = universe.position[j].x;
                    if (universe.position[j].y < yMin) yMin = universe.position[j].y;
                    if (universe.position[j].y > yMax) yMax = universe.position[j].y;
                    if (universe.position[j].z < zMin) zMin = universe.position[j].z;
                    if (universe.position[j].z > zMax) zMax = universe.position[j].z;
                }

                glm::vec3 universeCenter = glm::vec3((xMin + xMax) / 2.0, (yMin + yMax) / 2.0, (zMin + zMax) / 2.0);
                float universeHalfWidth = std::max({ (xMax - universeCenter.x), (yMax - universeCenter.y), (zMax - universeCenter.z) });
                BoundingBox universeBoundingBox(universeCenter, universeHalfWidth + 1.0f);

                // Copy states back (since drifter logic updated them)
                cudaMemcpy(d_states, universe.state.data(), particleCount * sizeof(int), cudaMemcpyHostToDevice);

                // 4. Generate Morton Codes
                int activeBodyCount = generateMortonAndGetActiveCount(d_positions, d_states, d_mortonCodes, d_bodyIndices, d_activeCount, universeBoundingBox, particleCount);

                launchSimulationStep(
                    d_positions, d_velocities, d_accelerations, d_masses, d_states,
                    d_mortonCodes, d_bodyIndices, d_mortonCodesTemp, d_bodyIndicesTemp,
                    d_karrasPool, d_activeCount, d_nodeFlags, universeBoundingBox,
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

        float camX = cameraRadius * cos(glm::radians(cameraPitch)) * cos(glm::radians(cameraYaw));
        float camY = cameraRadius * sin(glm::radians(cameraPitch));
        float camZ = cameraRadius * cos(glm::radians(cameraPitch)) * sin(glm::radians(cameraYaw));

        glm::vec3 cameraPos = glm::vec3(camX, camY, camZ);

        // Cam looks at center
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)display_w / (float)display_h, 0.1f, 10000.0f);

        unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
        unsigned int projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));


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