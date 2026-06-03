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

extern "C" {
    __declspec(dllexport) uint32_t NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

using namespace std;

const float G = 1.0f;
float dt = 0.01f;
float THETA = 0.5f;
const int particleCount = 10000;
const float SofteningSq = 0.2f;

// Camera Variables
float cameraRadius = 600.0f;
float cameraYaw = 45.0f;
float cameraPitch = 30.0f;

bool isDragging = false;
double lastMouseX = 0.0;
double lastMouseY = 0.0;


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

void generateUniformUniverse(vector<Body>& bodies, float spawnRange) {

    bodies.reserve(particleCount);

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> posGen(-spawnRange, spawnRange);
    uniform_real_distribution<float> velGen(-1.0, 1.0);
    uniform_real_distribution<float> massGen(1.0, 5.0);

    for (int i = 0; i < particleCount; i++) {
        Body b;
        b.position = glm::vec3(posGen(gen), posGen(gen), posGen(gen));
        b.velocity = glm::vec3(velGen(gen), velGen(gen), velGen(gen));
        b.acceleration = glm::vec3(0.0f);
        b.mass = massGen(gen);
        bodies.push_back(b);
    }
}

void generateBlackHoleCentricUniverse(vector<Body>& bodies, float spawnRange) {

    bodies.reserve(particleCount);

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<float> angleGen(0.0f, 2.0f * 3.14159265f);
    normal_distribution<float> radiusGen(0.0f, spawnRange * 0.4f);
    uniform_real_distribution<float> massGen(1.0, 5.0);
    uniform_real_distribution<float> velGen(-1.0, 1.0);

    Body centralCore;
    centralCore.position = glm::vec3(0.0, 0.0, 0.0);
    centralCore.velocity = glm::vec3(0.0, 0.0, 0.0);
    centralCore.acceleration = glm::vec3(0.0f);
    centralCore.mass = 100000.0f;

    bodies.push_back(centralCore);

    const float G = 1.0f;

    for (int i = 0; i < particleCount; i++) {
        Body b;
        
        float theta = angleGen(gen);
        float phi = angleGen(gen);
        float radius = radiusGen(gen) + 10.0f;
        float vel = velGen(gen);

        b.position = glm::vec3(radius * cos(theta), radius * sin(theta), radius * sin(phi));
        b.velocity = glm::vec3(cos(phi) * cos(theta) * vel, cos(phi) * sin(theta) * vel, sin(phi) * vel);
        b.acceleration = glm::vec3(0.0f);
        b.mass = massGen(gen);

        bodies.push_back(b);
    }
}

void generateGalaxyUniverse(vector<Body>& bodies, float spawnRange) {
    bodies.clear();
    bodies.reserve(particleCount + 1);

    random_device rd;
    mt19937 gen(rd());

    // 1. Spatial Distributions (Creating the thin disk)
    // Exponential distribution clusters stars near the center
    exponential_distribution<float> radiusDist(3.0f / spawnRange);
    uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    // Normal distribution creates a very thin height profile (Y-axis)
    normal_distribution<float> heightDist(-spawnRange * 0.02f, spawnRange * 0.02f);

    uniform_real_distribution<float> massDist(1.0f, 5.0f);
    float smbhMass = 100000.0f;

    // ---------------------------------------------------------
    // GENERATE THE SUPERMASSIVE BLACK HOLE
    // ---------------------------------------------------------
    Body smbh;
    smbh.position = glm::vec3(0.0f);
    smbh.velocity = glm::vec3(0.0f);
    smbh.acceleration = glm::vec3(0.0f);

    smbh.mass = smbhMass;
    bodies.push_back(smbh);

    // ---------------------------------------------------------
    // GENERATE THE ORBITING STARS
    // ---------------------------------------------------------
    for (int i = 0; i < particleCount; i++) {
        Body b;

        // 1. Calculate Spatial Position
        float r = radiusDist(gen);
        // Clamp bounds to prevent spawning inside the black hole or outside the map

        float theta = angleDist(gen);
        float y = heightDist(gen); // Disk is on the X-Z plane, Y is height

        b.position = glm::vec3(r * cos(theta), y, r * sin(theta));

        // 2. Calculate Orbital Velocity
        // v = sqrt(G * M / r)
        float orbitalVelocity = sqrt((G * smbhMass) / r);

        // Calculate the Tangent Vector (perpendicular to the radius in the X-Z plane)
        // If position is (x, z), tangent is (-z, x)
        glm::vec3 tangent = glm::normalize(glm::vec3(-b.position.z, 0.0f, b.position.x));

        // Add 5% random velocity noise (Velocity Dispersion)
        // This prevents the galaxy from looking like a perfect, artificial DVD ring 
        // and gives the Hermite solver organic clumping to work with.
        normal_distribution<float> noiseDist(0.0f, orbitalVelocity * 0.05f);
        glm::vec3 velocityNoise = glm::vec3(noiseDist(gen), noiseDist(gen), noiseDist(gen));

        b.velocity = (tangent * orbitalVelocity) + velocityNoise;

        // 3. Zero out ALL integration memory for GPU safety
        b.acceleration = glm::vec3(0.0f);

        b.mass = massDist(gen);


        bodies.push_back(b);
    }
}

// Force calculation
void calculateForce(OctNode* node, Body* targetBody) {

    if (node == nullptr || node->totalMass == 0.0f) return;

    // Base Case: Leaf Node
    if (node->isLeaf) {
        for (Body* b : node->bodies) {
            if (b == targetBody) continue;

            glm::vec3 direction = b->position - targetBody->position;
            float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
            float distance = sqrt(distanceSq);

            float accelMag = (G * b->mass) / (distanceSq + SofteningSq);
            targetBody->acceleration += (direction / distance) * accelMag;
        }
    }
    else {
        float width = node->bbox.halfWidth * 2.0f;
        glm::vec3 direction = node->centerOfMass - targetBody->position;
        float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;

        if ((width * width) < (THETA * THETA * distanceSq)) { // Treat node as one body
            float distance = sqrt(distanceSq);
            float accelMag = (G * node->totalMass) / (distanceSq + SofteningSq);
            targetBody->acceleration += (direction / distance) * accelMag;
        }
        else { // Recurse
            for (int i = 0; i < 8; i++) {
                if (node->children[i] != nullptr) {
                    calculateForce(node->children[i], targetBody);
                }
            }
        }

    }

}

void calculateDriftersForce(OctNode *root, Body *drifter) {

    if (drifter->mass == 0.0) return;

    glm::vec3 direction = root->centerOfMass - drifter->position;
    float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    float distance = sqrt(distanceSq);

    float accelMag = (G * root->totalMass) / (distanceSq + SofteningSq);
    drifter->acceleration += (direction / distance) * accelMag;

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

    vector<Body> bodies;
    generateGalaxyUniverse(bodies, 300);
    glBufferData(GL_ARRAY_BUFFER, bodies.size() * sizeof(Body), bodies.data(), GL_DYNAMIC_DRAW);

    // Tell OpenGL how to read position from body struct
    // Stride = sizeof(Body), offset is where 'position' starts
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Body), (void*)offsetof(Body, position));
    glEnableVertexAttribArray(0);

    // Allow z-depth sortin and change point size
    glEnable(GL_DEPTH_TEST);

    glPointSize(2.0f);

  


    bool pauseSimulation = false;
    int simulationSpeed = 1;
    int maxThreads = omp_get_max_threads();
    int activeThreads = 1;
    glm::vec3 systemCOM = glm::vec3(0.0f);


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


                // First 2 updates of KDK Leapfrog
#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    if (bodies[j].state == 2) continue;
                    bodies[j].velocity += bodies[j].acceleration * (dt / 2.0f);
                    bodies[j].position += bodies[j].velocity * (dt);
                }

                // Calculating Standard Deviation
                float sum_r = 0.0;
                int valid_count = 0;

                for (const Body& b : bodies) {
                    if (b.mass > 0.0f && b.state == 0) {
                        sum_r += glm::length(b.position);
                        valid_count++;
                    }
                }

                float mu = (valid_count > 0) ? (sum_r / valid_count) : 0.0f;

                float sum_variance = 0.0f;
                for (const Body& b : bodies) {
                    if (b.mass > 0.0f && b.state == 0) {
                        float r = glm::length(b.position);
                        sum_variance += (r - mu) * (r - mu);
                    }
                }

                float sigma = (valid_count > 0) ? sqrt(sum_variance / valid_count) : 0.0f;

                float MAX_BOUND = (valid_count > 0) ? max(300.0f, mu + (5 * sigma)) : 800.0f;

                // Calculating mass for escape velocity
                float totalMass = 0.0f;
#pragma omp parallel for reduction(+:totalMass) num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    if (bodies[j].state == 0) totalMass += bodies[j].mass;
                }

                float escVelTerm = 2 * G * totalMass;

#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    if (bodies[j].mass <= 0) continue;

                    float distance = glm::length(bodies[j].position);

                    if (distance <= MAX_BOUND) {
                        if (bodies[j].state == 1) bodies[j].state = 0;
                
                    }
                    else {
                        glm::vec3 vel = bodies[j].velocity;
                        float velsq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;

                        if (velsq >= (escVelTerm / max(0.1f, glm::length(bodies[j].position)))) bodies[j].state = 2; // Velocity greater than escape velocity (sqrt(2GM / r)); Comparing squares of both sides here
                        else bodies[j].state = 1; // Drifter but not removed
                    }
                }

                 

                float xMin = FLT_MAX, xMax = -FLT_MAX;
                float yMin = FLT_MAX, yMax = -FLT_MAX;
                float zMin = FLT_MAX, zMax = -FLT_MAX;

                // Dynamic Sizing 
                for (Body& b : bodies) {
                    if (b.state == 1 || b.state == 2) continue;
                    if (b.position.x < xMin) xMin = b.position.x;
                    if (b.position.x > xMax) xMax = b.position.x;
                    if (b.position.y < yMin) yMin = b.position.y;
                    if (b.position.y > yMax) yMax = b.position.y;
                    if (b.position.z < zMin) zMin = b.position.z;
                    if (b.position.z > zMax) zMax = b.position.z;
                }


                
                
                // Initialising universe
                glm::vec3 universeCenter = glm::vec3((xMin + xMax) / 2.0, (yMin + yMax) / 2.0, (zMin + zMax) / 2.0);
                float universeHalfWidth = max({ (xMax - universeCenter.x), (yMax - universeCenter.y), (zMax - universeCenter.z) });
                BoundingBox universeBoundingBox(universeCenter, universeHalfWidth + 1.0f);

                OctNode* root = new OctNode(universeBoundingBox, 0);

                for (Body& b : bodies) {
                    if (b.state == 0) root->insert(&b);
                }


                root->computeMassDistribution();

                float globalMass = root->totalMass;
                glm::vec3 globalCOM = root->centerOfMass;

                

#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    bodies[j].acceleration = glm::vec3(0.0f);
                    if (bodies[j].state == 0) {
                        calculateForce(root, &bodies[j]);
                    }
                    else if (bodies[j].state == 1) {
                        calculateDriftersForce(root, &bodies[j]);
                    }
                }

                // Final KDK Step
#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    if (bodies[j].state == 2) continue;
                    bodies[j].velocity += bodies[j].acceleration * (dt / 2.0f);
                }
                    

                delete root;


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

        // Update GPU
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.size() * sizeof(Body), bodies.data());

        // Draw
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)bodies.size());

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