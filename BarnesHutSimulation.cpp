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

extern "C" {
    __declspec(dllexport) uint32_t NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

using namespace std;

const float SOFTENING = 0.1f;
const float G = 1.0f;
const float dt = 0.01f;
float THETA = 0.5f;

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

void generateUniformUniverse(vector<Body>& bodies, const int &particleCount, float spawnRange) {

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
        b.force = glm::vec3(0.0f);
        b.oldforce = glm::vec3(0.0f);
        b.mass = massGen(gen);
        bodies.push_back(b);
    }
}

void generateBlackHoleCentricUniverse(vector<Body>& bodies, const int& particleCount, float spawnRange) {

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
    centralCore.force = glm::vec3(0.0, 0.0, 0.0);
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
        b.force = glm::vec3(0.0f);
        b.oldforce = glm::vec3(0.0f);
        b.mass = massGen(gen);

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

            float forceMag = (G * b->mass * targetBody->mass) / (distanceSq + SOFTENING * SOFTENING);
            targetBody->force += (direction / distance) * forceMag;
        }
    }
    else {
        float width = node->bbox.halfWidth * 2.0f;
        glm::vec3 direction = node->centerOfMass - targetBody->position;
        float distanceSq = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;

        if ((width * width) < (THETA * THETA * distanceSq)) { // Treat node as one body
            float distance = sqrt(distanceSq);
            float forceMag = (G * node->totalMass * targetBody->mass) / (distanceSq + SOFTENING * SOFTENING);
            targetBody->force += (direction / distance) * forceMag;
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

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
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
    generateUniformUniverse(bodies, 1000, 300);
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

            ImGui::SliderInt("CPu Threads", &activeThreads, 1, maxThreads);

            ImGui::Text("Application average: %.3f ms/frame; FPS: (%.1f FPS)", 1000.f / io.Framerate, io.Framerate);


            ImGui::End();
        }

        if (!pauseSimulation) {
            for (int i = 0; i < simulationSpeed; i++) {

#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    glm::vec3 acceleration = bodies[j].force / bodies[j].mass;
                    bodies[j].position += bodies[j].velocity * dt + 0.5f * acceleration * dt * dt;
                    bodies[j].oldforce = bodies[j].force;
                    bodies[j].force = glm::vec3(0.0f);
                }

                float xMin = FLT_MAX, xMax = -FLT_MAX;
                float yMin = FLT_MAX, yMax = -FLT_MAX;
                float zMin = FLT_MAX, zMax = -FLT_MAX;

                // Dynamic Sizing 
                for (Body& b : bodies) {
                    if (b.position.x < xMin) xMin = b.position.x;
                    if (b.position.x > xMax) xMax = b.position.x;
                    if (b.position.y < yMin) yMin = b.position.y;
                    if (b.position.y > yMax) yMax = b.position.y;
                    if (b.position.z < zMin) zMin = b.position.z;
                    if (b.position.z > zMax) zMax = b.position.z;
                }

                glm::vec3 universeCenter = glm::vec3((xMin + xMax) / 2.0, (yMin + yMax) / 2.0, (zMin + zMax) / 2.0);
                float universeHalfWidth = max({ (xMax - universeCenter.x), (yMax - universeCenter.y), (zMax - universeCenter.z) });
                BoundingBox universeBoundingBox(universeCenter, universeHalfWidth + 1.0f);

                OctNode* root = new OctNode(universeBoundingBox, 0);

                for (Body& b : bodies) root->insert(&b);

                root->computeMassDistribution();

#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < (int)bodies.size(); j++) {
                    calculateForce(root, &bodies[j]);
                }

                delete root;

#pragma omp parallel for num_threads(activeThreads)
                for (int j = 0; j < bodies.size(); j++) {
                    glm::vec3 oldacceleration = bodies[j].oldforce / bodies[j].mass;
                    glm::vec3 newacceleration = bodies[j].force / bodies[j].mass;
                    bodies[j].velocity += 0.5f * (oldacceleration + newacceleration) * dt;
                }

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