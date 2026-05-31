# BarnesHutSimulation

A C++ implementation of the Barnes-Hut algorithm for efficient N-body simulation. 

## Overview
This project is a hybrid N-body simulation engine combining the macroscopic efficiency of the Barnes-Hut algorithm with the microscopic accuracy of a Hermite integrator. It features dynamic clustering ("Submarines") to automatically identify and resolve close encounters in dense regions with high mathematical precision. The Barnes-Hut algorithm provides order $O(n \log n)$ complexity for distant particles, while the Hermite integrator handles tight clusters, ensuring stable physics without phase leakage. The project visualizes the particles and their interactions in real-time.

## Technologies Used
- **C++** 
- **OpenGL** via **GLAD** and **GLFW** for rendering and window management
- **GLM** for mathematics
- **ImGui** for the graphical user interface
- **vcpkg** for dependency management

## Building the Project

1. Make sure you have [vcpkg](https://github.com/microsoft/vcpkg) installed and integrated with your development environment (e.g. Visual Studio).
2. Open the solution file `BarnesHutSimulation.slnx` or open the directory in Visual Studio.
3. Dependencies are defined in `vcpkg.json` and will be automatically restored by vcpkg in manifest mode.
4. Build and run the project (x64 configuration).

## License
Please refer to the `LICENSE.txt` file for more details.