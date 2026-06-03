# BarnesHutSimulation

A C++ implementation of the Barnes-Hut algorithm for efficient N-body simulation. 

## Overview
This project is an N-body simulation engine implementing the Barnes-Hut algorithm for efficient computation. The Barnes-Hut algorithm groups nearby bodies into a tree structure, allowing distant clusters to be approximated as a single larger mass, which provides order $O(n \log n)$ complexity compared to the $O(n^2)$ complexity of a naive simulation. The project visualizes the particles and their interactions in real-time.

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