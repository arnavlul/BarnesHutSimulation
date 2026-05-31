# BarnesHutSimulation

A C++ implementation of the Barnes-Hut algorithm for efficient N-body simulation. 

## Overview
The Barnes-Hut algorithm is an approximation algorithm for performing an n-body simulation. It is notable for having order $O(n \log n)$ complexity compared to the brute-force $O(n^2)$ algorithm. This project visualizes the particles and their interactions in real-time.

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