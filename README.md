# GPU-Accelerated Barnes-Hut N-Body Simulation

An ultra-high-performance, CUDA-accelerated 3D N-body gravity simulator and cinematic renderer. This project implements the Barnes-Hut algorithm with a fully parallelized Karras Octree builder directly on the GPU, allowing for the simulation of hundreds of thousands of particles at real-time framerates. 

## Key Features

- **100% GPU-Accelerated Physics**: Uses advanced CUDA techniques including Structure of Arrays (SoA) memory layout, Morton space-filling curves, hardware Read-Only Cache (`__ldg()`), and zero-divergence Warp execution.
- **Direct CUDA-OpenGL Interoperability**: Real-time rendering pipes memory directly from the physics compute kernels to OpenGL Vertex Buffer Objects without ever transferring data to the CPU.
- **Memory-Mapped Offline Streaming**: Simulate systems at maximum speed headlessly, save them to disk (with FP16 compression support), and stream gigabytes of simulation data instantly using Windows Memory Mapping.
- **Interactive Cinematic Director**: Fly around the simulation, plot custom bezier camera paths using keyframes, and export directly to MP4 via FFmpeg.
- **Post-Processing Pipeline**: Includes adjustable HDR Tone-Mapping, Gaussian Bloom, and Accumulation-based Motion Blur.

---

## Launch Modes

Upon launching the application, you are presented with the ImGui Launcher Menu. There are four primary tabs/modes of operation:

### 1. Real-Time Engine
Run the physics calculations and the OpenGL renderer simultaneously in real-time.
- **Initial Condition**: Choose from 6 different procedural starting scenarios (see below).
- **Particle Count**: Define the number of bodies (N) in the simulation.
- **Barnes-Hut Theta**: Controls the approximation threshold (0.1 to 1.0). Lower values are more accurate but slower; higher values are faster but less accurate.
- **Enable V-Sync**: Caps the frame rate to your monitor's refresh rate to prevent screen tearing.

### 2. Pre-Calculate Offline
Run the physics engine headlessly (without rendering graphics) to simulate at maximum hardware speed. This is ideal for very high particle counts or small `dt` time-steps.
- **Physics dt**: Time step per integration tick. Smaller values improve numerical stability at the cost of requiring more ticks per frame.
- **Simulation Duration (s)** & **Target Playback FPS**: Dictates how many total frames will be simulated and exported.
- **Use FP16 Storage**: Compresses the exported position/velocity data by 50% using half-precision floating-point format, drastically reducing file size and improving SSD streaming speed.
- **Auto-Chunk Files**: For massive simulations (e.g., >5GB), automatically splits the output `.bhsim` files into chunks to prevent memory-mapping limits.

### 3. Playback Renderer
Instantly scrub through and watch previously exported `.bhsim` offline simulations.
- **Select Simulation**: Dropdown to select any `.bhsim` file in the `simulation files/` directory.

### 4. Cinematic MP4 Export
Render your simulation out to a final, high-quality `.mp4` video.
- **Interactive Custom Spline Editor & Post-Processing**:
  - Contains a **"CONFIGURE BLOOM & MOTION BLUR"** button that drops you into the viewport to adjust visual sliders and save them globally.
  - Contains an **"OPEN INTERACTIVE CINEMATIC DIRECTOR"** button to plot camera keyframes over time.
- **Use Pre-Programmed Demo Camera Spline**: Check this to automatically render using a smooth, pre-defined orbiting camera path.
- **Manual Camera Controls**: If the demo spline is disabled, manually adjust the fixed *Camera Radius*, *Camera Pitch*, and *Camera Yaw* for a static tripod shot.

---

## Starting Scenarios (Initial Conditions)

When launching a new simulation, you can select from the following procedural setups:

1. **Standard Spiral Galaxy**: A stable, rotating disc of stars orbiting a supermassive central black hole, complete with velocity dispersion for organic clumping.
2. **Galaxy Collision (The Milky Way & Andromeda)**: Two massive spiral galaxies placed on a collision course to demonstrate violent tidal forces and sweeping tails.
3. **Binary Accretion System**: Two supermassive black holes in a tight orbit, tearing apart and consuming a surrounding cloud of stars.
4. **Cosmological Web**: Was supposed to collapse into filament like structures, but just coalesces into a ball if waiting long enough.
5. **The Disrupted Ring**: A stable, hollow torus of stars that slowly succumbs to gravitational instabilities, breaking apart into dense nodes.
6. **The Cluster 3-Body Problem**: Three exceptionally dense, ultra-massive globular clusters arranged in an equilateral triangle and given chaotic intersecting velocities. They tear each other apart in a spectacular display of the 3-body problem.

---

## In-Viewport Controls (Renderer)

While inside the Real-Time or Playback viewport, an ImGui panel provides the following controls:

- **Playback Speed / Frame Slider**: Pause, play, or scrub through the simulation timeline (Playback mode only).
- **Lock Camera to Center of Mass**: Automatically calculates the center of mass of the entire system (excluding escaped drifters) and locks the camera's focus to it, keeping chaotic systems perfectly centered.
- **Heatmap Max Speed**: Adjusts the color-coding threshold. Slower particles are drawn red/orange, while fast particles approach bright blue/white.
- **Enable Bloom / Bloom Exposure**: Toggles the HDR multi-pass Gaussian blur to make dense clusters and fast stars glow intensely.
- **Enable Motion Blur / Motion Blur Strength**: Toggles the accumulation buffer. Higher strength results in longer, smoother star trails.
- **Interactive MP4 Exporter (Keyframe System)**:
  - Move your camera, click **"Add Spline Keyframe"**, and scrub to a different frame. 
  - The engine will automatically interpolate a smooth cinematic path between your keyframes.
  - Click **"RENDER MP4 NOW"** to instantly export your directed shot to video.
- **Save Sliders & Return to Launcher**: Exits the viewport, permanently saves your post-processing slider settings, and returns you to the main menu.
