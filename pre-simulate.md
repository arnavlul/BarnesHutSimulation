# Offline Pre-Simulation & Playback Plan

This document outlines the architecture for a dual-mode engine: an **Offline Simulation Mode** (maximum performance headless calculation) and a **Playback Mode** (scrubbable, adjustable speed rendering).

## 1. Core Architecture Change
When the application starts, the main menu will offer two primary pathways:
1. **Real-time Engine**: The current setup. Physics and rendering happen simultaneously in the same loop using CUDA-GL Interop.
2. **Pre-Calculate Engine**: The physics simulation runs headlessly (without an OpenGL window). It writes trajectory data directly to disk as fast as the GPU can compute it.
3. **Playback Engine**: An OpenGL window opens, streaming the pre-calculated data from the hard drive into the GPU for pure visual rendering.

## 2. High-Performance Pre-Calculation (The Simulation Mode)
To maximize throughput and finish calculations instantly, we must ensure the GPU never waits for the hard drive, and the hard drive never waits for the GPU.

### Async Double Buffering Strategy
1. **Pinned Host Memory**: We will allocate two "staging buffers" in RAM using `cudaHostAlloc`. This creates "pinned" memory, which allows the CUDA driver to perform lightning-fast Direct Memory Access (DMA) transfers.
2. **CUDA Streams**: We will use two separate CUDA streams. 
   - While the GPU computes Frame N in Stream 1...
   - `cudaMemcpyAsync` will simultaneously copy Frame N-1 from the GPU to Staging Buffer A.
3. **Background I/O Thread**: A dedicated CPU worker thread (`std::thread`) will take Staging Buffer B (which was filled on the previous step) and write it to the NVMe SSD.

By overlapping Compute + Memory Copy + Disk Write, the engine will run at the absolute maximum speed the GPU hardware allows.

## 3. The `.bhsim` Custom Binary Format
To handle massive data output efficiently, we will define a custom binary format (`.bhsim`). 

**Storage Precision Toggle:**
The engine will ALWAYS calculate physics in FP32 to prevent gravity errors (division by zero, precision loss). However, before the simulation starts, the user can choose the saving format:
* **FP32 Storage (High-Res):** Saves the exact 32-bit floats. 1.2 MB per frame for 100k particles.
* **FP16 Storage (Visual-Res):** Uses a fast CUDA kernel to compress the results to 16-bit half-floats before saving. This cuts the file size in half (~600 KB per frame) and doubles disk I/O speed.

**Data Size Estimation (FP16):**
* 100,000 particles * 3 half-floats (x,y,z) * 2 bytes = ~600 Kilobytes per frame.
* 60 seconds of simulation @ 60 ticks/sec (3600 frames) = **~2.1 Gigabytes**.
Modern NVMe SSDs write at 3,000+ MB/s, meaning the disk write will be virtually instantaneous.

**File Structure:**
```text
[Header - 24 bytes]
char[4] magic = "BHSM"
int32 version = 1
int32 particleCount
int32 totalFrames
float32 dt
int32 isFP16 // 1 for FP16, 0 for FP32

[Frame 0 Data]
float3 or half3 positions[particleCount] 

[Frame 1 Data]
float3 or half3 positions[particleCount]
...
```
We will use low-level C-style `std::fwrite` with a massively expanded buffer, or Windows native `WriteFile` with `FILE_FLAG_NO_BUFFERING` for direct-to-disk throughput.

## 4. The Playback Post-Renderer
When reading a 4GB+ file, we do not want to load the entire file into RAM at once, as that would crash the program if scaled to millions of particles.

### Memory Mapped Streaming
We will use **Windows Memory-Mapped Files** (`CreateFileMapping` and `MapViewOfFile`). This tells the Windows OS to map the `.bhsim` file on the SSD directly into virtual memory. 
* As the playback engine requests frames, Windows automatically and transparently streams the data from the SSD into RAM using hardware paging. 
* This allows instant, zero-load-time scrubbing and playback of files that are hundreds of gigabytes in size!

### Rendering Loop & Player Controls
* **Full Timeline Scrubber:** The UI will feature an ImGui timeline slider (like a YouTube player). Because we use Memory Mapped files, dragging the slider instantly jumps to that exact frame in the multi-gigabyte file with zero lag.
* **Speed Dial:** An adjustable 1x-15x playback speed dial will exist alongside the scrubber.
* **Camera Modes:** By default, the camera is free-roam. A new "Lock to Center of Mass" toggle will automatically track the system's gravitational center as it moves through space.
* **Heatmap Rendering:** Since we are now saving velocity data into the `.bhsim` file, a "Heatmap" tickbox will be available in the UI. When checked, a custom OpenGL fragment shader will colorize particles from deep red (slow) to bright blue/white (fast).

## 5. Implementation Steps
1. **Refactor `main.cpp`**: Update the CLI to ask for run mode (Real-time vs Pre-calculate vs Playback), target simulation time, FP16/FP32 preference, and Filename (custom or auto-generated).
2. **Build the Headless Loop**: Create the `runOfflineSimulation` function implementing the double-buffered `cudaMemcpyAsync` and `std::fwrite` logic for both positions and velocities.
3. **Build the Playback Loop**: Create the `runPlayback(filename)` function utilizing Memory Mapped I/O, `glBufferSubData`, and the ImGui timeline player.
4. **Update Shaders**: Write the velocity-based heatmap shader for the post-renderer.
