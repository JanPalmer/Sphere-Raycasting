# Sphere Raycasting with Phong shading

Renders a 3D scene showing a 1000 spheres lit by 10 lightsources.

## Description

The app uses Nvidia's CUDA to render each pixel's color in a parallel fashion.
The reflection model is based on the [Phong reflection model](https://en.wikipedia.org/wiki/Phong_reflection_model), with each ray being sent from the rendered pixel forward.\
The project is based on [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) from NVIDIA's Developer Blog.

## Guide
### Controls
'A' and 'D' - rotate the camera around the center of the sphere cluster;\
'W' and 'S' - bring the camera closer or futher from the center;\
Use mouse dragging to slightly adjust camera view perspective
### Render mode
The app supports both CPU rendering and rendering using an Nvidia GPU. To switch between these modes, use the following controls: \
'G' - render scene using the GPU (default mode)\
'g' - render scene using the CPU (recommended to decrease window size before switching)
### Scene changing
'1' - change scene to one sphere lit up by 3 lights with random colors\
'2' - change scene to 1000 spheres lit up by 10 lights (default)\
(repeated clicking reloads the scenes, changing the lighting)

## Project setup
The project utilizes C++, the Nvidia CUDA toolkit 11.7, freeglut 3.4 and glew32. Each dependency is supplied with the software - you can find it in Dependencies/GL (for freeglut and glew) and Dependencies/Helpers (for CUDA helpers).\
The project has been made using Visual Studio 2019.\
To run the project, you'll need to place the following .dll files together with the compiled .exe file:
- freeglutd.dll
- glew32.dll

You can find them in Dependencies/GL.