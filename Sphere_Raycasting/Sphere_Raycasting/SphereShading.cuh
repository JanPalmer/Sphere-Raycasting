#pragma once
#include <vector_types.h>

#include "rtconstants.cuh"
#include "Scene/scene.cuh"


__host__ __device__ float3 shade_point(s_hit_record& rec, const s_scene& scene);
__host__ __device__ float3 ray_color(const s_ray& r, const s_scene& scene);
__global__ void renderImageGPU(uchar4* dst, int width, int height, const s_scene& scene);
__host__ void renderImageCPU(uchar4* dst, int width, int height, const s_scene& scene);