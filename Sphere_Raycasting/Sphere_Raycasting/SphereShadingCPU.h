#pragma once
#include <vector_types.h>

#include "Basics/rtconstants.cuh"
#include "Basics/color.cuh"
#include "Objects/hittable_list.cuh"
#include "Objects/sphere.cuh"
#include "Scene/lights.cuh"
#include "Scene/camera.cuh"

color shade_point(hit_record& rec, const hittable& world, const lights_list& lights, const camera& cam);
color ray_color(const ray& r, const hittable& world, const lights_list& lights, const camera& cam);
void renderImageCPU(uchar4* dst, int w, int h, const hittable& world, const lights_list& lights, const camera& cam);