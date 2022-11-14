#pragma once
#include <vector_types.h>

#include "rtconstants.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "lights.h"
#include "camera.h"

color shade_point(hit_record& rec, const hittable& world, const lights_list& lights, const camera& cam);
color ray_color(const ray& r, const hittable& world, const lights_list& lights, const camera& cam);
void renderImageCPU(uchar4* dst, int w, int h, const hittable& world, const lights_list& lights, const camera& cam);