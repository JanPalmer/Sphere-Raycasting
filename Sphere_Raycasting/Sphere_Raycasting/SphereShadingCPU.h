#pragma once
#include <vector_types.h>
#include "vec3.h"
#include "color.h"
#include "ray.h"

float hit_sphere(const point3& center, float radius, const ray& r);
color ray_color(const ray& r);
void renderImageCPU(uchar4* dst, int w, int h);