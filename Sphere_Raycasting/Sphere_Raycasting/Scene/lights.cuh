#ifndef LIGHTS_H
#define LIGHTS_H

#include "../basic_types.cuh"
#include "../colors.cuh"

struct s_lights
{
	int count;
	s_positions center;
	s_colors colors;
};

__host__ inline void setLight(s_lights& lights, int i, float3 center, float3 color) {
	if (i < 0 || i >= lights.count) return;

	lights.center.x[i] = center.x;
	lights.center.y[i] = center.y;
	lights.center.z[i] = center.z;

	lights.colors.x[i] = color.x;
	lights.colors.y[i] = color.y;
	lights.colors.z[i] = color.z;
}

#endif