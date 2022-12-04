#ifndef BASIC_H
#define BASIC_H

#include <vector_types.h>

struct s_positions
{
	int count;
	float* x, * y, * z;
	float* angle;
};

struct s_hit_record {
	float3 p; // hit point
	float3 normal; // normal vector
	float t; // length coefficient
	int i; // index of the hit sphere
};

struct s_ray
{
	float3 origin;
	float3 direction;
};

__host__ __device__ inline float3 getPosition(const s_positions& positions, int i) {
	return make_float3(positions.x[i], positions.y[i], positions.z[i]);
}

__host__ __device__ inline float3 point_at_t(const s_ray& ray, float t) {
	return ray.origin + t * ray.direction;
}

#endif