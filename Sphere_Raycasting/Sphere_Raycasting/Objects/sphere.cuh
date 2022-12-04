#ifndef SPHERE_H
#define SPHERE_H

#include "../Dependencies/Helpers/helper_math.h"

#include "../basic_types.cuh"
#include "../colors.cuh"

struct s_spheres
{
	int count;
	s_positions center;
	float *radius;
	s_colors colors; // color of the object
	float *ks, *kd, *ka; // coefficients for specular, diffuse and ambient light
};

__host__ inline void setSphere(s_spheres& spheres, int i, float3 center, float radius, float3 color, float ka = 0.1f, float kd = 0.7f, float ks = 0.7f) {
	if (i < 0 || i >= spheres.count) return;

	spheres.center.x[i] = center.x;
	spheres.center.y[i] = center.y;
	spheres.center.z[i] = center.z;
	spheres.radius[i] = radius;
	spheres.colors.x[i] = color.x;
	spheres.colors.y[i] = color.y;
	spheres.colors.z[i] = color.z;
	spheres.ka[i] = ka;
	spheres.kd[i] = kd;
	spheres.ks[i] = ks;
}

__host__ __device__ inline bool hit_checksingle(const s_ray &r, const s_spheres& spheres, int i, float t_min, float t_max, s_hit_record& rec) {
	float3 center = make_float3(spheres.center.x[i], spheres.center.y[i], spheres.center.z[i]);
	float radius = spheres.radius[i];

	float3 oc = r.origin - center;
	float a = dot(r.direction, r.direction);
	float half_b = dot(oc, r.direction);
	float c = dot(oc, oc) - radius * radius;
	
	float discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	float sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range
	float root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max) return false;
	}

	rec.t = root;
	rec.p = point_at_t(r, rec.t);
	rec.normal = (rec.p - center) / radius;
	rec.i = i;
	return true;
}

__host__ __device__ inline bool hit_checkall(const s_ray& r, const s_spheres& spheres, float t_min, float t_max, s_hit_record& rec) {
	s_hit_record temp_rec;
	bool hit_anything = false;
	float closest_so_far = t_max;

	for (int i = 0; i < spheres.count; i++) {
		if (hit_checksingle(r, spheres, i, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

#endif