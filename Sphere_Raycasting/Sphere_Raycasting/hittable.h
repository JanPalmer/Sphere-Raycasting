#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "color.h"

struct hit_record {
	point3 p;
	vec3 normal;
	float t, ks, kd, ka;
	color object_color;
};

class hittable {
public:
	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};
#endif