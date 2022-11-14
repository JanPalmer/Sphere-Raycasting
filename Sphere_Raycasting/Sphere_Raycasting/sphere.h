#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
	sphere() {}

	sphere(
		point3 cen, 
		float r, 
		color col = color(1.0f, 1.0f, 1.0f), 
		float k_s = 0.5f, 
		float k_d = 0.5f,
		float k_a = 0.1f)
		: center(cen), radius(r), object_color(col), ks(k_s), kd(k_d), ka(k_a) {}

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
public:
	point3 center;
	float radius;
	color object_color; // color of the object
	float ks, kd, ka; // coefficients for specular, diffuse and ambient light
};

inline bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	auto sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range
	auto root = (-half_b - sqrtd) / a;
	if (root < t_min || root > t_max) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || root > t_max) return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	rec.normal = (rec.p - center) / radius;
	rec.object_color = object_color;
	rec.ks = ks; rec.kd = kd; rec.ka = ka;

	return true;
}
#endif