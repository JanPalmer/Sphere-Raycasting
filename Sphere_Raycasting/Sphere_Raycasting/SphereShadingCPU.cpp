// Sphere Raycasting on CPU

#include "SphereShadingCPU.h"

float hit_sphere(const point3& center, float radius, const ray& r) {
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) {
		return -1.0f;
	}
	else {
		return (-half_b - sqrt(discriminant)) / a;
	}
}

color ray_color(const ray& r) {
	auto t = hit_sphere(point3(0, 0, -1.0f), 0.5f, r);
	if (t > 0.0f) {
		vec3 N = unit_vector(r.at(t) -vec3(0, 0, -1.0f));
		return 0.5f * color(N.x() + 1.0f, N.y() + 1.0f, N.z() + 1.0f);
	}
	vec3 unit_direction = unit_vector(r.direction());
	t = 0.5f * (unit_direction.y() + 1);

	return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

void renderImageCPU(uchar4* dst, int w, int h) {

	float viewport_height = 2.0f;
	float viewport_width = ((float)w / (float)h) * viewport_height;
	float focal_length = 1.0f;

	auto origin = point3(0, 0, 0);
	auto horizontal = vec3(viewport_width, 0, 0);
	auto vertical = vec3(0, viewport_height, 0);
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

	int i;
	for (int iy = 0; iy < h; iy++)
		for (int ix = 0; ix < w; ix++) {
			i = w * iy + ix;
			auto u = float(ix) / (w - 1);
			auto v = float(iy) / (h - 1);
			ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
			color pixel_color = ray_color(r);
			setColor(dst[i], pixel_color);
		}
}