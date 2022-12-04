// Sphere Raycasting on CPU

#include "SphereShadingCPU.h"

using std::shared_ptr;

color shade_point(hit_record& rec, const hittable& world, const lights_list& lights, const camera& cam) {
	color col = rec.object_color;
	color ambientsum = color(0, 0, 0);
	color diffusesum = color(0, 0, 0);
	color specularsum = color(0, 0, 0);
	int m = 32;

	for (shared_ptr<light> l : lights.objects) {
		vec3 lightVec = unit_vector(l.get()->position - rec.p);

		ray lightRay = ray(l.get()->position, lightVec);
		hit_record lightrec;
		if (world.hit(lightRay, 0, infinity, lightrec)) {
			if (lightrec.t - rec.t > 0.01f) continue;
		}

		//std::cout << rec.ka << " | " << l.get()->color_light << "\n";
		ambientsum += l.get()->color_light;

		vec3 norm = unit_vector(rec.normal);
		float diff = std::max(dot(norm, lightVec), 0.0f);
		color diffuse = diff * l.get()->color_light;
		diffusesum += diffuse;

		vec3 viewVec = unit_vector(cam.getPos() - rec.p);
		vec3 reflectVec = unit_vector(-lightVec - 2 * dot(-lightVec, norm) * norm);
		float spec = pow(std::max(dot(viewVec, reflectVec), 0.0f), m);
		color specular = spec * l.get()->color_light;
		specularsum += specular;

		//std::cout << ambient << ' ' << diffuse << ' ' << specular << "\n";	
	}
	ambientsum *= rec.ka;
	diffusesum *= rec.kd;
	specularsum *= rec.ks;

	col = col * (ambientsum + diffusesum + specularsum);
	//std::cout << col << "\n";
	return col;
}

color ray_color(const ray& r, const hittable& world, const lights_list& lights, const camera& cam) {
	hit_record rec;

	if (world.hit(r, 0, infinity, rec)) {
		return shade_point(rec, world, lights, cam);
	}
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1);

	return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

void renderImageCPU(uchar4* dst, int w, int h, const hittable& world, const lights_list& lights, const camera& cam) {

	float viewport_height = 5.0f;
	float viewport_width = ((float)w / (float)h) * viewport_height;
	float focal_length = 2.0f;

	auto origin = cam.pos;
	auto horizontal = cam.right * viewport_width;
	auto vertical = cam.up * viewport_height;
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

	int i;
	for (int iy = 0; iy < h; iy++)
		for (int ix = 0; ix < w; ix++) {
			i = w * iy + ix;
			auto u = float(ix) / (w - 1);
			auto v = float(iy) / (h - 1);
			ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
			color pixel_color = ray_color(r, world, lights, cam);
			setColor(dst[i], pixel_color);
		}
}