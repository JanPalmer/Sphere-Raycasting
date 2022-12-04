// Sphere Raycasting on CPU

#include "SphereShading.cuh"

__host__ __device__ float3 shade_point(s_hit_record& rec, const s_scene& scene) {
	float3 ambientsum = make_float3(0, 0, 0);
	float3 diffusesum = make_float3(0, 0, 0);
	float3 specularsum = make_float3(0, 0, 0);
	int m = 32;

	float3 sphere_color = getColor(scene.spheres.colors, rec.i);
	//float3 sphere_color = make_float3(scene.spheres.color->x[rec.i], scene.spheres.color->y[rec.i], scene.spheres.color->z[rec.i]);
	float ka = scene.spheres.ka[rec.i];
	float kd = scene.spheres.kd[rec.i];
	float ks = scene.spheres.ks[rec.i];

	float3 normal = normalize(rec.normal);
	float3 viewPosition = scene.camera.position;
	float3 viewVec = normalize(viewPosition - rec.p);
	s_hit_record tmp;

	for (int i = 0; i < scene.lights.count; i++) {
		// Check whether current light doesn't hit any other sphere on its way to this point
		float3 lightPos = getPosition(scene.lights.center, i);
		//s_ray r = { lightPos, rec.p};
		//if (hit_checkall(r, scene.spheres, 0, 1000000.0f, tmp) == false) continue;
		//if (tmp.i != rec.i) continue;

		float3 lightColor = getColor(scene.lights.colors, i);
		if (lightColor.x < 0 || lightColor.y < 0 || lightColor.z < 0) continue;

		//float3 lightPos = make_float3(scene.lights.center->x[i], scene.lights.center->y[i], scene.lights.center->z[i]);
		float3 lightVec = normalize(lightPos - rec.p);
		//float3 lightColor = make_float3(scene.lights.colors->x[i], scene.lights.colors->y[i], scene.lights.colors->z[i]);

		
		// Sprawdzenie czy ray dociera do powierzchni?
		//s_ray lightRay = { lightPos, lightVec };

		ambientsum += lightColor;

		float diff = max(dot(normal, lightVec), 0.0f);
		diffusesum += diff * lightColor;

		float3 reflectionVec = normalize(-lightVec - 2 * dot(-lightVec, normal) * normal);
		float spec = pow(max(dot(viewVec, reflectionVec), 0.0f), m);
		specularsum += spec * lightColor;
	}

	ambientsum *= ka;
	diffusesum *= kd;
	specularsum *= ks;

	//printf("Sphere %d - %f %f %f\n", rec.i, ambientsum.x, ambientsum.y, ambientsum.z);
	//printf("Sphere %d - %f %f %f\n", rec.i, diffusesum.x, diffusesum.y, diffusesum.z);
	//printf("Sphere %d - %f %f %f\n", rec.i, specularsum.x, specularsum.y, specularsum.z);
	//printf("Sphere %d - %f %f %f\n", rec.i, sphere_color.x, sphere_color.y, sphere_color.z);
	sphere_color = sphere_color * (ambientsum + diffusesum + specularsum);

	return sphere_color;
}



__host__ __device__ float3 ray_color(const s_ray& r, const s_scene& scene) {
	s_hit_record rec;
	if (hit_checkall(r, scene.spheres, 0, 1000000.0f, rec)) {
		//return { 1, 0, 0 };
		return shade_point(rec, scene);
	}

	return { 0.0f, 0.0f, 0.0f };

	float3 unit_direction = normalize(r.direction);
	float t = 0.5f * (unit_direction.y + 1);

	return (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
}

__global__ void renderImageGPU(uchar4* dst, int width, int height, const s_scene& scene) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height) return;

	auto origin = scene.camera.position;
	//auto horizontal = scene.camera.right * width;
	auto horizontal = scene.camera.viewport_width;
	//auto horizontal = scene.camera.right;
	//auto vertical = scene.camera.up * height;
	auto vertical = scene.camera.viewport_height;
	//auto vertical = scene.camera.up;
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_float3(0, 0, 2.0f);
	//auto lower_left_corner = scene.camera.left_down;

	s_ray r = { origin, lower_left_corner + x / (float)width * horizontal  + y / (float)height * vertical - origin };
	float3 pixel_color = ray_color(r, scene);
	setColor(dst[y * width + x], pixel_color);

}

__host__ void renderImageCPU(uchar4* dst, int width, int height, const s_scene& scene) {

	auto origin = scene.camera.position;
	//auto horizontal = scene.camera.right * scene.camera.viewport_width;
	//auto vertical = scene.camera.up * scene.camera.viewport_height;
	//auto horizontal = scene.camera.viewport_width;
	//auto vertical = scene.camera.viewport_height;
	//auto lower_left_corner = origin - horizontal / 2 - vertical / 2;
	auto horizontal = scene.camera.viewport_width;
	auto vertical = scene.camera.viewport_height;
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_float3(0, 0, 2.0f);
	//float3 lower_left_corner = scene.camera.left_down;
	//float3 camwidth = scene.camera.viewport_width;
	//float3 camheight = scene.camera.viewport_height;

	for(int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			s_ray r = { origin, lower_left_corner + x / (float)width * horizontal + y / (float)height * vertical - origin };
			float3 pixel_color = ray_color(r, scene);
			setColor(dst[y * width + x], pixel_color);
		}
}