// Sphere Raycasting on CPU

#include "SphereShading.cuh"

__host__ __device__ float3 shade_point(s_hit_record& rec, const s_scene& scene) {
	float3 ambientsum = make_float3(0, 0, 0);
	float3 diffusesum = make_float3(0, 0, 0);
	float3 specularsum = make_float3(0, 0, 0);

	float3 sphere_color = getColor(scene.spheres.colors, rec.i);
	float ka = scene.spheres.ka[rec.i];
	float kd = scene.spheres.kd[rec.i];
	float ks = scene.spheres.ks[rec.i];
	int m = scene.spheres.m[rec.i];

	float3 normal = normalize(rec.normal);
	float3 viewPosition = scene.camera.position;
	float3 viewVec = normalize(viewPosition - rec.p);
	s_hit_record tmp;

	for (int i = 0; i < scene.lights.count; i++) {

		float3 lightPos = getPosition(scene.lights.center, i);

		// Check whether current light doesn't hit any other sphere on its way to this point (looks pretty dark)
		//s_ray r = { lightPos, rec.p};
		//if (hit_checkall(r, scene.spheres, 0, 1000000.0f, tmp) == false) continue;
		//if (tmp.i != rec.i) continue;

		float3 lightColor = getColor(scene.lights.colors, i);
		if (lightColor.x < 0 || lightColor.y < 0 || lightColor.z < 0) continue;
		float3 lightVec = normalize(lightPos - rec.p);


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
	sphere_color = sphere_color * (ambientsum + diffusesum + specularsum);

	return sphere_color;
}



__host__ __device__ float3 ray_color(const s_ray& r, const s_scene& scene) {
	s_hit_record rec;
	if (hit_checkall(r, scene.spheres, 0, 10000000.0f, rec)) {
		return shade_point(rec, scene);
	}

	return { 0.0f, 0.0f, 0.0f };
}

__global__ void renderImageGPU(uchar4* dst, int width, int height, const s_scene& scene) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height) return;

	auto origin = scene.camera.position;
	auto horizontal = scene.camera.viewport_width;
	auto vertical = scene.camera.viewport_height;
	auto lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - (scene.camera.direction * 2.0f);

	s_ray r = { origin, lower_left_corner + x / (float)width * horizontal  + y / (float)height * vertical - origin };
	float3 pixel_color = ray_color(r, scene);
	setColor(dst[y * width + x], pixel_color);

}

__host__ void renderImageCPU(uchar4* dst, int width, int height, const s_scene& scene) {

	auto origin = scene.camera.position;
	auto horizontal = scene.camera.viewport_width;
	auto vertical = scene.camera.viewport_height;
	auto lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - (scene.camera.direction * 2.0f);

	for(int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {
			s_ray r = { origin, lower_left_corner + x / (float)width * horizontal + y / (float)height * vertical - origin };
			float3 pixel_color = ray_color(r, scene);
			setColor(dst[y * width + x], pixel_color);
		}
}