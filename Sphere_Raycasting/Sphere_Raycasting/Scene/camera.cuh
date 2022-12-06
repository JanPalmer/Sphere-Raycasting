#ifndef CAMERA_H
#define CAMERA_H

#include "../basic_types.cuh"

struct s_camera
{
    float3 position;
	float3 direction;
	float speed = 1.5f;

    float3 left;
    float3 right;

    float3 up;
    float3 down;

    float3 left_down;
    float3 right_up;

    float3 viewport_width;
    float3 viewport_height;

    float aspect_ratio;
};

__host__ inline float3 normalizeFloat3(float3 vector) {
    float length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    float3 result = make_float3(vector.x / length, vector.y / length, vector.z / length);
    return result;
}

// Directs camera at given point in 3D space
__device__ __host__ inline void look_at(s_camera& camera, float x, float y, float z) {
    camera.direction.x = camera.position.x - x;
    camera.direction.y = camera.position.y - y;
    camera.direction.z = camera.position.z - z;
    camera.direction = normalizeFloat3(camera.direction);

    float3 a, b;
    a = make_float3(-camera.direction.z, 0, camera.direction.x);
    b = make_float3(camera.direction.z, 0, -camera.direction.x);
	camera.left = normalizeFloat3(a);
    camera.right = normalizeFloat3(b);

    camera.up = normalizeFloat3(cross(camera.left, camera.direction));
    camera.down = normalizeFloat3(cross(camera.right, camera.direction));

    camera.left_down = camera.position + (camera.left + camera.down);
    camera.right_up = camera.position + (camera.right + camera.up);
    camera.viewport_height = (camera.up - camera.down);
    camera.viewport_width = (camera.right - camera.left) / camera.aspect_ratio;
}

// Directs camera at given point in 3D space
__device__ __host__ inline void look_at(s_camera& camera, float3& point)
{
    look_at(camera, point.x, point.y, point.z);
}

// Sets resolution of camera
__device__ __host__ inline void set_resolution(s_camera& camera, int resolution_horizontal, int resolution_vertical)
{
    camera.aspect_ratio = (float)resolution_horizontal / (float)resolution_vertical;
}

#endif