#ifndef CAMERA_H
#define CAMERA_H

#define EPS 0.0001f

#include "../basic_types.cuh"

struct s_camera
{
	float3 position;
	float3 direction;
	float speed = 0.1f;

    float3 left;
    float3 right;

    float3 up;
    float3 down;

    float3 left_down;
    float3 right_up;

    float3 viewport_width;
    float3 viewport_height;

    float resolution_horizontal;
    float resolution_vertical;

    float aspect_ratio;

    float fov;
};

// Directs camera at given point in 3D space
__device__ __host__ inline void look_at(s_camera& camera, float x, float y, float z) {
    //camera.direction.x = x - camera.position.x;
    //camera.direction.y = y - camera.position.y;
    //camera.direction.z = z - camera.position.z;
    camera.direction.x = camera.position.x - x;
    camera.direction.y = camera.position.y - y;
    camera.direction.z = camera.position.z - z;
    normalize(camera.direction);

    camera.up = make_float3(0, 1, 0);
    camera.down = make_float3(0, -1, 0);
    camera.left = make_float3(-1, 0, 0);
    camera.right = make_float3(1, 0, 0);

    float3 a, b;

    if (abs(camera.direction.x) < EPS && abs(camera.direction.y) < EPS) {
        a = { 0, -camera.direction.z, 0 };
        b = { 0, camera.direction.z, 0 };
    }
    else {
        a = { camera.direction.y, - camera.direction.x, 0 };
        b = { -camera.direction.y, camera.direction.x, 0 };
    }

    //camera.left = camera.aspect_ratio * normalize(a);
    //camera.right = camera.aspect_ratio * normalize(b);

    //camera.up = normalize(cross(camera.left, camera.direction));
    //camera.down = normalize(cross(camera.right, camera.direction));

    camera.left_down = camera.position + (camera.left + camera.down);
    camera.right_up = camera.position + (camera.right + camera.up);
    camera.viewport_height = (camera.up - camera.down);
    camera.viewport_width = (camera.right - camera.left) * camera.aspect_ratio;
}

// Directs camera at given point in 3D space
__device__ __host__ inline void look_at(s_camera& camera, float3& point)
{
    look_at(camera, point.x, point.y, point.z);
}

// Sets resolution of camera
__device__ __host__ inline void set_resolution(s_camera& camera, int resolution_horizontal, int resolution_vertical)
{
    camera.resolution_horizontal = (float)resolution_horizontal;
    camera.resolution_vertical = (float)resolution_vertical;
    camera.aspect_ratio = (float)resolution_horizontal / (float)resolution_vertical;
}

#endif