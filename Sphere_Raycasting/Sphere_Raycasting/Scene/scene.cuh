#ifndef SCENE_H
#define SCENE_H

#include "../Objects/sphere.cuh"
#include "lights.cuh"
#include "camera.cuh"

struct s_scene {
	s_camera camera;
	s_lights lights;
	s_spheres spheres;
};

#endif