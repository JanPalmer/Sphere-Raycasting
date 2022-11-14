#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"

class camera {
public:
	camera() {}
	camera(point3 position, vec3 right, vec3 up) : pos(position), right(right), up(up)
	{
		forward = cross(right, up);
	}

	point3 getPos() const { return pos; }
	void setPos(point3 newPos) { pos = newPos; }
	void setForward(vec3 newFront) { forward = newFront; }
	void setRight(vec3 newRight) { right = newRight; }
	void setUp(vec3 newUp) { up = newUp; }

	void move(point3 deltaMove) { pos += deltaMove; }
	void rotate(float angle) {  }
public:
	point3 pos;
	vec3 forward, right, up;
	float speed = 0.0001f;
};

#endif