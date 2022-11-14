#ifndef LIGHTS_H
#define LIGHTS_H

#include <memory>
#include <vector>

#include "vec3.h"

using std::shared_ptr;
using std::make_shared;

class light{
public:
	light() {}
	light(point3 pos, color col) : position(pos), color_light(col) {}

public:
	point3 position;
	const color color_light;
};

class lights_list {
public:
	lights_list() {}
	lights_list(shared_ptr<light> object) { add(object); }

	void clear() { objects.clear(); }
	void add(shared_ptr<light> object) { objects.push_back(object); }

public:
	std::vector<shared_ptr<light>> objects;
};


#endif