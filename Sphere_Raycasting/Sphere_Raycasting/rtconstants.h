#ifndef RTCONSTANTS_H
#define RTCONSTANTS_H

#include <cmath>
#include <limits>
#include <memory>

// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;
const float pi_over_two = pi / 2;

// Utility functions

inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

// Common headers

#include "ray.h"
#include "vec3.h"

#endif