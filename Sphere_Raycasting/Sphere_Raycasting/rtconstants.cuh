#ifndef RTCONSTANTS_H
#define RTCONSTANTS_H

#include <cmath>
#include <limits>
#include <memory>

// Constants
const float infinity = std::numeric_limits<float>::max();
const float pi = 3.1415926535897932385f;
const float pi_over_two = pi / 2.0f;

// Utility functions

inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.0f;
}

// Common headers


#endif