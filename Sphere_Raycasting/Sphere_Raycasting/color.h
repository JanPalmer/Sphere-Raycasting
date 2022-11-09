#pragma once
#include "vec3.h"
#include <iostream>
#include <vector_types.h>

#define BACKGROUNDCOLOR 0

static void write_color(std::ostream& out, color pixel_color) {
	// Write the translated [0, 255] value of each color component
	out << static_cast<int>(255.999 * pixel_color.x()) << ' '
		<< static_cast<int>(255.999 * pixel_color.y()) << ' '
		<< static_cast<int>(255.999 * pixel_color.z());
}

static uchar3 ConvertColorToUchar3(const color& col) {
    unsigned char r, g, b;
    r = static_cast<int>(255.999 * col.x());
    g = static_cast<int>(255.999 * col.y());
    b = static_cast<int>(255.999 * col.z());
    uchar3 uc;
    uc.x = r; uc.y = g; uc.z = b;
    return uc;
}

static void setColor(uchar4& dst, int red, int green, int blue) {
    if (&dst == nullptr) {
        dst.x = 0;
        dst.y = 0;
        dst.z = 0;
        return;
    }

    dst.x = red;
    dst.y = green;
    dst.z = blue;
}

static void setColor(uchar4& dst, color& col) {
    if (&dst == nullptr) {
        dst.x = 0;
        dst.y = 0;
        dst.z = 0;
        return;
    }

    uchar3 color = ConvertColorToUchar3(col);

    dst.x = color.x;
    dst.y = color.y;
    dst.z = color.z;
}