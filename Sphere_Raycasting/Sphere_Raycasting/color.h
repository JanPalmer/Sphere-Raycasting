#ifndef COLOR_H
#define COLOR_H

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
    float rf, gf, bf;
    rf = 255.999 * col.x();
    gf = 255.999 * col.y();
    bf = 255.999 * col.z();

    r = static_cast<int>((rf > 255) ? 255 : rf);
    g = static_cast<int>((gf > 255) ? 255 : gf);
    b = static_cast<int>((bf > 255) ? 255 : bf);
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

    dst.x = (red > 255) ? 255 : red;
    dst.y = (green > 255) ? 255 : green;
    dst.z = (blue > 255) ? 255 : blue;
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

    //dst.x = (color.x > 255) ? 255 : color.x;
    //dst.y = (color.y > 255) ? 255 : color.y;
    //dst.z = (color.z > 255) ? 255 : color.z;
}

#endif