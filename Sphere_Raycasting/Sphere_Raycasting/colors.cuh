#ifndef COLOR_H
#define COLOR_H

#include <iostream>
#include <vector_types.h>

#define BACKGROUNDCOLOR 0

struct s_colors
{
    int count;
    float *x, *y, *z;
};

__host__ __device__ inline float3 getColor(const s_colors& colors, int i) {
    return make_float3(colors.x[i], colors.y[i], colors.z[i]);
}

inline void write_color(std::ostream& out, float3 pixel_color) {
	// Write the translated [0, 255] value of each color component
	out << static_cast<int>(255.999f * pixel_color.x) << ' '
		<< static_cast<int>(255.999f * pixel_color.y) << ' '
		<< static_cast<int>(255.999f * pixel_color.z);
}

__host__ __device__ inline uchar3 ConvertColorToUchar3(const float x, const float y, const float z) {
    unsigned char r, g, b;
    float rf, gf, bf;
    rf = 255.999f * x;
    gf = 255.999f * y;
    bf = 255.999f * z;

    r = static_cast<int>((rf > 255) ? 255 : rf);
    g = static_cast<int>((gf > 255) ? 255 : gf);
    b = static_cast<int>((bf > 255) ? 255 : bf);
    uchar3 uc;
    uc.x = r; uc.y = g; uc.z = b;
    return uc;
}
__host__ __device__ inline uchar3 ConvertColorToUchar3(const float3& color) {
    return ConvertColorToUchar3(color.x, color.y, color.z);
}

__host__ __device__ inline void setColor(uchar4& dst, const float x, const float y, const float z) {
    if (&dst == nullptr) {
        dst.x = 0;
        dst.y = 0;
        dst.z = 0;
        return;
    }

    uchar3 color = ConvertColorToUchar3(x, y, z);

    dst.x = color.x;
    dst.y = color.y;
    dst.z = color.z;
}
__host__ __device__ inline void setColor(uchar4& dst, float3& col) {
    setColor(dst, col.x, col.y, col.z);
}
__host__ __device__ inline void setColor(uchar4& dst, int red, int green, int blue) {
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

#endif