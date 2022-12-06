
#include "Scene/scene.cuh"
#include "Dependencies/Helpers/helper_cuda.h"

static void copyHostToDevice_positions(s_positions& dst, s_positions& src, int count) {
    checkCudaErrors(cudaMemcpy(dst.x, src.x, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.y, src.y, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.z, src.z, sizeof(float) * count, cudaMemcpyHostToDevice));
}
static void copyHostToDevice_colors(s_colors& dst, s_colors& src, int count) {
    checkCudaErrors(cudaMemcpy(dst.x, src.x, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.y, src.y, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.z, src.z, sizeof(float) * count, cudaMemcpyHostToDevice));
}
static void copyHostToDevice_spheres(s_spheres& dst, s_spheres& src) {
    int count = src.count;
    copyHostToDevice_positions(dst.center, src.center, count);
    checkCudaErrors(cudaMemcpy(dst.radius, src.radius, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.ka, src.ka, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.kd, src.kd, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.ks, src.ks, sizeof(float) * count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.m, src.m, sizeof(int) * count, cudaMemcpyHostToDevice));
    copyHostToDevice_colors(dst.colors, src.colors, count);
}
static void copyHostToDevice_lights(s_lights& dst, s_lights& src) {
    int count = src.count;
    copyHostToDevice_positions(dst.center, src.center, count);
    copyHostToDevice_colors(dst.colors, src.colors, count);
}
static void copyHostToDevice_float(float& dst, float& src) {
    checkCudaErrors(cudaMemcpy(&dst, &src, sizeof(float), cudaMemcpyHostToDevice));
}
static void copyHostToDevice_float3(float3& dst, float3& src) {
    checkCudaErrors(cudaMemcpy(&dst.x, &src.x, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.y, &src.y, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.z, &src.z, sizeof(float), cudaMemcpyHostToDevice));
}

static void copyHostMemoryToDevice(s_scene* dst, s_scene* src, s_scene* helper, bool copySceneElements) {
    checkCudaErrors(cudaMemcpy(dst, src, sizeof(s_scene), cudaMemcpyHostToDevice));
    if (copySceneElements == true) {
        copyHostToDevice_spheres(helper->spheres, src->spheres);
        copyHostToDevice_lights(helper->lights, src->lights);
    }

    checkCudaErrors(cudaMemcpy(&dst->spheres.center.x, &helper->spheres.center.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.center.y, &helper->spheres.center.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.center.z, &helper->spheres.center.z, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.radius, &helper->spheres.radius, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.ka, &helper->spheres.ka, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.kd, &helper->spheres.kd, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.ks, &helper->spheres.ks, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.m, &helper->spheres.m, sizeof(int*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.colors.x, &helper->spheres.colors.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.colors.y, &helper->spheres.colors.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->spheres.colors.z, &helper->spheres.colors.z, sizeof(float*), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(&dst->lights.center.x, &helper->lights.center.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->lights.center.y, &helper->lights.center.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->lights.center.z, &helper->lights.center.z, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->lights.colors.x, &helper->lights.colors.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->lights.colors.y, &helper->lights.colors.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst->lights.colors.z, &helper->lights.colors.z, sizeof(float*), cudaMemcpyHostToDevice));
}

//static void copyDeviceToHost(uchar4* dst, uchar4* src, int w, int h) {
//    checkCudaErrors(cudaMemcpy(dst, src, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
//}