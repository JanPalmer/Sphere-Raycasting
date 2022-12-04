#pragma once
#include "scene.cuh"

static void Scene1(s_scene* scene) {
    setSphere(scene->spheres, 0, make_float3(0, 0, -1), 0.5f, make_float3(1, 1, 1));
    setSphere(scene->spheres, 1, make_float3(0, -100.5, -1), 100, make_float3(1, 1, 1));

    setLight(scene->lights, 0, make_float3(0, 10, 0), make_float3(2.0f, 1.2f, 0.0f));
    setLight(scene->lights, 1, make_float3(0, 0, -1), make_float3(0.0f, 1.0f, 0.0f));
    setLight(scene->lights, 2, make_float3(0, 1, 0), make_float3(0.0f, 0.0f, 2.0f));
}

static float getRandomfloat(float start, float end) {
    return start + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (end - start)));
}

static float3 getRandomfloat3(float start, float end) {
    float r1 = getRandomfloat(start, end);
    float r2 = getRandomfloat(start, end);
    float r3 = getRandomfloat(start, end);
    return make_float3(r1, r2, r3);
}

static void SceneRandom(s_scene* scene, int sphere_count, int light_count) {
    scene->camera.position.x = 0.0f;
    scene->camera.position.y = 0.0f;
    scene->camera.position.z = 3000.0f;
    look_at(scene->camera, make_float3(0, 0, 0));

    const float min = -990.0f, max = 990.0f;
    const float minradius = 1.0f, maxradius = 100.0f;

    srand(time(NULL));

    for (int i = 0; i < sphere_count; i++) {
        setSphere(
            scene->spheres,
            i,
            getRandomfloat3(min, max),
            getRandomfloat(minradius, maxradius),
            getRandomfloat3(0, 1),
            getRandomfloat(0, 0.1f),
            getRandomfloat(0, 1),
            getRandomfloat(0, 1)
            );
    }

    for (int i = 0; i < light_count; i++) {
        setLight(
            scene->lights,
            i,
            getRandomfloat3(min, max),
            getRandomfloat3(0, 1.5f)
        );
    }
}