#pragma once
#include "scene.cuh"

static float getRandomint(int start, int end) {
    return start + rand() / (RAND_MAX / (end - start));
}

static float getRandomfloat(float start, float end) {
    return start + static_cast <float> (rand()) / (static_cast <float> ((float)RAND_MAX / (end - start)));
}

static float3 getRandomfloat3(float start, float end) {
    float r1 = getRandomfloat(start, end);
    float r2 = getRandomfloat(start, end);
    float r3 = getRandomfloat(start, end);
    return make_float3(r1, r2, r3);
}

static void Scene1(s_scene* scene) {
    scene->camera.position.x = 2500.0f;
    scene->camera.position.y = 2500.0f;
    scene->camera.position.z = 350.0f;
    look_at(scene->camera, make_float3(2500, 2500, 0));

    setSphere(scene->spheres, 1, make_float3(2500, 2500, 0), 100, make_float3(1, 1, 1));

    for (int i = 0; i < 3; i++) {
        setLight(
            scene->lights,
            i,
            getRandomfloat3(2000, 3000),
            getRandomfloat3(0, 1.0f)
        );
    }

    setLight(scene->lights, 0, make_float3(2000, 2000, -150), getRandomfloat3(0, 1.0f));
    setLight(scene->lights, 1, make_float3(3000, 2000, 300), getRandomfloat3(0, 1.0f));
    setLight(scene->lights, 2, make_float3(2000, 3000, 0), getRandomfloat3(0, 1.0f));

    for (int i = 3; i < 10; i++) {
        setLight(
            scene->lights,
            i,
            make_float3(0, 0, 0),
            make_float3(0, 0, 0)
        );
    }
}

// Generates a scene with random sphere placement
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
            getRandomfloat(0, 1),
            getRandomint(32, 128)
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