﻿
#include <stdio.h>

#include "Dependencies/GL/glew.h"
#include "Dependencies/GL/freeglut.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "Dependencies/Helpers/helper_cuda.h"
#include "Dependencies/Helpers/helper_gl.h"
#include "Dependencies/Helpers/helper_timer.h"

#include "SphereShading.cuh"
#include "Scene/SceneCreator.h"
#include "DataCopying.cuh"

#define REFRESH_DELAY 10  // ms
#define SPHERE_COUNT 1000
#define LIGHT_COUNT 10

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource* cuda_pbo_resource;  // handles OpenGL-CUDA exchange

// Source image on the host side
uchar4* h_Src = NULL;

// Destination image on the GPU side
uchar4* d_dst = NULL;

// Check whether frames should be rendered by CPU or GPU
bool g_runcpu = false;

// Original image width and height
// Application allows window resizing
int imageW = 800, imageH = 600;

// Thread count per dimension
const int tx = 16, ty = 16;

int numSMs = 0;   // number of multiprocessors
int version = 1;  // Compute Capability

// Scenes used for holding the sphere, light and camera data
s_scene* h_scene; // host scene
s_scene d_scene_allocationhelper; // used as a helper to hold d_scene's nested arrays
s_scene* d_scene; // device (GPU) scene

// Center of the scene (used for camera rotation)
float3 center = { 0, 0, 0 };

// Camera rotation variables
int ox, oy;
float angle_x = 0, angle_y = 0;
float start_angle_x, start_angle_y;
float angle = 0;

// Timer ID
StopWatchInterface* hTimer = NULL;
StopWatchInterface* globalTimer = NULL;

int frameCount, fpsCount, fpsLimit = 15;
float deltaTime = 0;
float lastFrameTime = 0;
float currentFrameTime = 0;
float framesPerSecond;

float copyToDeviceTime;
float calculationTime;
float copyToHostTime;
float timeFor15frames;
int frameiterator = 0;

#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define BUFFER_DATA(i) ((char *)0 + i)

static void computeFPS_CPU(float renderTime) {
    if (frameiterator < 15) {
        frameiterator++;
        timeFor15frames += renderTime;
    }
    else {
        char fps[256];
        sprintf(fps, "<CUDA %s Set> %3.1f fps, %5.8f render time",
            "Sphere Shading", 1.0f / timeFor15frames * 15.0f, timeFor15frames / 15.0f);
        glutSetWindowTitle(fps);
        frameiterator = 0;
        timeFor15frames = 0;
    }
}

static void computeFPS_GPU(float copyToDeviceTime, float calculationTime, float copyToHostTime) {
    if (frameiterator < 15) {
        frameiterator++;
        timeFor15frames += copyToDeviceTime + calculationTime + copyToHostTime;
    }
    else {
        char fps[256];
        sprintf(fps, "<CUDA %s Set> %3.1f fps, %5.8f render time, %1.4f, %1.4f, %1.4f",
            "Sphere Shading", 1.0f / timeFor15frames * 15.0f, timeFor15frames / 15.0f,
            copyToDeviceTime, calculationTime, copyToHostTime);
        glutSetWindowTitle(fps);
        frameiterator = 0;
        timeFor15frames = 0;
    }
}

// gl_Shader for displaying floating-point texture
static const char* shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";
GLuint compileASMShader(GLenum program_type, const char* code) {
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
        (GLsizei)strlen(code), (GLubyte*)code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1) {
        const GLubyte* error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
            error_string);
        return 0;
    }

    return program_id;
}

void initOpenGLBuffers(int w, int h) {
    // delete old buffers
    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }

    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO) {
        // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4*)malloc(w * h * 4);

    //printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
        h_Src);
    //printf("Texture created.\n");

    //printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is only used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it only once.

    // DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
    //printf("PBO created.\n");

    checkCudaErrors(cudaMalloc(&d_dst, (w * h * 4) * sizeof(uchar4)));
    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

static void keyboard(unsigned char key, int /*x*/, int /*y*/) {

    s_camera* cam = &h_scene->camera;

	switch (key) {
	case 'w':
        cam->position += deltaTime * cam->speed * (-cam->direction);
		break;
	case 's':
        cam->position += deltaTime * cam->speed * cam->direction;
		break;
	case 'a':
        cam->position += deltaTime * cam->speed * cam->left;
        look_at(h_scene->camera, center);
		break;
	case 'd':
        cam->position += deltaTime * cam->speed * cam->right;
        look_at(h_scene->camera, center);
		break;
    case 'g':
        g_runcpu = true;
        break;
    case 'G':
        g_runcpu = false;
        break;
    case '1':
        Scene1(h_scene);
        copyHostMemoryToDevice(d_scene, h_scene, &d_scene_allocationhelper, true);
        break;
    case '2':
        SceneRandom(h_scene, SPHERE_COUNT, LIGHT_COUNT);
        copyHostMemoryToDevice(d_scene, h_scene, &d_scene_allocationhelper, true);
        break;
	}

    glutPostRedisplay();
}

// Used to turn the camera
void clickFunc(int button, int state, int x, int y) {

    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        ox = x;
        oy = y;
        start_angle_x = angle_x;
        start_angle_y = angle_y;
    }

    glutPostRedisplay();
}

// Allows for small camera turning when dragging the mouse over the viewport
void motionFunc(int x, int y) {
    
    angle_x = start_angle_x + ((float)(x - ox) / 10.0f);
    angle_y = start_angle_y + ((float)(y - oy) / 10.0f);

    float yaw = -deltaTime * h_scene->camera.speed * angle_x;
    float pitch = deltaTime * h_scene->camera.speed * angle_y;

    look_at(h_scene->camera, make_float3(yaw, pitch, 0 ));

    glutPostRedisplay();
}

// Fix camera to the new aspect ratio
// and set imageW and imageH to call the correct number of threads when rendering
static void reshape(int w, int h) {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w != 0 && h != 0) {
        // Do not call when window is minimized that is when
                           // width && height == 0
        initOpenGLBuffers(w, h);
    }
    imageW = w;
    imageH = h;
    set_resolution(h_scene->camera, imageH, imageW);
    look_at(h_scene->camera, make_float3(0, 0, 0));

    frameiterator = 0;
    timeFor15frames = 0;

    printf("Viewport - width: %d, height: %d\n", imageW, imageH);

    glutPostRedisplay();
}

// Deallocating every resource used
static void cleanup() {
    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }

    delete h_scene->spheres.center.x;
    delete h_scene->spheres.center.y;
    delete h_scene->spheres.center.z;
    delete h_scene->spheres.colors.x;
    delete h_scene->spheres.colors.y;
    delete h_scene->spheres.colors.z;
    delete h_scene->spheres.ka;
    delete h_scene->spheres.kd;
    delete h_scene->spheres.ks;
    delete h_scene->spheres.m;
    delete h_scene->spheres.radius;
    delete h_scene->lights.center.x;
    delete h_scene->lights.center.y;
    delete h_scene->lights.center.z;
    delete h_scene->lights.colors.x;
    delete h_scene->lights.colors.y;
    delete h_scene->lights.colors.z;
    delete h_scene;

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    sdkStopTimer(&globalTimer);
    sdkDeleteTimer(&globalTimer);

    // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);

    cudaFree(d_dst);

    cudaFree(d_scene_allocationhelper.spheres.center.x);
    cudaFree(d_scene_allocationhelper.spheres.center.y);
    cudaFree(d_scene_allocationhelper.spheres.center.z);
    cudaFree(d_scene_allocationhelper.spheres.colors.x);
    cudaFree(d_scene_allocationhelper.spheres.colors.y);
    cudaFree(d_scene_allocationhelper.spheres.colors.z);
    cudaFree(d_scene_allocationhelper.spheres.ka);
    cudaFree(d_scene_allocationhelper.spheres.kd);
    cudaFree(d_scene_allocationhelper.spheres.ks);
    cudaFree(d_scene_allocationhelper.spheres.m);
    cudaFree(d_scene_allocationhelper.spheres.radius);
    cudaFree(d_scene_allocationhelper.lights.center.x);
    cudaFree(d_scene_allocationhelper.lights.center.y);
    cudaFree(d_scene_allocationhelper.lights.center.z);
    cudaFree(d_scene_allocationhelper.lights.colors.x);
    cudaFree(d_scene_allocationhelper.lights.colors.y);
    cudaFree(d_scene_allocationhelper.lights.colors.z);
    cudaFree(d_scene);
}

// Function for rendering one frame
void renderImage(bool runcpu) {
    if (runcpu == true) {
		sdkResetTimer(&hTimer);

		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
			(void**)&d_dst, &num_bytes, cuda_pbo_resource));

        // Function for generating the rendered bitmap
        renderImageCPU(h_Src, imageW, imageH, *h_scene);

        // Copying the bitmap to GPU for display
        checkCudaErrors(cudaMemcpy(d_dst, h_Src, imageW * imageH * sizeof(uchar4),
            cudaMemcpyHostToDevice));

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        computeFPS_CPU(0.001f * sdkGetTimerValue(&hTimer));
    }
    else {
        
        sdkResetTimer(&hTimer);

        dim3 blocks(imageW / tx + 1, imageH / ty + 1);
        dim3 threads(tx, ty);

        copyHostMemoryToDevice(d_scene, h_scene, &d_scene_allocationhelper, false);

        copyToDeviceTime = 0.001f * sdkGetTimerValue(&hTimer);
        sdkResetTimer(&hTimer);

        renderImageGPU<<<blocks, threads>>>(d_dst, imageW, imageH, *d_scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        calculationTime = 0.001f * sdkGetTimerValue(&hTimer);
        sdkResetTimer(&hTimer);

        copyToHostTime = 0.001f * sdkGetTimerValue(&hTimer);

        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            (void**)&d_dst, &num_bytes, cuda_pbo_resource));

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        computeFPS_GPU(copyToDeviceTime, calculationTime, copyToHostTime);
    }
}

static void display(void)
{
    sdkStartTimer(&hTimer);

    // render the Scene
    renderImage(g_runcpu);

    // load texture from PBO
    //  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA,
        GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    //  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    sdkStopTimer(&hTimer);
    glutSwapBuffers();

    lastFrameTime = currentFrameTime;
    currentFrameTime = sdkGetTimerValue(&globalTimer);
    deltaTime = currentFrameTime - lastFrameTime;
}

void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

static void idle(void)
{
    glutPostRedisplay();
}

// DATA INITIALIZATION

void initGL(int* argc, char** argv) {
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(clickFunc);
    glutMotionFunc(motionFunc);
    glutReshapeFunc(reshape);
    glutCloseFunc(cleanup);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    if (!isGLVersionSupported(1, 5) ||
        !areGLExtensionsSupported(
            "GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");
}

void initData(int argc, char** argv) {
    // check for hardware double precision support
    int dev = 0;
    dev = findCudaDevice(argc, (const char**)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    version = deviceProp.major * 10 + deviceProp.minor;

    numSMs = deviceProp.multiProcessorCount;

    printf("Data initialization done.\n");
}

void initPositions(s_positions& positions, int count) {
    positions.count = count;
    positions.x = new float[count];
    positions.y = new float[count];
    positions.z = new float[count];
}
void initPositionsCUDA(s_positions& positions, int count) {
    checkCudaErrors(cudaMalloc((void**)&positions.x, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&positions.y, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&positions.z, count * sizeof(float)));
}
void initColors(s_colors& colors, int count) {
    colors.count = count;
    colors.x = new float[count];
    colors.y = new float[count];
    colors.z = new float[count];
}
void initColorsCUDA(s_colors& colors, int count) {
    checkCudaErrors(cudaMalloc((void**)&colors.x, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&colors.y, count * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&colors.z, count * sizeof(float)));
}

void initSpheres(s_scene& scene, int sphere_count, bool useCuda) {

    if (useCuda == true) {
        initPositionsCUDA(scene.spheres.center, sphere_count);
        initColorsCUDA(scene.spheres.colors, sphere_count);

        checkCudaErrors(cudaMalloc((void**)&scene.spheres.radius, sizeof(float) * sphere_count));
        checkCudaErrors(cudaMalloc((void**)&scene.spheres.ka, sizeof(float) * sphere_count));
        checkCudaErrors(cudaMalloc((void**)&scene.spheres.kd, sizeof(float) * sphere_count));
        checkCudaErrors(cudaMalloc((void**)&scene.spheres.ks, sizeof(float) * sphere_count));
        checkCudaErrors(cudaMalloc((void**)&scene.spheres.m, sizeof(int) * sphere_count));
    }
	else {
        scene.spheres.count = sphere_count;
		initPositions(scene.spheres.center, sphere_count);
		initColors(scene.spheres.colors, sphere_count);

        scene.spheres.radius = new float[sphere_count];
        scene.spheres.ka = new float[sphere_count];
        scene.spheres.kd = new float[sphere_count];
        scene.spheres.ks = new float[sphere_count];
        scene.spheres.m = new int[sphere_count];
	}
}
void initLights(s_scene& scene, int light_count, bool useCuda) {

    if (useCuda == true) {
        scene.lights.count = light_count;
        initPositionsCUDA(scene.lights.center, light_count);
        checkCudaErrors(cudaMalloc((void**)&scene.lights.colors.x, light_count * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&scene.lights.colors.y, light_count * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&scene.lights.colors.z, light_count * sizeof(float)));
    }
    else {
        scene.lights.count = light_count;
        initPositions(scene.lights.center, light_count);
        initColors(scene.lights.colors, light_count);
    }
}

void initCamera(s_camera& camera, bool useCuda) {
	camera.position.x = 0.0f;
	camera.position.y = 0.0f;
	camera.position.z = 1.0f;

	set_resolution(h_scene->camera, imageW, imageH);
	look_at(h_scene->camera, make_float3(0, 0, -1));
}

void initScene() {
    h_scene = new s_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(s_scene)));

    initSpheres(*h_scene, SPHERE_COUNT, false);
    initLights(*h_scene, LIGHT_COUNT, false);

    printf("CPU Spheres initialized\n");

    initSpheres(d_scene_allocationhelper, SPHERE_COUNT, true);
    initLights(d_scene_allocationhelper, LIGHT_COUNT, true);

    printf("GPU Spheres initialized\n");

    initCamera(h_scene->camera, false);

    SceneRandom(h_scene, SPHERE_COUNT, LIGHT_COUNT);

    copyHostMemoryToDevice(d_scene, h_scene, &d_scene_allocationhelper, true);

    printf("Scene initialized\n");
}

int main(int argc, char* argv[])
{
    initData(argc, argv);
    initScene();

    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);

    sdkCreateTimer(&hTimer);
    sdkStartTimer(&hTimer);

    sdkCreateTimer(&globalTimer);
    sdkStartTimer(&globalTimer);

    printf("Sphere Rendering App:\n");
    printf("'A' and 'D' - rotate the camera around the center of the sphere cluster\n");
    printf("'W' and 'S' - bring the camera closer or futher from the center\n");
    printf("Use mouse dragging to slightly adjust camera view perspective (experimental)\n");
    printf("'G' - render scene using the GPU (default mode)\n");
    printf("'g' - render scene using the CPU (recommended to decrease window size before switching\n");
    printf("'1' - change scene to one sphere lit up by 3 lights with random colors\n");
    printf("'2' - change scene to 1000 spheres lit up by 10 lights (default)\n");
    printf("(repeated clicking reloads the scenes, changing the lighting)\n");

    glutMainLoop();

    return 0;
}