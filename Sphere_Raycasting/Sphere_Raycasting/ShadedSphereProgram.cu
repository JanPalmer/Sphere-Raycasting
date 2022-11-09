
#include <stdio.h>


#include "Dependencies/GL/glew.h"
#include "Dependencies/GL/freeglut.h"



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "Dependencies/Helpers/helper_cuda.h"
#include "Dependencies/Helpers/helper_gl.h"
#include "Dependencies/Helpers/helper_timer.h"

#include "SphereShadingCPU.h"

#define REFRESH_DELAY 10  // ms

// OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource* cuda_pbo_resource;  // handles OpenGL-CUDA exchange

// Source image on the host side
uchar4* h_Src = 0;

// Destination image on the GPU side
uchar4* d_dst = NULL;

// Check whether frames should be rendered by CPU or GPU
bool g_runcpu = true;

// Original image width and height
int imageW = 800, imageH = 600;

// Starting position and scale
double xOff = -0.5;
double yOff = 0.0;
double scale = 3.2;

// Starting stationary position and scale motion
double xdOff = 0.0;
double ydOff = 0.0;
double dscale = 1.0;

// Starting color multipliers and random seed
int colorSeed = 0;
uchar4 colors;

int numSMs = 0;   // number of multiprocessors
int version = 1;  // Compute Capability

float framesPerSecond;

static int DisplaySize() {
    return imageW * imageH;
}

int frameIndex = 0;

// Timer ID
StopWatchInterface* hTimer = NULL;
// Auto-Verification Code
const int frameCheckNumber = 60;
int fpsCount = 0;   // FPS count for averaging
int fpsLimit = 15;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif
#define BUFFER_DATA(i) ((char *)0 + i)

void computeFPS() {
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&hTimer) / 1000.f);
        framesPerSecond = ifps;
        sprintf(fps, "<CUDA %s Set> %3.1f fps",
            "Mandelbrot", ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, (float)ifps);
        sdkResetTimer(&hTimer);
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

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
        h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
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
    printf("PBO created.\n");

    checkCudaErrors(cudaMalloc(&d_dst, (w * h * 4) * sizeof(uchar4)));

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w != 0 && h != 0)  // Do not call when window is minimized that is when
                           // width && height == 0
        initOpenGLBuffers(w, h);

    imageW = w;
    imageH = h;

    glutPostRedisplay();
}

void cleanup() {
    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }

    sdkStopTimer(&hTimer);
    sdkDeleteTimer(&hTimer);

    // DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);

    cudaFree(d_dst);
}

void renderImage(bool runcpu) {
    if (runcpu) {
        //int startPass = pass;
        float xs, ys;
        sdkResetTimer(&hTimer);

        //if (bUseOpenGL) {
        //    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_dst,
        //    // gl_PBO));
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
            size_t num_bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
                (void**)&d_dst, &num_bytes, cuda_pbo_resource));
        //}

        // Get the pixel scale and offset
        double s = scale / (double)imageW;
        double x = (xs - (double)imageW * 0.5f) * s + xOff;
        double y = (ys - (double)imageH * 0.5f) * s + yOff;

        // Run the mandelbrot generator
        renderImageCPU(h_Src, imageW, imageH);
        
        // Use the adaptive sampling version when animating.

        checkCudaErrors(cudaMemcpy(d_dst, h_Src, imageW * imageH * sizeof(uchar4),
            cudaMemcpyHostToDevice));

        //if (bUseOpenGL) {
        //    // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        //}

#if RUN_TIMING
        printf("CPU = %5.8f\n", 0.001f * sdkGetTimerValue(&hTimer));
#endif
    }
}

static void display(void)
{
    sdkStartTimer(&hTimer);

    // render the Mandelbrot image
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

    printf("Width: % d, Height: %d, FPS: %3.1f\n", imageW, imageH, framesPerSecond);

    computeFPS();
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

void initGL(int* argc, char** argv) {
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    //glutMouseFunc(clickFunc);
    //glutMotionFunc(motionFunc);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    //initMenus();

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

    // initialize some of the arguments
    if (checkCmdLineFlag(argc, (const char**)argv, "xOff")) {
        xOff = getCmdLineArgumentFloat(argc, (const char**)argv, "xOff");
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "yOff")) {
        yOff = getCmdLineArgumentFloat(argc, (const char**)argv, "yOff");
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "scale")) {
        scale = getCmdLineArgumentFloat(argc, (const char**)argv, "xOff");
    }

    colors.w = 0;
    colors.x = 3;
    colors.y = 5;
    colors.z = 7;
    printf("Data initialization done.\n");
}


int main(int argc, char* argv[])
{
    initData(argc, argv);

    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);

    sdkCreateTimer(&hTimer);
    sdkStartTimer(&hTimer);

    glutMainLoop();

    return 0;
}