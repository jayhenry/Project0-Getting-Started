#include "termcolor.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// LOOK: This is a macro that calls a CUDA function and checks for errors
#define CUDA(call) do {                             \
    cudaError_t e = (call);                         \
    if (e == cudaSuccess) break;                    \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
            __LINE__, cudaGetErrorString(e), e);    \
    exit(1);                                        \
} while (0)

// This function divides up the n by div - similar to ceil
// Example, divup(10, 3) = 4
inline unsigned divup(unsigned size, unsigned div)
{
    // TODO 5: implement a 1 line function to return the divup operation.
    // Note: You only need to use addition, subtraction, and division operations.
    return 0;
}

// Check errors
bool postprocess(const float *ref, const float *res, unsigned size)
{
    bool passed = true;
    for (unsigned i = 0; i < size; i++)
    {
        // LOOK: Check if floating point values are equal within an epsilon as returns can vary slightly between CPU and GPU
        if (std::fabs(res[i] - ref[i]) > 1e-6)
        {
            std::cout << "ID: " << i << " \t Res: " << res[i] << " \t Ref: " << ref[i] << std::endl;
            std::cout << termcolor::blink << termcolor::white << termcolor::on_red << "*** FAILED ***" << termcolor::reset << std::endl;
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << termcolor::green << "Post process check passed!!" << termcolor::reset << std::endl;

    return passed;
}

void preprocess(float *res, float *dev_res, unsigned size)
{
    const float defaultFill = -1.0;
    std::fill(res, res + size, defaultFill);

    // LOOK: See how we fill the array with the same number to clear it.
    CUDA(cudaMemset(dev_res, defaultFill, size * sizeof(float)));
}

__global__ void saxpy(float* const z, const float* const x, const float* const y, const float a, const unsigned size)
{
    // TODO 9: Compute the global index for each thread.
    unsigned idx = 0;

    // TODO 10: Check if idx is out of bounds. If yes, return.
    if (idx >= 0)
        return;

    // TODO 11: Perform the SAXPY operation: z = a * x + y.
}

int main(int argc, char *argv[])
{
    // TODO 1: Set the size. Start with something simple like 64.
    // TODO Optional: Try out these sizes: 256, 1024, 2048, 14, 103, 1025, 3127
    const unsigned size = 0;

    // Host arrays.
    float* x = new float[size];
    float* y = new float[size];
    float* z = new float[size];

    // LOOK: We use this "gold" array to store the CPU result to be compared with GPU result
    float* z_gold = new float[size];

    // LOOK: Setup random number generator and fill host arrays and the scalar a.
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Fill matrix x and y, then a
    for (unsigned i = 0; i < size; i++) {
        x[i] = dist(mt);
        y[i] = dist(mt);
    }
    const float a = dist(mt);

    // Compute "gold" reference standard
    for (unsigned i = 0; i < size; i++)
        z_gold[i] = a * x[i] + y[i];

    // Device arrays
    float *d_x, *d_y, *d_z;

    // TODO 2: Allocate memory on the device. Fill in the blanks for d_x, then do the same commands for d_y and d_z.
    // CUDA(cudaMalloc((void **)& pointer, size in bytes)));

    // TODO 3: Copy array contents of X and Y from the host (CPU) to the device (GPU). Follow what you did for 2,
    // CUDA(cudaMemcpy(dest ptr, source ptr, size in bytes, direction enum));

    CUDA(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////
    std::cout << "****************************************************" << std::endl;
    std::cout << "***SAXPY***" << std::endl;

    // LOOK: Use the preprocess function to clear z and d_z
    preprocess(z, d_z, size);

    // TODO 4: Setup threads and blocks.
    // Start threadPerBlock as 128, then try out differnt configurations: 32, 64, 256, 512, 1024
    // Use divup to get the number of blocks to launch.
    const unsigned threadsPerBlock = 0;
    const unsigned blocks = divup(size, threadsPerBlock);

    // TODO 6: Launch the GPU kernel with blocks and threadPerBlock as launch configuration
    // saxpy<<< >>> (....);

    // TODO 7: Copy the answer back to the host (CPU) from the device (GPU).
    // Copy what you did in 3, except for d_z -> z.

    // LOOK: Use postprocess to check the result
    postprocess(z_gold, z, size);
    std::cout << "****************************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    // TODO 8: free device memory using cudaFree
    // CUDA(cudaFree(device pointer));

    // free host memory
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] z_gold;

    // successful program termination
    return 0;
}
