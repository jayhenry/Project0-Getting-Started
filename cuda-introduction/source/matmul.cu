#include "termcolor.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// LOOK: Use DIMS as a struct to store launch configurations.
// dimBlock is dimension of block (threads), and dimGrid is dimension of the launch (number of blocks).
// dim3 is a CUDA provided type, which as 3 components - x, y, z, which are initialized by default to 1.
struct DIMS
{
    dim3 dimBlock;
    dim3 dimGrid;
};

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
    return (size + div - 1) / div;
}

// Check errors
bool postprocess(const float *ref, const float *res, unsigned n)
{
    bool passed = true;
    for (unsigned i = 0; i < n; i++)
    {
        if (std::fabs(res[i] - ref[i]) > 1e-3)
        {
            std::cout << "ID: " << i << " \t Res: " << res[i] << " \t Ref: " << ref[i] << std::endl;
            std::cout << termcolor::blink << termcolor::white << termcolor::on_red << "*** FAILED ***" << termcolor::reset << std::endl;
            passed = false;
            break;
        }
    }

    if (passed)
    {
        std::cout << termcolor::green << "Post process check passed!!" << termcolor::reset << std::endl;
    }

    return passed;
}

void preprocess(float *res, float *dev_res, unsigned n)
{
    const float defaultFill = -1.0;
    std::fill(res, res + n, defaultFill);

    // LOOK: See how we fill the array with the same number to clear it.
    CUDA(cudaMemset(dev_res, defaultFill, n * sizeof(float)));
}

// TODO 10: Implement the matrix multiplication kernel
__global__ void matrixMultiplicationNaive(float* const matrixP, const float* const matrixM, const float* const matrixN,
                                          const unsigned sizeMX, const unsigned sizeNY, const unsigned sizeXY)
{
    // TODO 10a: Compute the P matrix global index for each thread along x and y dimentions.
    // Remember that each thread of the kernel computes the result of 1 unique element of P
    unsigned px;
    unsigned py;

    // TODO 10b: Check if px or py are out of bounds. If they are, return.

    // TODO 10c: Compute the dot product for the P element in each thread
    // This loop will be the same as the host loop
    float dot = 0.0;

    // TODO 10d: Copy dot to P matrix
    // matrixP[] = dot;
}

int main(int argc, char *argv[])
{
    // TODO 1: Initialize sizes. Start with simple like 16x16, then try 32x32.
    // Then try large multiple-block square matrix like 64x64 up to 2048x2048.
    // Then try square, non-power-of-two like 15x15, 33x33, 67x67, 123x123, and 771x771
    // Then try rectangles with powers of two and then non-power-of-two.
    const unsigned sizeMX = 0;
    const unsigned sizeXY = 0;
    const unsigned sizeNY = 0;

    // TODO 2: Allocate host 1D arrays for:
    // matrixM[sizeMX, sizeXY]
    // matrixN[sizeXY, sizeNY]
    // matrixP[sizeMX, sizeNY]
    // matrixPGold[sizeMX, sizeNY]
    float* matrixM;
    float* matrixN;
    float* matrixP;
    float* matrixPGold;

    // LOOK: Setup random number generator and fill host arrays and the scalar a.
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Fill matrix M on host
    for (unsigned i = 0; i < sizeMX * sizeXY; i++)
        matrixM[i] = dist(mt);

    // Fill matrix N on host
    for (unsigned i = 0; i < sizeXY * sizeNY; i++)
        matrixN[i] = dist(mt);

    // TODO 3: Compute "gold" reference standard
    // for py -> 0 to sizeNY
    //   for px -> 0 to sizeMX
    //     initialize dot product accumulator
    //     for k -> 0 to sizeXY
    //       dot = m[k, px] * n[py, k]
    //  matrixPGold[py, px] = dot

    // Device arrays
    float *d_matrixM, *d_matrixN, *d_matrixP;

    // TODO 4: Allocate memory on the device for d_matrixM, d_matrixN, d_matrixP.

    // TODO 5: Copy array contents of M and N from the host (CPU) to the device (GPU)

    CUDA(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////
    std::cout << "****************************************************" << std::endl;
    std::cout << "***Matrix Multiplication***" << std::endl;

    // LOOK: Use the preprocess function to clear matrixP and d_matrixP
    preprocess(matrixP, d_matrixP, sizeMX * sizeNY);

    // TODO 6: Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid" using divup
    // HINT: The shape of matrices has no impact on launch configuaration
    DIMS dims;
    dims.dimBlock = dim3(1, 1, 1);
    dims.dimGrid  = dim3(divup(sizeMX, dims.dimBlock.x),
                         divup(sizeNY, dims.dimBlock.y),
                         1);

    // TODO 7: Launch the matrix transpose kernel
    // matrixMultiplicationNaive<<<>>>();

    // TODO 8: copy the answer back to the host (CPU) from the device (GPU)

    // LOOK: Use postprocess to check the result
    postprocess(matrixPGold, matrixP, sizeMX * sizeNY);

    std::cout << "****************************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    // TODO 9: free device memory using cudaFree

    // free host memory
    delete[] matrixM;
    delete[] matrixN;
    delete[] matrixP;
    delete[] matrixPGold;

    // successful program termination
    return 0;
}
