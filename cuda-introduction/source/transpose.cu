#include "termcolor.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * *****************************************************************************
 * README FIRST
 * In this example, we'll implement both a Copy Kernel and a Transpose Kernel.
 * The only difference in the two kernels is the index we use for the actual copy / transpose operation.
 * In copy, the destination index is same as source index. In transpose, the destination index is transpose of source index.
 * In this exercise, first get the copy kernel working correctly, which is simpler. Then move to transpose.
 * *****************************************************************************
 */

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
    // TODO: implement a 1 line function to return the divup operation.
    // Note: You only need to use addition, subtraction, and division operations.
    return 0;
}

// Check errors
bool postprocess(const float *ref, const float *res, unsigned n)
{
    bool passed = true;
    for (unsigned i = 0; i < n; i++)
    {
        if (res[i] != ref[i])
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

// TODO 6: Implement the copy kernel
__global__ void copyKernel(const float* const a, float* const b, const unsigned sizeX, const unsigned sizeY)
{
    // TODO 6a: Compute the global index for each thread along x and y dimentions.
    unsigned i = 0;
    unsigned j = 0;;

    // TODO 6b: Check if i or j are out of bounds. If they are, return.

    // TODO 6c: Compute global 1D index from i and j
    unsigned index = 0;

    // TODO 6d: Copy data from A to B. Note that in copy kernel source and destination indices are the same
    // b[] = a[];
}

// TODO 10: Implement the transpose kernel
// Start by copying everything from the copy kernel.
// Then make the change to compute different index_in and index_out from i and j
// Then change the final operation to use the correct index variables.
__global__ void matrixTransposeNaive(const float* const a, float* const b, const unsigned sizeX, const unsigned sizeY)
{
    // TODO 10a: Compute the global index for each thread along x and y dimentions.
    unsigned i = 0;
    unsigned j = 0;

    // TODO 10b: Check if i or j are out of bounds. If they are, return.

    // TODO 10c: Compute index_in as (i,j) (same as index in copy kernel) and index_out as (j,i)
    unsigned index_in  = 0;  // Compute input index (i,j) from matrix A
    unsigned index_out = 0;  // Compute output index (j,i) in matrix B = transpose(A)

    // TODO 10d: Copy data from A to B using transpose indices
}

int main(int argc, char *argv[])
{
    // TODO: Initialize sizes. Start with simple like 32 x 32.
    // TODO Optional: Try different sizes - both square and non-square. Use these as examples:
    // 1024 x 1024, 2048 x 2048, 64 x 16, 128 x 768, 63 x 63, 31 x 15, 1025 x 1025, 1234 x 3153
    const unsigned sizeX = 1234;
    const unsigned sizeY = 3153;

    // LOOK: Allocate host arrays. The gold arrays are used to store the results from CPU.
    float* a = new float[sizeX * sizeY];
    float* b = new float[sizeX * sizeY];
    float* a_gold = new float[sizeX * sizeY];
    float* b_gold = new float[sizeX * sizeY];

    // Fill matrix A
    for (unsigned i = 0; i < sizeX * sizeY; i++)
        a[i] = (float)i;

    // Compute "gold" reference standard
    for (unsigned jj = 0; jj < sizeY; jj++)
    {
        for (unsigned ii = 0; ii < sizeX; ii++)
        {
            a_gold[jj * sizeX + ii] = a[jj * sizeX + ii]; // Reference for copy kernel
            b_gold[ii * sizeY + jj] = a[jj * sizeX + ii]; // Reference for transpose kernel
        }
    }

    // Device arrays
    float *d_a, *d_b;

    // TODO 1: Allocate memory on the device for d_a and d_b.

    // TODO 2: Copy array contents of A from the host (CPU) to the device (GPU)

    CUDA(cudaDeviceSynchronize());

    ////////////////////////////////////////////////////////////
    std::cout << "****************************************************" << std::endl;
    std::cout << "***Device To Device Copy***" << std::endl;
    {
        // LOOK: Use the preprocess function to clear b and d_b
        preprocess(b, d_b, sizeX * sizeY);

        // TODO 3: Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid" using divup
        DIMS dims;
        dims.dimBlock = dim3(1, 1, 1);

        // LOOK: Launch the copy kernel
        copyKernel<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b, sizeX, sizeY);

        // TODO 4: copy the answer back to the host (CPU) from the device (GPU)

        // LOOK: Use postprocess to check the result
        postprocess(a_gold, b, sizeX * sizeY);
    }
    std::cout << "****************************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    std::cout << "****************************************************" << std::endl;
    std::cout << "***Naive Transpose***" << std::endl;
    {
        // LOOK: Use the preprocess function to clear b and d_b
        preprocess(b, d_b, sizeX * sizeY);

        // TODO 7: Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid" using divup
        DIMS dims;
        dims.dimBlock = dim3(1, 1, 1);
        dims.dimGrid  = dim3(divup(sizeX, dims.dimBlock.x),
                             divup(sizeY, dims.dimBlock.y),
                             1);

        // TODO 8: Launch the matrix transpose kernel
        // matrixTransposeNaive<<<>>>(......);

        // TODO 9: copy the answer back to the host (CPU) from the device (GPU)

        // LOOK: Use postprocess to check the result
        postprocess(b_gold, b, sizeX * sizeY);
    }
    std::cout << "****************************************************" << std::endl << std::endl;
    ////////////////////////////////////////////////////////////

    // TODO 5: free device memory using cudaFree

    // free host memory
    delete[] a;
    delete[] b;

    // successful program termination
    return 0;
}
