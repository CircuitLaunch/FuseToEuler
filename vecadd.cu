#include "vecadd.cuh"

#include <cuda_runtime.h>
#include "Compute.hpp"

__global__ void cuda_vecadd(const float *a, const float *b, float *c, int count)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < count) {
        c[i] = a[i] + b[i];
    }
}

cudaError_t VecAdd::operator()(size_t iBlocksPerGrid, size_t iThreadsPerBlock, ...)
{
    va_list args;
    va_start(args, iThreadsPerBlock);
    const Compute::Buffer *devA = va_arg(args, const Compute::Buffer *);
    const Compute::Buffer *devB = va_arg(args, const Compute::Buffer *);
    Compute::Buffer *devC = va_arg(args, Compute::Buffer *);
    int s = va_arg(args, int);
    va_end(args);
    const float *a = devA->getDeviceBuffer();
    const float *b = devB->getDeviceBuffer();
    float *c = devC->getDeviceBuffer();

    cuda_vecadd<<<iBlocksPerGrid, iThreadsPerBlock>>>(a, b, c, s);
    cudaError_t err = cudaGetLastError();
    return err;
}
