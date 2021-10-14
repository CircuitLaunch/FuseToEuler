#include "vecadd.cuh"

#include <cuda_runtime.h>
#include "Compute.hpp"

__device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__global__ void cuda_vecadd(const float3 *a, const float3 *b, float3 *c, int count)
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
    const Compute::Buffer<float3> *devA = va_arg(args, const Compute::Buffer<float3> *);
    const Compute::Buffer<float3> *devB = va_arg(args, const Compute::Buffer<float3> *);
    Compute::Buffer<float3> *devC = va_arg(args, Compute::Buffer<float3> *);
    int s = va_arg(args, int);
    va_end(args);
    const float3 *a = devA->getDeviceBuffer();
    const float3 *b = devB->getDeviceBuffer();
    float3 *c = devC->getDeviceBuffer();

    cuda_vecadd<<<iBlocksPerGrid, iThreadsPerBlock>>>(a, b, c, s);
    cudaError_t err = cudaGetLastError();
    return err;
}
