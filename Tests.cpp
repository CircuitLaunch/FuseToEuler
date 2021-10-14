#include "LinearAlgebra.hpp"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "Compute.hpp"
#include "vecadd.cuh"

using namespace std;

/*
__host__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

bool operator>(const float3 &a, float s)
{
    return a.x > s || a.y > s || a.z > s;
}

float3 fabs(const float3 &a)
{
    return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}
*/

int main(int argc, char **argv)
{
    float a0[] = { 3.0, 4.0, 5.0, 2.0, 6.0, 1.0, 1.0, 4.0, 4.0, -2.0, 5.0, 8.0, 2.0, 8.0, 7.0, 3.0 };
    mat4 m0(a0);

    cout << m0.determinant() << endl;

    float a1[] = { 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
    mat4 m1(a1);
    mat4 m2(m1.reciprocal());

    int j = 16;
    while(j--) {
        cout << "    " << m2[15 - j];
        if(!(j&3)) cout << endl;
    }

    size_t size = 50000;

    float3 *hostA, *hostB, *hostC;
    hostA = new float3[size];
    hostB = new float3[size];
    hostC = new float3[size];

    // Initialize the host input vectors
    for (int i = 0; i < size; ++i)
    {
        hostA[i].x = rand()/(float)RAND_MAX;
        hostA[i].y = rand()/(float)RAND_MAX;
        hostA[i].z = rand()/(float)RAND_MAX;
        hostB[i].x = rand()/(float)RAND_MAX;
        hostB[i].y = rand()/(float)RAND_MAX;
        hostB[i].z = rand()/(float)RAND_MAX;
    }

    Compute::Buffer<float3>
        devA = Compute::createBuffer<float3>(size, hostA),
        devB = Compute::createBuffer<float3>(size, hostB),
        devC = Compute::createBuffer<float3>(size);

    VecAdd functor;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t err = functor(blocksPerGrid, threadsPerBlock, &devA, &devB, &devC);

    if(err != cudaSuccess) {
        cerr << "vecadd failed: " << err << " " << cudaGetErrorString(err) << endl;
    } else {
        err = devC.copyToHost(hostC, size);
        if(err != cudaSuccess) {
            cerr << "Failed to retrieve results from device: " << err << " " << cudaGetErrorString(err) << endl;
        } else {
            bool valid = true;
            for(size_t i = 0; i < size; i++) {
                if(fabs(hostA[i].x + hostB[i].x - hostC[i].x) > 1e-5) {
                    cerr << "vecadd results incorrect!" << endl;
                    valid = false;
                    break;
                }
                if(fabs(hostA[i].y + hostB[i].y - hostC[i].y) > 1e-5) {
                    cerr << "vecadd results incorrect!" << endl;
                    valid = false;
                    break;
                }
                if(fabs(hostA[i].z + hostB[i].z - hostC[i].z) > 1e-5) {
                    cerr << "vecadd results incorrect!" << endl;
                    valid = false;
                    break;
                }
            }
            if(valid) {
                cout << "vecadd verified" << endl;
            }
        }
    }

    delete [] hostA;
    delete [] hostB;
    delete [] hostC;

    cout << "OK\n";
}
