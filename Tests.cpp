#include "LinearAlgebra.hpp"
#include <stdio.h>
#include <iostream>
#include "Compute.hpp"

using namespace std;

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

    Compute compute;

    switch(compute.getDeviceType()) {
        case CL_DEVICE_TYPE_GPU: cout << "OpenCL device type: GPU\n"; break;
        case CL_DEVICE_TYPE_ACCELERATOR: cout << "OpenCL device type: ACCELERATOR\n"; break;
        case CL_DEVICE_TYPE_CPU: cout << "OpenCL device type: CPU\n"; break;
        case CL_DEVICE_TYPE_DEFAULT: cout << "OpenCL device type: CUSTOM\n"; break;
        case CL_DEVICE_TYPE_CUSTOM: cout << "OpenCL device type: DEFAULT\n"; break;
    }

    Compute::Program program = compute.createProgram(string("vecadd"));
    Compute::Kernel kernel = compute.createKernel(program, string("vecadd"));

    const size_t size = 128;
    cl_float dataA[size];
    cl_float dataB[size];
    cl_float dataC[size];

    for(size_t i = 0; i < size; i++) {
        dataA[i] = (cl_float) i;
        dataA[i] = (cl_float) (size - i);
        dataC[i] = 0.0;
    }

    poclu_bswap_cl_float_array(compute.device, (cl_float *) dataA, 4 * size);
    poclu_bswap_cl_float_array(compute.device, (cl_float *) dataB, 4 * size);

    Compute::Buffer
        bufA = compute.createBuffer(CL_MEM_READ_ONLY, sizeof(dataA)),
        bufB = compute.createBuffer(CL_MEM_READ_ONLY, sizeof(dataB)),
        bufC = compute.createBuffer(CL_MEM_READ_WRITE, sizeof(dataC));

    compute.writeBuffer(bufA, dataA, sizeof(dataA));
    compute.writeBuffer(bufB, dataB, sizeof(dataB));
    
    vector<Compute::Buffer *> args;
    args.push_back(&bufA);
    args.push_back(&bufB);
    args.push_back(&bufC);
    kernel.setArgs(args);
    compute.enqueueKernel(kernel, size, size);

    compute.readBuffer(bufC, dataC, sizeof(dataC));

    poclu_bswap_cl_float_array(compute.device, (cl_float *) dataC, 4 * size);

    for(size_t i = 0; i < size; i++) {
        cout << dataC[i] << endl;
    }

    cout << "OK\n";
}
