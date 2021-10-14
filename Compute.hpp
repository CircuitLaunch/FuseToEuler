#ifndef __COMPUTE_HPP__
#define __COMPUTE_HPP__

#include <cstdarg>
#include <string>
#include <vector>

#include "Exception.hpp"

using namespace std;

#define USECUDA true

#include <cuda_runtime.h>

namespace Compute
{

    template <class T>
    class Buffer
    {
        friend class Compute;
        
        public:
            Buffer(T *iDeviceBuffer, size_t iCount);
            virtual ~Buffer();

            cudaError_t copyToDevice(const T *iHostBuffer, size_t iCount);
            cudaError_t copyToHost(T *oHostBuffer, size_t iCount);

            T *getDeviceBuffer() { return buffer; }
            const T *getDeviceBuffer() const { return buffer; }

        protected:
            T *buffer;
            size_t count;
    };


    template <class T>
    Buffer<T> createBuffer(size_t iCount, T *iData = nullptr);

    class Functor
    {
        friend class Compute;

        public:
            virtual cudaError_t operator()(size_t iBlocksPerGrid, size_t iThreadsPerBlock, ...) = 0;
    };

    template <class T>
    Buffer<T>::Buffer(T *iDeviceBuffer, size_t iCount)
    : buffer(iDeviceBuffer), count(iCount)
    { }

    template <class T>
    Buffer<T>::~Buffer()
    {
        cudaError_t err = cudaFree(buffer);
        if(err != cudaSuccess) {
            cerr << "Compute::Buffer::~Buffer: failed to free device buffer" << endl;
        }
    }

    template <class T>
    cudaError_t Buffer<T>::copyToDevice(const T *iHostBuffer, size_t iCount)
    {
        cudaMemcpy(buffer, (void *) iHostBuffer, iCount * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <class T>
    cudaError_t Buffer<T>::copyToHost(T *oHostBuffer, size_t iCount)
    {
        cudaMemcpy((void *) oHostBuffer, buffer, iCount * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <class T>
    Buffer<T> createBuffer(size_t iCount, T *iData)
    {
        T *buffer;
        cudaError_t err = cudaMalloc((void **) &buffer, iCount * sizeof(T));
        if(err != cudaSuccess) throw(Exception(__FILE__, __LINE__, "Compute::createBuffer", err, cudaGetErrorString(err)));
        if(iData) {
            err = cudaMemcpy(iData, (void *) buffer, iCount * sizeof(T), cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) throw(Exception(__FILE__, __LINE__, "Compute::createBuffer", err, cudaGetErrorString(err)));
        }
        return Buffer<T>(buffer, iCount);
    }

}

#endif
