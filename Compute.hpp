#ifndef __COMPUTE_HPP__
#define __COMPUTE_HPP__

#include <pocl/poclu.h>
#include <CL/cl.h>

#include <string>
#include <vector>

using namespace std;

class Compute
{
    public:
        Compute();
        virtual ~Compute();

        class Buffer
        {
            friend class Compute;
            friend class Kernel;
            
            public:
                virtual ~Buffer();
                size_t size();

            protected:
                Buffer(cl_mem iBufferId, size_t iSize);

                cl_mem buffer;
                size_t byteSize;
        };

        Buffer createBuffer(int iType, size_t iSize, void *iData = nullptr);

        void writeBuffer(Buffer &iBuffer, void *iData, size_t iSize);
        void readBuffer(Buffer &iBuffer, void *oData, size_t iSize);

        class Program
        {
            friend class Compute;

            public:
                virtual ~Program();

            protected:
                Program(cl_program iProgramId);

                cl_program program;
        };

        Program createProgram(const string &iSourceName);

        class Kernel
        {
            friend class Compute;

            public:
                virtual ~Kernel();

                void getGlobalWorkSize(size_t *oSizes);
                size_t getWorkGroupSize();
                void setArgs(const vector<Buffer *> &iBuffers);

            protected:
                Kernel(cl_kernel iKernelId);

                cl_kernel kernel;
        };

        Kernel createKernel(Program &iProgram, const string &iFuncName);

        cl_device_type getDeviceType();
        cl_uint getDeviceMaxComputeUnits();
        cl_uint getDeviceMaxWorkItemDimensions();
        void getDeviceMaxWorkItemSizes(size_t *oSizes);
        size_t getDeviceMaxWorkGroupSize();

        void enqueueKernel(Kernel &iKernel, size_t iGlobalWorkSize, size_t iLocalWorkSize);

    // protected:
        cl_context context;
        cl_device_id device;
        cl_command_queue queue;
        cl_platform_id platform;
};

#endif
