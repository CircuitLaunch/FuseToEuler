#include "Compute.hpp"
#include <iostream>

using namespace std;

Compute::Compute()
{
    cl_uint numPlats;
    clGetPlatformIDs(0, nullptr, &numPlats);
    cl_platform_id *platIds = new cl_platform_id[numPlats];
    clGetPlatformIDs(numPlats, platIds, nullptr);

    for(cl_uint i = 0; i < numPlats; i++) {
        cl_uint numDevs;
        size_t nameSize;
        clGetPlatformInfo(platIds[i], CL_PLATFORM_NAME, 0, nullptr, &nameSize);
        char *name = new char[nameSize+1];
        clGetPlatformInfo(platIds[i], CL_PLATFORM_NAME, nameSize, name, nullptr);
        name[nameSize] = 0;
        cout << "Platform \"" << name << "\":\n";
        delete [] name;
        clGetDeviceIDs(platIds[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevs);
        cl_device_id *devIds = new cl_device_id[numDevs];
        clGetDeviceIDs(platIds[i], CL_DEVICE_TYPE_ALL, numDevs, devIds, nullptr);
        cout << "\tDevices:\n";
        for(cl_uint j = 0; j < numDevs; j++) {
            cl_device_type devType;
            clGetDeviceInfo(devIds[j], CL_DEVICE_TYPE, sizeof(devType), &devType, nullptr);
            switch(devType) {
                case CL_DEVICE_TYPE_ACCELERATOR:
                    cout << "\t\tACCELERATOR\n"; break;
                case CL_DEVICE_TYPE_CPU:
                    cout << "\t\tCPU\n"; break;
                case CL_DEVICE_TYPE_GPU:
                    cout << "\t\tGPU\n";
                    platform = platIds[i];
                    device = devIds[j];
                    break;
                case CL_DEVICE_TYPE_CUSTOM:
                    cout << "\t\tCUSTOM\n"; break;
                case CL_DEVICE_TYPE_DEFAULT:
                    cout << "\t\tDEFAULT\n"; break;
            }
        }
        delete [] devIds;
    }

    delete [] platIds;

    cl_context_properties contextProps[] = { CL_CONTEXT_PLATFORM, cl_context_properties(platform), 0 };

    cl_int err;
    context = clCreateContext(contextProps, 1, &device, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        cerr << "Failed to create OpenCL context: " << err << " ";
        switch(err) {
            case CL_INVALID_PLATFORM:
                cerr << "INVALID PLATFORM";
                break;
            case CL_INVALID_PROPERTY:
                cerr << "INVALID PROPERTY";
                break;
            case CL_INVALID_VALUE:
                cerr << "INVALID VALUE";
                break;
            case CL_INVALID_DEVICE:
                cerr << "INVALID DEVICE";
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                cerr << "DEVICE NOT AVAILABLE";
                break;
            case CL_OUT_OF_RESOURCES:
                cerr << "OUT OF RESOURCES";
                break;
            case CL_OUT_OF_HOST_MEMORY:
                cerr << "OUT OF HOST MEMORY";
                break;
            default:
                cerr << "UNKNOWN";
        }
        cerr << endl;
    }
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " clCreateCommandQueueWithProperties\n";
    }
}

Compute::~Compute()
{
    clReleaseCommandQueue(queue);
    clUnloadPlatformCompiler(platform);
    clReleaseContext(context);
}

    Compute::Buffer::Buffer(cl_mem iBufferId, size_t iSize)
    : buffer(iBufferId), byteSize(iSize)
    { }

    Compute::Buffer::~Buffer()
    {
        clReleaseMemObject(buffer);
    }

    size_t Compute::Buffer::size()
    {
        return byteSize;
    }

Compute::Buffer Compute::createBuffer(int iType, size_t iSize, void *iData)
{
    cl_int err;
    cl_mem id = clCreateBuffer(context, iType, iSize, iData, &err);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " clCreateBuffer\n";
    }
    return Buffer(id, iSize);
}

void Compute::writeBuffer(Buffer &iBuffer, void *iData, size_t iSize)
{
    cl_int err = clEnqueueWriteBuffer(queue, iBuffer.buffer, true, 0, iSize, iData, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " clEnqueueWriteBuffer\n";
    }
}

void Compute::readBuffer(Buffer &iBuffer, void *oData, size_t iSize)
{
    cl_int err = clEnqueueReadBuffer(queue, iBuffer.buffer, true, 0, iSize, oData, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " clEnqueueReadBuffer\n";
    }
}

    Compute::Program::Program(cl_program iProgramId)
    : program(iProgramId)
    { }

    Compute::Program::~Program()
    {
        clReleaseProgram(program);
    }

Compute::Program Compute::createProgram(const string &iSourceName)
{
    cl_program id;
    cl_int err = poclu_load_program(context, device, iSourceName.c_str(), 0, 0, 0, NULL, NULL, &id);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " poclu_load_program\n";
    }
    return Program(id);
}

    Compute::Kernel::Kernel(cl_kernel iKernelId)
    : kernel(iKernelId)
    { }

    Compute::Kernel::~Kernel()
    {
        clReleaseKernel(kernel);
    }

    void Compute::Kernel::setArgs(const vector<Compute::Buffer *> &iBuffers)
    {
        int i = 0;
        for(Buffer *buf : iBuffers) {
            cl_int err = clSetKernelArg(kernel, i, sizeof(cl_mem), (void *) &(buf->buffer));
            if(err != CL_SUCCESS) {
                cerr << "Error: " << err << " clSetKernelArg\n";
            }
            i++;
        }
    }

    void Compute::Kernel::getGlobalWorkSize(size_t *oSizes)
    {
        clGetKernelInfo(kernel, CL_KERNEL_GLOBAL_WORK_SIZE, sizeof(size_t) * 3, oSizes, NULL);
    }

    size_t Compute::Kernel::getWorkGroupSize()
    {
        size_t val;
        clGetKernelInfo(kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(val), &val, NULL);
        return val;
    }

Compute::Kernel Compute::createKernel(Program &iProgram, const string &iFuncName)
{
    cl_kernel id = clCreateKernel(iProgram.program, iFuncName.c_str(), NULL);
    return Kernel(id);
}

cl_device_type Compute::getDeviceType()
{
    cl_device_type val;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(val), &val, NULL);
    return val;
}

cl_uint Compute::getDeviceMaxComputeUnits()
{
    cl_uint val;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(val), &val, NULL);
    return val;
}

cl_uint Compute::getDeviceMaxWorkItemDimensions()
{
    cl_uint val;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(val), &val, NULL);
    return val;
}

void Compute::getDeviceMaxWorkItemSizes(size_t *oSizes)
{
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, oSizes, NULL);
}

size_t Compute::getDeviceMaxWorkGroupSize()
{
    size_t val;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(val), &val, NULL);
    return val;
}

void Compute::enqueueKernel(Kernel &iKernel, size_t iGlobalWorkSize, size_t iLocalWorkSize)
{
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue, iKernel.kernel, 1, NULL, &iGlobalWorkSize, &iLocalWorkSize, 0, NULL, &event);
    if(err != CL_SUCCESS) {
        cerr << "Error: " << err << " clEnqueueNDRangeKernel\n";
    }
    switch(clWaitForEvents(1, &event)) {
        case CL_INVALID_VALUE:
            cerr << "Invalid value\n";
            break;
        case CL_INVALID_CONTEXT:
            cerr << "Invalid context\n";
            break;
        case CL_INVALID_EVENT:
            cerr << "Invalid event\n";
            break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            cerr << "Exec status error for events in wait list\n";
            break;
        case CL_OUT_OF_RESOURCES:
            cerr << "Out of resources\n";
            break;
        case CL_OUT_OF_HOST_MEMORY:
            cerr << "Out of host memory\n";
            break;
    }
}
