#ifndef __VECADD_CUH__
#define __VECADD_CUH__

#include <vector_types.h>
#include "Compute.hpp"

class VecAdd : public Compute::Functor
{
    public:
        cudaError_t operator()(size_t iBlocksPerGrid, size_t iThreadsPerBlock, ...);
};

#endif