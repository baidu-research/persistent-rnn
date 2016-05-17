
#pragma once

#include <prnn/detail/parallel/concurrent_collectives.h>
#include <prnn/detail/parallel/synchronization.h>

namespace prnn
{
namespace parallel
{
namespace detail
{

#ifdef __NVCC__

inline void checkCudaErrors(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template<typename FunctionType>
__global__ void kernelLauncher(FunctionType function)
{
    function(ThreadGroup(blockDim.x * gridDim.x, threadIdx.x + blockIdx.x * blockDim.x));
}

template<typename FunctionType>
void launchCudaKernel(FunctionType function)
{
    int ctasPerSM = 4;
    int threads   = 512;

    int multiprocessorCount = 0;

    checkCudaErrors(cudaDeviceGetAttribute(&multiprocessorCount,
        cudaDevAttrMultiProcessorCount, 0));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctasPerSM, kernelLauncher<FunctionType>, threads, 0));

    size_t ctas = multiprocessorCount * ctasPerSM;

    kernelLauncher<<<ctas, threads>>>(function);
}
#endif

}

template<typename FunctionType>
void multiBulkSynchronousParallel(FunctionType function)
{
    if(isCudaEnabled())
    {
        #ifdef __NVCC__
        setNotSynchronized();

        detail::launchCudaKernel(function);
        #else
        function(ThreadGroup(1, 0));
        #endif
    }
    else
    {
        function(ThreadGroup(1, 0));
    }
}

}
}


