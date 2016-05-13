

// Persistent RNN Includes
#include <prnn/detail/parallel/memory.h>
#include <prnn/detail/parallel/cuda_runtime_library.h>

// Standard Library Includes
#include <cstdlib>

namespace prnn
{
namespace parallel
{

void* malloc(size_t size)
{
    if(CudaRuntimeLibrary::loaded())
    {
        #ifdef __APPLE__
        return CudaRuntimeLibrary::cudaHostAlloc(size);
        #else
        return CudaRuntimeLibrary::cudaMallocManaged(size);
        #endif
    }
    else
    {
        return std::malloc(size);
    }
}

void free(void* address)
{
    if(CudaRuntimeLibrary::loaded())
    {
        #ifdef __APPLE__
        return CudaRuntimeLibrary::cudaFreeHost(address);
        #else
        return CudaRuntimeLibrary::cudaFree(address);
        #endif
    }
    else
    {
        std::free(address);
    }
}

}
}




