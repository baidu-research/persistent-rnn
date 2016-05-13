
// Persistent RNN Includes
#include <prnn/detail/parallel/synchronization.h>
#include <prnn/detail/parallel/cuda_runtime_library.h>

// Standard Library Includes
#include <atomic>

namespace prnn
{

namespace parallel
{

static std::atomic<bool> isSynchronized(true);

void setNotSynchronized()
{
    isSynchronized.store(false, std::memory_order_release);
}

void synchronize()
{
    if(!CudaRuntimeLibrary::loaded())
    {
        return;
    }

    if(isSynchronized.load(std::memory_order_acquire))
    {
        return;
    }

    CudaRuntimeLibrary::cudaDeviceSynchronize();

    isSynchronized.store(true, std::memory_order_release);
}

}

}





