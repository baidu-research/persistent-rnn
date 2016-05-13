

#include <prnn/detail/parallel/cuda.h>
#include <prnn/detail/parallel/cuda_runtime_library.h>

namespace prnn
{

namespace parallel
{

bool isCudaEnabled()
{
    return CudaRuntimeLibrary::loaded();
}

}
}

