#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/shared_memory_allocator.h>

namespace prnn
{
namespace parallel
{

template <typename T, size_t capacity>
CUDA_DECORATOR T* SharedMemoryAllocator<T, capacity>::allocate()
{
    #ifdef __CUDA_ARCH__
    __shared__ T memory[capacity];
    #else
    static T memory[capacity];
    #endif
    return memory;
}

}
}

