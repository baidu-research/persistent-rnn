
#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/cuda.h>

// Standard Library Includes
#include <cstddef>

namespace prnn
{
namespace parallel
{

template <typename T, size_t capacity>
class SharedMemoryAllocator
{
public:
    static CUDA_DECORATOR T* allocate();

};

}
}

#include "shared_memory_allocator.inl"




