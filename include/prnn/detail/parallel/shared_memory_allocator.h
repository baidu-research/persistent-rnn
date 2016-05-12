
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

// Standard Library Includes
#include <cstddef>

namespace lucius
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

#include <lucius/parallel/implementation/SharedMemoryAllocator.inl>




