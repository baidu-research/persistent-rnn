
#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/cuda.h>

// Standard Library Includes
#include <cstdint>

namespace prnn
{
namespace parallel
{

class ThreadGroup
{
public:
    CUDA_DECORATOR inline ThreadGroup(uint32_t size, uint32_t id);

public:
    CUDA_DECORATOR inline uint32_t size() const;
    CUDA_DECORATOR inline uint32_t id()   const;

public:
    uint32_t _size;
    uint32_t _id;

};

template<int level>
class GroupLevelSize
{
public:
    CUDA_DECORATOR static constexpr uint32_t size()
    {
        #ifdef __CUDA_ARCH__
        return level == 0 ? 1  :
               ((level == 1) ? 32 : 512);
        #else
        return 1;
        #endif
    }
};

CUDA_DECORATOR inline ThreadGroup partitionThreadGroup(ThreadGroup g, uint32_t subgroupSize);
CUDA_DECORATOR inline ThreadGroup partitionThreadGroupAtLevel(ThreadGroup g, uint32_t level);

CUDA_DECORATOR inline ThreadGroup getRelativeGroup(ThreadGroup inner, ThreadGroup outer);

CUDA_DECORATOR inline void barrier(ThreadGroup g);

template<typename T>
CUDA_DECORATOR inline T gather(ThreadGroup g, T value, uint32_t index);

template<typename T, typename Function>
CUDA_DECORATOR inline T reduce(ThreadGroup g, T value, Function f);

}
}

#include <prnn/detail/parallel/concurrent_collectives.inl>



