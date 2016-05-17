
#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/concurrent_collectives.h>

// Standard Library Includes
#include <cassert>

namespace prnn
{
namespace parallel
{

CUDA_DECORATOR inline ThreadGroup::ThreadGroup(uint32_t size, uint32_t id)
: _size(size), _id(id)
{

}

CUDA_DECORATOR inline uint32_t ThreadGroup::size() const
{
	return _size;
}

CUDA_DECORATOR inline uint32_t ThreadGroup::id() const
{
	return _id;
}

CUDA_DECORATOR inline ThreadGroup partitionThreadGroup(ThreadGroup g, uint32_t subgroupSize)
{
    return ThreadGroup(subgroupSize, g.id() % subgroupSize);
}

CUDA_DECORATOR inline ThreadGroup partitionThreadGroupAtLevel(ThreadGroup g, uint32_t level)
{
    if(level == 0)
    {
        return partitionThreadGroup(g, GroupLevelSize<0>::size());
    }
    else if(level == 1)
    {
        return partitionThreadGroup(g, GroupLevelSize<1>::size());
    }
    else if(level == 2)
    {
        return partitionThreadGroup(g, GroupLevelSize<2>::size());
    }

    return g;
}

CUDA_DECORATOR inline ThreadGroup getRelativeGroup(ThreadGroup inner, ThreadGroup outer)
{
    return ThreadGroup(outer.size() / inner.size(), outer.id() / inner.size());
}

CUDA_DECORATOR inline void barrier(ThreadGroup g)
{
    if(g.size() <= GroupLevelSize<1>::size())
    {
        return;
    }
    else if(g.size() <= GroupLevelSize<2>::size())
    {
        #ifdef __CUDA_ARCH__
        __syncthreads();
        #endif
        return;
    }

    //assert(false && "Not implemented.");

}

template<typename T>
CUDA_DECORATOR inline T gather(ThreadGroup g, T value, uint32_t index)
{
    if(g.size() == GroupLevelSize<0>::size())
    {
        return value;
    }
    else if(g.size() <= GroupLevelSize<1>::size())
    {
        #ifdef __CUDA_ARCH__
        return __shfl(value, index, g.size());
        #endif
    }
    else if(g.size() <= GroupLevelSize<2>::size())
    {
        T result = value;
        #ifdef __CUDA_ARCH__
        __shared__ T data[GroupLevelSize<2>::size()];
        data[g.id()] = value;
        barrier(g);

        if(index < GroupLevelSize<2>::size())
        {
            result = data[index];
        }

        barrier(g);
        #endif

        return result;
    }

    //assert(false && "Not implemented.");

    return value;
}

template<typename T, typename Function>
CUDA_DECORATOR inline T reduce(ThreadGroup g, T value, Function f)
{
    T result = value;

    for(uint32_t i = g.size() / 2; i >= 1; i /= 2)
    {
        result = f(result, gather(g, value, g.id() + i));
    }

    return result;
}

}
}

