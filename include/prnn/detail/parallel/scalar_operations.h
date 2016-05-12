

#pragma once

#include <lucius/parallel/interface/cuda.h>

#include <cstddef>
#include <algorithm>

namespace lucius
{
namespace parallel
{

template <typename T>
CUDA_DECORATOR inline T min(const T& left, const T& right)
{
    #ifdef __NVCC__
    return ::min(left, right);
    #else
    return std::min(left, right);
    #endif
}

CUDA_DECORATOR inline size_t min(const size_t& left, const size_t& right)
{
    #ifdef __NVCC__
    return ::min((unsigned long long)left, (unsigned long long)right);
    #else
    return std::min(left, right);
    #endif
}

template <typename T>
CUDA_DECORATOR inline T max(const T& left, const T& right)
{
    #ifdef __NVCC__
    return ::max(left, right);
    #else
    return std::max(left, right);
    #endif
}

CUDA_DECORATOR inline size_t max(const size_t& left, const size_t& right)
{
    #ifdef __NVCC__
    return ::max((unsigned long long)left, (unsigned long long)right);
    #else
    return std::max(left, right);
    #endif
}

}
}


