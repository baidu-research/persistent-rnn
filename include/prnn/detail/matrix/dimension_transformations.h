
#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/cuda.h>
#include <prnn/detail/matrix/dimension.h>

namespace prnn
{
namespace matrix
{

CUDA_DECORATOR inline Dimension linearStride(const Dimension& );
CUDA_DECORATOR inline Dimension zeros(const Dimension& );
CUDA_DECORATOR inline Dimension ones(const Dimension& );
CUDA_DECORATOR inline Dimension removeDimensions(const Dimension& base, const Dimension& toRemove);
CUDA_DECORATOR inline Dimension selectDimensions(const Dimension& base, const Dimension& selected);
CUDA_DECORATOR inline Dimension selectReverseMappingDimensions(const Dimension& base,
    const Dimension& selected);
CUDA_DECORATOR inline Dimension intersection(const Dimension& base, const Dimension& toRemove);
CUDA_DECORATOR inline size_t dotProduct(const Dimension& left, const Dimension& right);
CUDA_DECORATOR inline Dimension linearToDimension(size_t linearIndex, const Dimension& size);
CUDA_DECORATOR inline Dimension selectNamedDimensions(const Dimension& selectedDimensions,
    const Dimension& left, const Dimension& right);
CUDA_DECORATOR inline bool isContiguous(const Dimension& dimensions);

CUDA_DECORATOR inline void* getAddress(const Dimension& stride, const Dimension& position,
    void* data, size_t elementSize);
CUDA_DECORATOR inline const void* getAddress(const Dimension& stride, const Dimension& position,
    const void* data, size_t elementSize);

CUDA_DECORATOR inline Dimension fillInDimension(const Dimension& newSize,
    const Dimension& inputSize);
CUDA_DECORATOR inline Dimension computeSpacing(const Dimension& stride, const Dimension& size);
CUDA_DECORATOR inline Dimension fillInStride(const Dimension& newSize,
    const Dimension& inputStride, const Dimension& inputSize);

}
}

#include "dimension_transformations.inl"


