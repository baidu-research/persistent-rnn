
#pragma once

// Lucius Includes
#include <prnn/matrix/interface/DimensionTransformations.h>

#include <prnn/parallel/interface/ScalarOperations.h>

namespace prnn
{
namespace matrix
{

CUDA_DECORATOR Dimension linearStride(const Dimension& size)
{
    Dimension stride;

    size_t step = 1;

    for (auto sizeStep : size)
    {
        stride.push_back(step);
        step *= sizeStep;
    }

    return stride;
}

CUDA_DECORATOR Dimension zeros(const Dimension& size)
{
    Dimension result;

    for(size_t i = 0, arity = size.size(); i < arity; ++i)
    {
        result.push_back(0);
    }

    return result;
}

CUDA_DECORATOR Dimension ones(const Dimension& size)
{
    Dimension result;

    for(size_t i = 0, arity = size.size(); i < arity; ++i)
    {
        result.push_back(1);
    }

    return result;
}

CUDA_DECORATOR static bool isContained(const Dimension& set, size_t element)
{
    for (auto i : set)
    {
        if (i == element)
        {
            return true;
        }
    }

    return false;
}

CUDA_DECORATOR Dimension removeDimensions(const Dimension& base, const Dimension& removed)
{
    if(removed.size() == 0)
    {
        return Dimension({1});
    }

    Dimension result;

    for(size_t i = 0; i < base.size(); ++i)
    {
        if(!isContained(removed, i))
        {
            result.push_back(base[i]);
        }
    }

    return result;
}

CUDA_DECORATOR Dimension selectDimensions(const Dimension& base, const Dimension& selected)
{
    Dimension result;

    for(auto i : selected)
    {
        result.push_back(base[i]);
    }

    return result;
}

CUDA_DECORATOR Dimension selectReverseMappingDimensions(const Dimension& base,
    const Dimension& selected)
{
    Dimension result;

    result.resize(base.size());

    size_t index = 0;
    for(auto i : selected)
    {
        result[i] = base[index++];
    }

    return result;
}


CUDA_DECORATOR Dimension intersection(const Dimension& left, const Dimension& right)
{
    size_t totalDimensions = parallel::min(left.size(), right.size());

    Dimension result;

    for(size_t i = 0; i < totalDimensions; ++i)
    {
        result.push_back(parallel::min(left[i], right[i]));
    }

    return result;
}

CUDA_DECORATOR size_t dotProduct(const Dimension& left, const Dimension& right)
{
    assert(left.size() == right.size());

    size_t product = 0;

    for(auto i = left.begin(), j = right.begin(); i != left.end(); ++i, ++j)
    {
        product += *i * *j;
    }

    return product;
}

CUDA_DECORATOR Dimension linearToDimension(size_t linearIndex, const Dimension& size)
{
    Dimension result;

    for(auto dimensionSize : size)
    {
        result.push_back(linearIndex % dimensionSize);

        linearIndex /= dimensionSize;
    }

    return result;
}

CUDA_DECORATOR Dimension selectNamedDimensions(const Dimension& selectedDimensions,
    const Dimension& left, const Dimension& right)
{
    Dimension result;

    if(selectedDimensions.size() == 0)
    {
        return right;
    }

    size_t selectedDimensionIndex = 0;
    size_t leftIndex = 0;

    for(size_t rightIndex = 0; rightIndex != right.size(); ++rightIndex)
    {
        if(selectedDimensionIndex < selectedDimensions.size() &&
            selectedDimensions[selectedDimensionIndex] == rightIndex)
        {
            result.push_back(right[rightIndex]);
            ++selectedDimensionIndex;
        }
        else
        {
            result.push_back(left[leftIndex]);
            ++leftIndex;
        }
    }

    return result;
}

CUDA_DECORATOR bool isContiguous(const Dimension& dimensions)
{
    if(dimensions.size() == 0)
    {
        return true;
    }

    size_t next = dimensions[0] + 1;

    for(size_t i = 1; i < dimensions.size(); ++i, ++next)
    {
        if(dimensions[i] != next)
        {
            return false;
        }
    }

    return true;
}

CUDA_DECORATOR static size_t getOffset(const Dimension& stride, const Dimension& position)
{
    size_t offset = 0;
    size_t arity = parallel::min(stride.size(), position.size());

    for(size_t i = 0; i < arity; ++i)
    {
        offset += stride[i] * position[i];
    }

    return offset;
}

CUDA_DECORATOR void* getAddress(const Dimension& stride, const Dimension& position,
    void* data, size_t elementSize)
{
    size_t offset = getOffset(stride, position);

    uint8_t* address = static_cast<uint8_t*>(data);

    return address + elementSize * offset;
}

CUDA_DECORATOR const void* getAddress(const Dimension& stride, const Dimension& position,
    const void* data, size_t elementSize)
{
    size_t offset = getOffset(stride, position);

    const uint8_t* address = static_cast<const uint8_t*>(data);

    return address + elementSize * offset;
}

}
}

