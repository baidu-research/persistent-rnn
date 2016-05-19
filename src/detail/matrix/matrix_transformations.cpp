

// Persistent RNN Includes
#include <prnn/detail/matrix/matrix_transforms.h>

#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/dimension_transformations.h>
#include <prnn/detail/matrix/copy_operations.h>

// Standard Library Includes
#include <set>

namespace prnn
{
namespace matrix
{

Matrix reshape(const Matrix& input, const Dimension& size)
{
    Matrix tempInput(input);

    auto newSize = fillInDimension(size, input.size());

    return Matrix(newSize, fillInStride(newSize, input.stride(), input.size()),
        input.precision(), tempInput.allocation(), tempInput.data());
}

Matrix flatten(const Matrix& matrix)
{
    return reshape(matrix, {matrix.elements()});
}

Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end)
{
    auto size = end - begin;

    Matrix tempInput(input);

    return Matrix(size, input.stride(), input.precision(),
        tempInput.allocation(), tempInput[begin].address());
}

Matrix slice(const Matrix& input, const Dimension& begin,
    const Dimension& end, const Dimension& stride)
{
    auto size = (end - begin) / stride;

    Matrix tempInput(input);

    return Matrix(size, input.stride() * stride, input.precision(),
        tempInput.allocation(), tempInput[begin].address());
}

Matrix resize(const Matrix& input, const Dimension& size)
{
    Matrix result(size, input.precision());

    auto overlap = intersection(size, input.size());

    auto resultSlice = slice(result, zeros(overlap), overlap);
    auto inputSlice  = slice(input,  zeros(overlap), overlap);

    copy(resultSlice, inputSlice);

    return result;
}

Matrix concatenate(const Matrix& left, const Matrix& right, size_t dimension)
{
    auto size = left.size();

    size[dimension] += right.size()[dimension];

    Matrix result(size, left.precision());

    auto leftStart  = zeros(size);
    auto rightStart = zeros(size);

    rightStart[dimension] = left.size()[dimension];

    auto leftSlice  = slice(result, leftStart,  left.size());
    auto rightSlice = slice(result, rightStart, result.size());

    copy(leftSlice,  left);
    copy(rightSlice, right);

    return result;
}

}
}



