
#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;    } }
namespace prnn { namespace matrix { class Dimension; } }

namespace prnn
{
namespace matrix
{

Matrix reshape(const Matrix& matrix, const Dimension& );
Matrix flatten(const Matrix& matrix);

Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end);
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end,
    const Dimension& stride);
Matrix resize(const Matrix& input, const Dimension& size);
Matrix reshape(const Matrix& input, const Dimension& size);

Matrix concatenate(const Matrix& left, const Matrix& right, size_t dimension);

}
}



