#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;    } }
namespace prnn { namespace matrix { class Operation; } }
namespace prnn { namespace matrix { class Precision; } }
namespace prnn { namespace matrix { class Dimension; } }

namespace prnn
{
namespace matrix
{

void srand(size_t seed);

void rand (Matrix& result);
void randn(Matrix& result);

Matrix rand (const Dimension&, const Precision& );
Matrix randn(const Dimension&, const Precision& );

}
}

