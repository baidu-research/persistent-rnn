#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;    } }
namespace prnn { namespace matrix { class Precision; } }
namespace prnn { namespace matrix { class Operation; } }
namespace prnn { namespace matrix { class Dimension; } }

namespace prnn
{
namespace matrix
{

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

}
}


