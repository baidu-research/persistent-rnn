#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;           } }
namespace prnn { namespace matrix { class Precision;        } }
namespace prnn { namespace matrix { class Operation;        } }
namespace prnn { namespace matrix { class Dimension;        } }
namespace prnn { namespace matrix { class ConstDynamicView; } }
namespace prnn { namespace matrix { class DynamicView;      } }

namespace prnn
{
namespace matrix
{

void copy(const DynamicView& result, const ConstDynamicView& input);
void copy(const Matrix& result, const Matrix& input);
void copy(const Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

}
}


