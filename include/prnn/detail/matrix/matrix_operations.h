
#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class ConstDynamicView; } }
namespace prnn { namespace matrix { class Matrix;           } }
namespace prnn { namespace matrix { class DynamicView;      } }
namespace prnn { namespace matrix { class Operation;        } }
namespace prnn { namespace matrix { class Precision;        } }
namespace prnn { namespace matrix { class Dimension;        } }

namespace prnn
{
namespace matrix
{

void apply(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right, const Operation& op);
void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix apply(const Matrix& left, const Matrix& right, const Operation& op);

void apply(const DynamicView& result, const ConstDynamicView& input, const Operation& op);
void apply(Matrix& result, const Matrix& input, const Operation& op);
Matrix apply(const Matrix& input, const Operation& op);

void reduce(Matrix& result, const Matrix& input, const Dimension& d, const Operation& op);
Matrix reduce(const Matrix& input, const Dimension& d, const Operation& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right,
    const Dimension& d, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op);

void zeros(const DynamicView& result);
void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);

void ones(Matrix& result);
Matrix ones(const Dimension& size, const Precision& precision);

}
}


