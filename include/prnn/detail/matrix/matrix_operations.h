
#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;    } }
namespace prnn { namespace matrix { class Operation; } }
namespace prnn { namespace matrix { class Precision; } }
namespace prnn { namespace matrix { class Dimension; } }

namespace prnn
{
namespace detail
{
namespace matrix
{

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op);
Matrix apply(const Matrix& left, const Matrix& right, const Operation& op);

void apply(Matrix& result, const Matrix& input, const Operation& op);
Matrix apply(const Matrix& input, const Operation& op);

void reduce(Matrix& result, const Matrix& input, const Dimension& d, const Operation& op);
Matrix reduce(const Matrix& input, const Dimension& d, const Operation& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right,
    const Dimension& d, const Operation& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op);

void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);

void ones(Matrix& result);
Matrix ones(const Dimension& size, const Precision& precision);

void reduceGetPositions(Matrix& result, const Matrix& input, const Dimension& d,
    const Operation& op);
Matrix reduceGetPositions(const Matrix& input, const Dimension& d, const Operation& op);

}
}
}


