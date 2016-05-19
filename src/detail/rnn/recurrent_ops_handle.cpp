
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops_handle.h>

#include <prnn/detail/matrix/operation.h>

namespace prnn
{

RecurrentActivationFunction::RecurrentActivationFunction()
{

}

RecurrentActivationFunction::~RecurrentActivationFunction()
{

}

RecurrentActivationFunction::RecurrentActivationFunction(
    const RecurrentActivationFunction& function)
: forwardOperation(new matrix::Operation(*function.forwardOperation)),
  reverseOperation(new matrix::Operation(*function.reverseOperation))
{

}

RecurrentActivationFunction& RecurrentActivationFunction::operator=(
    const RecurrentActivationFunction& function)
{
    *forwardOperation = *function.forwardOperation;
    *reverseOperation = *function.reverseOperation;

    return *this;
}

RecurrentRectifiedLinear::RecurrentRectifiedLinear()
{
    forwardOperation.reset(new matrix::RectifiedLinear);
    reverseOperation.reset(new matrix::RectifiedLinearDerivative);
}

RecurrentHyperbolicTangent::RecurrentHyperbolicTangent()
{
    forwardOperation.reset(new matrix::Tanh);
    reverseOperation.reset(new matrix::TanhDerivative);
}

}



