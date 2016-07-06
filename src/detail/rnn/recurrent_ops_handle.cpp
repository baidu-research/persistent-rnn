
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops_handle.h>

#include <prnn/detail/matrix/operation.h>

// Standard Library Includes
#include <sstream>

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

static std::string toString(const RecurrentActivationFunction& function)
{
    if(*function.forwardOperation == matrix::Tanh())
    {
        return "Tanh";
    }
    else if(*function.forwardOperation == matrix::RectifiedLinear())
    {
        return "ReLU";
    }

    return "Unknown Activation Function";
}

static std::string toString(RecurrentLayerDirection direction)
{
    if(direction == RECURRENT_FORWARD)
    {
        return "forward";
    }
    else if(direction == RECURRENT_REVERSE)
    {
        return "reverse";
    }
    else
    {
        return "bidirectional";
    }
}

static std::string toString(RecurrentLayerType layerType)
{
    if(layerType == RECURRENT_SIMPLE_TYPE)
    {
        return "simple rnn";
    }
    else if(layerType == RECURRENT_GRU_TYPE)
    {
        return "gru";
    }
    else
    {
        return "lstm";
    }
}

static std::string toString(RecurrentLayerInputMode inputMode)
{
    if(inputMode == RECURRENT_LINEAR_INPUT)
    {
        return "linear input";
    }
    else
    {
        return "skip input";
    }
}

std::string RecurrentOpsHandle::toString() const
{
    std::stringstream stream;

    stream << "RNN (" << layerSize << ", " << miniBatchSize << ", " << timesteps << ", "
        << layers << ") (" << prnn::toString(activationFunction) << ", "
        << prnn::toString(direction) << ", " << prnn::toString(layerType) << ", "
        << prnn::toString(inputMode) << ")";

    return stream.str();
}

}



