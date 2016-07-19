
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/matrix/operation.h>
#include <prnn/detail/matrix/precision.h>
#include <prnn/detail/matrix/cudnn_library.h>

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
{
    if(function.forwardOperation)
    {
        forwardOperation.reset(new matrix::Operation(*function.forwardOperation));
    }
    if(function.reverseOperation)
    {
        reverseOperation.reset(new matrix::Operation(*function.reverseOperation));
    }
}

RecurrentActivationFunction& RecurrentActivationFunction::operator=(
    const RecurrentActivationFunction& function)
{
    if(function.forwardOperation)
    {
        *forwardOperation = *function.forwardOperation;
    }
    if(function.reverseOperation)
    {
        *reverseOperation = *function.reverseOperation;
    }

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
    if(!function.forwardOperation)
    {
        return "";
    }

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

static std::string toString(RecurrentLayerBackend backend)
{
    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        return "cudnn";
    }
    else if(backend == RECURRENT_PERSISTENT_BACKEND)
    {
        return "persistent";
    }
    else if(backend == RECURRENT_GENERIC_BACKEND)
    {
        return "generic";
    }
    else
    {
        return "best";
    }
}

static bool isPersistentBackendSupported(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return parallel::isCudaEnabled() &&
            handle.direction == RECURRENT_FORWARD && // TODO
            handle.layers == 1 && // TODO
            handle.miniBatchSize >= 3 &&
            handle.layerType == RECURRENT_SIMPLE_TYPE &&
            handle.inputMode == RECURRENT_SKIP_INPUT && // TODO
            handle.layerSize % 4 == 0 &&
            handle.layerSize <= rnn::getMaximumSizeRNNForThisGPU(precision) &&
            (precision == matrix::SinglePrecision() || precision == matrix::HalfPrecision());
}

bool isCudnnBackendSupported(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return matrix::CudnnLibrary::isSupported() &&
        handle.direction != RECURRENT_REVERSE &&
        handle.direction != RECURRENT_BIDIRECTIONAL && // test this
        handle.skipConnectionScale == 0.0;
}

static bool isGenericBackendSupported(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return handle.direction != RECURRENT_BIDIRECTIONAL &&
        handle.layers == 1 && // TODO
        handle.layerType == RECURRENT_SIMPLE_TYPE && // TODO
        handle.inputMode == RECURRENT_SKIP_INPUT && // TODO
        handle.skipConnectionScale == 0.0; // TODO
}

RecurrentLayerBackend getBackend(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto result = handle.backend;

    if(handle.backend == RECURRENT_BEST_BACKEND)
    {
        if(isPersistentBackendSupported(handle, precision))
        {
            result = RECURRENT_PERSISTENT_BACKEND;
        }
        else if(isCudnnBackendSupported(handle, precision))
        {
            result = RECURRENT_CUDNN_BACKEND;
        }
        else
        {
            result = RECURRENT_GENERIC_BACKEND;
        }
    }

    return result;
}

RecurrentLayerBackend getBackendThrowOnError(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto result = getBackend(handle, precision);

    if(result == RECURRENT_CUDNN_BACKEND && !isCudnnBackendSupported(handle, precision))
    {
        throw NotSupported();
    }
    else if(result == RECURRENT_PERSISTENT_BACKEND &&
        !isPersistentBackendSupported(handle, precision))
    {
        throw NotSupported();
    }
    else if(result == RECURRENT_GENERIC_BACKEND &&
        !isGenericBackendSupported(handle, precision))
    {
        throw NotSupported();
    }

    return result;
}

std::string RecurrentOpsHandle::toString() const
{
    std::stringstream stream;

    stream << "RNN (" << layerSize << ", " << miniBatchSize << ", " << timesteps << ", "
        << layers << ") (" << prnn::toString(activationFunction) << ", "
        << prnn::toString(direction) << ", " << prnn::toString(layerType) << ", "
        << prnn::toString(inputMode) << ", " << prnn::toString(backend) << ")";

    return stream.str();
}

}



