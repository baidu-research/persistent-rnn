#pragma once

// Standard Library Includes
#include <map>
#include <memory>
#include <string>

// Forward Declarations
namespace prnn { namespace matrix { class Operation; } }
namespace prnn { namespace matrix { class Precision; } }

namespace prnn
{

enum RecurrentLayerDirection
{
    RECURRENT_FORWARD,
    RECURRENT_REVERSE,
    RECURRENT_BIDIRECTIONAL
};

enum RecurrentLayerType
{
    RECURRENT_SIMPLE_TYPE,
    RECURRENT_GRU_TYPE,
    RECURRENT_LSTM_TYPE
};

enum RecurrentLayerInputMode
{
    RECURRENT_LINEAR_INPUT,
    RECURRENT_SKIP_INPUT
};

enum RecurrentLayerBackend
{
    RECURRENT_CUDNN_BACKEND,
    RECURRENT_PERSISTENT_BACKEND,
    RECURRENT_GENERIC_BACKEND,
    RECURRENT_BEST_BACKEND
};

class NotSupported
{

};

class RecurrentActivationFunction
{
public:
    RecurrentActivationFunction();
    ~RecurrentActivationFunction();

public:
    RecurrentActivationFunction(const RecurrentActivationFunction&);
    RecurrentActivationFunction& operator=(const RecurrentActivationFunction&);

public:
    std::unique_ptr<matrix::Operation> forwardOperation;
    std::unique_ptr<matrix::Operation> reverseOperation;

};

class RecurrentRectifiedLinear : public RecurrentActivationFunction
{
public:
    RecurrentRectifiedLinear();

};

class RecurrentHyperbolicTangent : public RecurrentActivationFunction
{
public:
    RecurrentHyperbolicTangent();
};

/*! \brief A single handle to store configuration data for a recurrent operation. */
class RecurrentOpsHandle
{
public:
    RecurrentOpsHandle(size_t layerSize, size_t miniBatchSize, size_t timesteps,
        size_t layers = 1,
        const RecurrentActivationFunction& activationFunction = RecurrentRectifiedLinear(),
        RecurrentLayerDirection direction = RECURRENT_FORWARD,
        RecurrentLayerType layerType = RECURRENT_SIMPLE_TYPE,
        RecurrentLayerInputMode inputMode = RECURRENT_SKIP_INPUT,
        RecurrentLayerBackend backend = RECURRENT_BEST_BACKEND,
        double skipConnectionScale = 0.0,
        void* stream = nullptr) :

        layerSize(layerSize),
        miniBatchSize(miniBatchSize),
        timesteps(timesteps),
        layers(layers),
        skipConnectionScale(skipConnectionScale),
        activationFunction(activationFunction),
        direction(direction),
        layerType(layerType),
        inputMode(inputMode),
        backend(backend),
        stream(stream)
    {}

public:
    std::string toString() const;

public:
    size_t layerSize;
    size_t miniBatchSize;
    size_t timesteps;

public:
    size_t layers;

public:
    double skipConnectionScale;

public:
    RecurrentActivationFunction activationFunction;
    RecurrentLayerDirection     direction;
    RecurrentLayerType          layerType;
    RecurrentLayerInputMode     inputMode;
    RecurrentLayerBackend       backend;

public:
    void* stream;
};

RecurrentLayerBackend getBackend(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
RecurrentLayerBackend getBackendThrowOnError(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
bool isCudnnBackendSupported(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);

}


