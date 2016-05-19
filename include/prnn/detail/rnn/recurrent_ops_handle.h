#pragma once

// Standard Library Includes
#include <map>
#include <memory>
#include <string>

// Forward Declarations
namespace prnn { namespace matrix { class Operation; } }

namespace prnn
{

enum RecurrentLayerDirection
{
    RECURRENT_FORWARD,
    RECURRENT_REVERSE
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
        const RecurrentActivationFunction& activationFunction,
        RecurrentLayerDirection direction,
        bool allowPersistentKernels = true,
        double skipConnectionScale = 0.0) :

        layerSize(layerSize),
        miniBatchSize(miniBatchSize),
        timesteps(timesteps),
        allowPersistentKernels(allowPersistentKernels),
        skipConnectionScale(skipConnectionScale),
        activationFunction(activationFunction),
        direction(direction),
        stream(0)
    {}

public:
    size_t layerSize;
    size_t miniBatchSize;
    size_t timesteps;
    bool   allowPersistentKernels;
    double skipConnectionScale;

public:
    RecurrentActivationFunction activationFunction;
    RecurrentLayerDirection     direction;

public:
    void* stream;
};

}


