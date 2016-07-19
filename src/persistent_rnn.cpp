/*! \file  persistent_rnn.cpp
    \date  May 10, 2016
    \brief C language interface to persistent RNN kernels, modeled after the CUDNNv5 RNN interface
           for maximum compatibility.
*/

// Persistent RNN Includes
#include <persistent_rnn.h>

#include <prnn/detail/rnn/recurrent_ops.h>
#include <prnn/detail/rnn/recurrent_ops_handle.h>

#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/dimension_transformations.h>
#include <prnn/detail/matrix/cudnn_library.h>

// Standard Library Includes
#include <new>
#include <cstddef>

size_t prnnGetVersion(void)
{
    return PRNN_VERSION;
}

const char* prnnGetErrorString(prnnStatus_t status)
{
    switch(status)
    {
    case PRNN_STATUS_SUCCESS:
    {
        return "success";
    }
    case PRNN_STATUS_NOT_INITIALIZED:
    {
        return "not initialized";
    }
    case PRNN_STATUS_ALLOC_FAILED:
    {
        return "memory allocation failed";
    }
    case PRNN_STATUS_BAD_PARAM:
    {
        return "bad parameter";
    }
    case PRNN_STATUS_INTERNAL_ERROR:
    {
        return "internal error";
    }
    case PRNN_STATUS_INVALID_VALUE:
    {
        return "invalid value";
    }
    case PRNN_STATUS_ARCH_MISMATCH:
    {
        return "wrong architecture";
    }
    case PRNN_STATUS_MAPPING_ERROR:
    {
        return "mapping error";
    }
    case PRNN_STATUS_EXECUTION_FAILED:
    {
        return "kernel execution failed";
    }
    case PRNN_STATUS_NOT_SUPPORTED:
    {
        return "not supported";
    }
    default:
    {
        break;
    }
    }

    return "unknown error code";
}

struct prnnContext
{
public:
    prnnContext() : stream(nullptr) {}

public:
    void* stream;
};

prnnStatus_t prnnCreate(prnnHandle_t* handle)
{
    try
    {
        *handle = new prnnContext;
    }
    catch(std::bad_alloc)
    {
        return PRNN_STATUS_ALLOC_FAILED;
    }

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnDestroy(prnnHandle_t handle)
{
    delete handle;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnSetStream(prnnHandle_t handle, void* streamId)
{
    handle->stream = streamId;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetStream(prnnHandle_t handle, void** streamId)
{
    *streamId = handle->stream;

    return PRNN_STATUS_SUCCESS;
}

struct prnnTensorStruct
{
public:
    prnnTensorFormat_t format;

public:
    prnn::matrix::Dimension size;
    prnn::matrix::Dimension stride;
    prnn::matrix::Precision precision;
};

prnnStatus_t prnnCreateTensorDescriptor(prnnTensorDescriptor_t* handle)
{
    try
    {
        *handle = new prnnTensorStruct;
    }
    catch(std::bad_alloc)
    {
        return PRNN_STATUS_ALLOC_FAILED;
    }

    return PRNN_STATUS_SUCCESS;
}

static prnn::matrix::Precision getPrecision(prnnDataType_t dataType)
{
    if(dataType == PRNN_DATA_HALF)
    {
        return prnn::matrix::HalfPrecision();
    }
    else if(dataType == PRNN_DATA_FLOAT)
    {
        return prnn::matrix::SinglePrecision();
    }
    else
    {
        return prnn::matrix::DoublePrecision();
    }
}

static prnnDataType_t getDataType(const prnn::matrix::Precision& precision)
{
    if(precision == prnn::matrix::HalfPrecision())
    {
        return PRNN_DATA_HALF;
    }
    else if(precision == prnn::matrix::SinglePrecision())
    {
        return PRNN_DATA_FLOAT;
    }
    else
    {
        return PRNN_DATA_DOUBLE;
    }
}

prnnStatus_t prnnSetTensorNdDescriptor(prnnTensorDescriptor_t descriptor,
                                       prnnDataType_t         dataType,
                                       int                    nbDims,
                                       const int*             dimA,
                                       const int*             strideA)
{
    descriptor->precision = getPrecision(dataType);

    if (nbDims < 0 || nbDims > PRNN_DIM_MAX)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    descriptor->size.clear();
    descriptor->stride.clear();

    for (int i = 0; i < nbDims; ++i)
    {
        descriptor->size.push_back(dimA[i]);
        descriptor->stride.push_back(strideA[i]);
    }

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetTensorNdDescriptor(const prnnTensorDescriptor_t descriptor,
                                       int                          nbDimsRequested,
                                       prnnDataType_t*              dataType,
                                       int*                         nbDims,
                                       int*                         dimA,
                                       int*                         strideA)
{

    *dataType = getDataType(descriptor->precision);

    if (nbDimsRequested < 0 || nbDimsRequested > PRNN_DIM_MAX)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    int dims = std::min(descriptor->size.size(), static_cast<size_t>(nbDimsRequested));

    for (int i = 0; i < dims; ++i)
    {
        dimA[i]    = descriptor->size[i];
        strideA[i] = descriptor->stride[i];
    }

    *nbDims = dims;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnDestroyTensorDescriptor(prnnTensorDescriptor_t descriptor)
{
    delete descriptor;

    return PRNN_STATUS_SUCCESS;
}

struct prnnRNNStruct
{
public:
    size_t hiddenSize;
    size_t sequenceLength;
    size_t numberOfLayers;

public:
    prnnRNNInputMode_t  inputMode;
    prnnDirectionMode_t direction;
    prnnRNNMode_t       mode;
    prnnDataType_t      dataType;
    prnnBackend_t       backend;


};

prnnStatus_t prnnCreateRNNDescriptor(prnnRNNDescriptor_t* descriptor)
{
    try
    {
        *descriptor = new prnnRNNStruct;
    }
    catch(std::bad_alloc)
    {
        return PRNN_STATUS_ALLOC_FAILED;
    }

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnDestroyRNNDescriptor(prnnRNNDescriptor_t descriptor)
{
    delete descriptor;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnSetRNNDescriptor(prnnRNNDescriptor_t rnnDescriptor,
                                  int hiddenSize,
                                  int numberOfLayers,
                                  prnnDropoutDescriptor_t dropoutDescriptor,
                                  prnnRNNInputMode_t inputMode,
                                  prnnDirectionMode_t direction,
                                  prnnRNNMode_t mode,
                                  prnnDataType_t dataType,
                                  prnnBackend_t backend)
{
    rnnDescriptor->hiddenSize     = hiddenSize;
    rnnDescriptor->numberOfLayers = numberOfLayers;

    rnnDescriptor->inputMode = inputMode;
    rnnDescriptor->direction = direction;
    rnnDescriptor->mode      = mode;
    rnnDescriptor->dataType  = dataType;
    rnnDescriptor->backend   = backend;

    return PRNN_STATUS_SUCCESS;
}

static prnn::matrix::DynamicView constructView(const prnnTensorDescriptor_t descriptor,
    void* data, size_t timesteps)
{
    auto size   = selectDimensions(descriptor->size,   {1, 0, 2});
    auto stride = selectDimensions(descriptor->stride, {1, 0, 2});

    stride[2] = size[0] * size[1];
    size[2]   = timesteps;

    return prnn::matrix::DynamicView(data, size, stride, descriptor->precision);
}

static prnn::matrix::ConstDynamicView constructView(const prnnTensorDescriptor_t descriptor,
    const void* data, size_t timesteps)
{
    auto size   = selectDimensions(descriptor->size,   {1, 0, 2});
    auto stride = selectDimensions(descriptor->stride, {1, 0, 2});

    stride[2] = size[0] * size[1];
    size[2]   = timesteps;

    return prnn::matrix::ConstDynamicView(data, size, stride, descriptor->precision);
}

static prnn::matrix::ConstDynamicView constructView(const prnnTensorDescriptor_t descriptor,
    size_t timesteps)
{
    auto size   = selectDimensions(descriptor->size,   {1, 0, 2});
    auto stride = selectDimensions(descriptor->stride, {1, 0, 2});

    stride[2] = size[0] * size[1];
    size[2]   = timesteps;

    return prnn::matrix::ConstDynamicView(nullptr, size, stride, descriptor->precision);
}

static prnn::matrix::DynamicView constructWeightsView(const prnnTensorDescriptor_t descriptor,
    void* data)
{
    return prnn::matrix::DynamicView(data, descriptor->size, descriptor->stride,
        descriptor->precision);
}

static prnn::matrix::ConstDynamicView constructWeightsView(
    const prnnTensorDescriptor_t descriptor, const void* data)
{
    return prnn::matrix::ConstDynamicView(data, descriptor->size, descriptor->stride,
        descriptor->precision);
}

static bool isSupported(prnnRNNDescriptor_t desc)
{
    if(desc->direction == PRNN_REVERSE)
    {
        return false;
    }

    return true;
}

static bool isForwardPropSupported(const void* x, const void* hx,
    const void* cx, const prnnFilterDescriptor_t wDesc,
    void* y, void* hy, void* cy)
{
    return true;
}

static bool isBackPropDeltasSupported(const void* hx,
    const void* cx, const prnnFilterDescriptor_t wDesc,
    const void* y, const void* dy, const void* dhy, const void* dcy)
{
    return true;
}

static bool isBackPropGradientsSupported(const void* x, const void* hx,
    const prnnFilterDescriptor_t dwDesc,
    const void* y)
{
    return true;
}

static prnn::RecurrentActivationFunction getActivationFunction(prnnRNNMode_t mode)
{
    if(mode == PRNN_RNN_RELU)
    {
        return prnn::RecurrentRectifiedLinear();
    }
    else if(mode == PRNN_RNN_TANH)
    {
        return prnn::RecurrentHyperbolicTangent();
    }
    else
    {
        return prnn::RecurrentActivationFunction();
    }
}

static prnn::RecurrentLayerDirection getDirection(prnnDirectionMode_t mode)
{
    if(mode == PRNN_UNIDIRECTIONAL)
    {
        return prnn::RECURRENT_FORWARD;
    }
    else if (mode == PRNN_BIDIRECTIONAL)
    {
        return prnn::RECURRENT_BIDIRECTIONAL;
    }
    else
    {
        return prnn::RECURRENT_REVERSE;
    }
}

static prnn::RecurrentLayerType getLayerType(prnnRNNMode_t mode)
{
    if(mode == PRNN_RNN_RELU || mode == PRNN_RNN_TANH)
    {
        return prnn::RECURRENT_SIMPLE_TYPE;
    }
    else if(mode == PRNN_GRU)
    {
        return prnn::RECURRENT_GRU_TYPE;
    }
    else
    {
        return prnn::RECURRENT_LSTM_TYPE;
    }
}

static prnn::RecurrentLayerInputMode getLayerInputMode(prnnRNNInputMode_t mode)
{
    if(mode == PRNN_LINEAR_INPUT)
    {
        return prnn::RECURRENT_LINEAR_INPUT;
    }
    else
    {
        return prnn::RECURRENT_SKIP_INPUT;
    }
}

static prnn::RecurrentLayerBackend getBackend(prnnBackend_t backend)
{
    if(backend == PRNN_PERSISTENT_BACKEND)
    {
        return prnn::RECURRENT_PERSISTENT_BACKEND;
    }
    else if(backend == PRNN_CUDNN_BACKEND)
    {
        return prnn::RECURRENT_CUDNN_BACKEND;
    }
    else
    {
        return prnn::RECURRENT_BEST_BACKEND;
    }
}

static prnn::RecurrentOpsHandle constructHandle(prnnHandle_t handle,
    const prnnRNNDescriptor_t rnnDesc, size_t miniBatchSize, size_t timesteps)
{
    return prnn::RecurrentOpsHandle(rnnDesc->hiddenSize, miniBatchSize, timesteps,
        rnnDesc->numberOfLayers,
        getActivationFunction(rnnDesc->mode),
        getDirection(rnnDesc->direction),
        getLayerType(rnnDesc->mode),
        getLayerInputMode(rnnDesc->inputMode),
        getBackend(rnnDesc->backend));
}

prnnStatus_t prnnGetRNNWorkspaceSize(prnnHandle_t cHandle,
                                     const prnnRNNDescriptor_t rnnDesc,
                                     const int timesteps,
                                     const prnnTensorDescriptor_t* xDesc,
                                     size_t* sizeInBytes)
{

    if(xDesc == nullptr)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    if((*xDesc)->size.size() != 3)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    prnn::RecurrentOpsHandle handle = constructHandle(cHandle, rnnDesc,
        (*xDesc)->size[0], timesteps);

    *sizeInBytes = prnn::rnn::getForwardPropScratchSize(handle, (*xDesc)->precision);

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetRNNTrainingReserveSize(prnnHandle_t handle,
                                           const prnnRNNDescriptor_t rnnDesc,
                                           const int timesteps,
                                           const prnnTensorDescriptor_t* xDesc,
                                           size_t* sizeInBytes)
{
    if((*xDesc)->size.size() != 3)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    auto activationsView = constructView(*xDesc, timesteps);

    size_t miniBatchSize = activationsView.size()[1];

    *sizeInBytes = getPrecision(rnnDesc->dataType).size() * prnn::rnn::getReserveDimensions(
        constructHandle(handle, rnnDesc, miniBatchSize, timesteps), (*xDesc)->precision).product();

    return PRNN_STATUS_SUCCESS;
}


prnnStatus_t prnnGetRNNParamsSize(prnnHandle_t handle,
                                  const prnnRNNDescriptor_t rnnDesc,
                                  const prnnTensorDescriptor_t* xDesc,
                                  size_t* sizeInBytes)
{
    if((*xDesc)->size.size() != 3)
    {
        return PRNN_STATUS_BAD_PARAM;
    }

    auto activationsView = constructView(*xDesc, 1);

    size_t miniBatchSize = activationsView.size()[1];

    *sizeInBytes = getPrecision(rnnDesc->dataType).size() * prnn::rnn::getWeightDimensions(
        constructHandle(handle, rnnDesc, miniBatchSize, 1), (*xDesc)->precision).product();

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetRNNLinLayerMatrixParams(prnnHandle_t handle,
                                            const prnnRNNDescriptor_t rnnDesc,
                                            const int layer,
                                            const prnnTensorDescriptor_t* xDesc,
                                            const prnnFilterDescriptor_t wDesc,
                                            const void* w,
                                            const int linLayerID,
                                            prnnFilterDescriptor_t linLayerMatDesc,
                                            void** linLayerMat)
{
    return PRNN_STATUS_NOT_SUPPORTED;
}

prnnStatus_t prnnGetRNNLinLayerBiasParams(prnnHandle_t handle,
                                          const prnnRNNDescriptor_t rnnDesc,
                                          const int layer,
                                          const prnnTensorDescriptor_t* xDesc,
                                          const prnnFilterDescriptor_t wDesc,
                                          const void* w,
                                          const int linLayerID,
                                          prnnFilterDescriptor_t linLayerBiasDesc,
                                          void** linLayerBias)
{
    return PRNN_STATUS_NOT_SUPPORTED;
}

static prnn::matrix::DynamicView getView(void* workspace, size_t size,
    const prnn::matrix::Precision& precision)
{
    return prnn::matrix::DynamicView(workspace, {size / precision.size()}, {1}, precision);
}

static prnn::matrix::ConstDynamicView getView(const void* workspace, size_t size,
    const prnn::matrix::Precision& precision)
{
    return prnn::matrix::ConstDynamicView(workspace, {size / precision.size()}, {1}, precision);
}

prnnStatus_t prnnRNNForward(prnnHandle_t handle,
                            const prnnRNNDescriptor_t rnnDesc,
                            const int seqLength,
                            const prnnTensorDescriptor_t* xDesc,
                            const void* x,
                            const prnnTensorDescriptor_t hxDesc,
                            const void* hx,
                            const prnnTensorDescriptor_t cxDesc,
                            const void* cx,
                            const prnnFilterDescriptor_t wDesc,
                            const void* w,
                            const prnnTensorDescriptor_t* yDesc,
                            void* y,
                            const prnnTensorDescriptor_t hyDesc,
                            void* hy,
                            const prnnTensorDescriptor_t cyDesc,
                            void* cy,
                            void* workspace,
                            size_t workSpaceSizeInBytes,
                            void* reserveSpace,
                            size_t reserveSpaceSizeInBytes)
{

    if (!isSupported(rnnDesc))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    if (!isForwardPropSupported(x, hx, cx, wDesc, y, hy, cy))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    size_t timesteps     = seqLength;

    auto activationsView = constructView(*yDesc, y, timesteps);
    auto inputView       = constructView(*xDesc, x, timesteps);
    auto weightsView     = constructWeightsView( wDesc, w);

    size_t miniBatchSize = activationsView.size()[1];

    auto opsHandle = constructHandle(handle, rnnDesc, miniBatchSize, timesteps);

    auto scratchView = getView(workspace, workSpaceSizeInBytes,
        activationsView.precision());

    auto reserveView = getView(reserveSpace, reserveSpaceSizeInBytes,
        activationsView.precision());

    prnn::rnn::forwardPropRecurrent(activationsView, inputView, weightsView,
        scratchView, reserveView, opsHandle);

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnRNNBackwardData(prnnHandle_t handle,
                                 const prnnRNNDescriptor_t rnnDesc,
                                 const int seqLength,
                                 const prnnTensorDescriptor_t* yDesc,
                                 const void* y,
                                 const prnnTensorDescriptor_t* dyDesc,
                                 const void* dy,
                                 const prnnTensorDescriptor_t dhyDesc,
                                 const void* dhy,
                                 const prnnTensorDescriptor_t dcyDesc,
                                 const void* dcy,
                                 const prnnFilterDescriptor_t wDesc,
                                 const void* w,
                                 const prnnTensorDescriptor_t hxDesc,
                                 const void* hx,
                                 const prnnTensorDescriptor_t cxDesc,
                                 const void* cx,
                                 const prnnTensorDescriptor_t* dxDesc,
                                 void* dx,
                                 const prnnTensorDescriptor_t dhxDesc,
                                 void* dhx,
                                 const prnnTensorDescriptor_t dcxDesc,
                                 void* dcx,
                                 void* workspace,
                                 size_t workSpaceSizeInBytes,
                                 void* reserveSpace,
                                 size_t reserveSpaceSizeInBytes)
{
    if (!isSupported(rnnDesc))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    if (!isBackPropDeltasSupported(hx, cx, wDesc, y, dy, dhy, dcy))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    size_t timesteps     = seqLength;

    auto activationsView  = constructView(*yDesc,  y,  timesteps);
    auto deltasView       = constructView(*dxDesc, dx, timesteps);
    auto outputDeltasView = constructView(*dyDesc, dy, timesteps);
    auto weightsView      = constructWeightsView(wDesc,  w);

    size_t miniBatchSize = activationsView.size()[1];

    auto opsHandle = constructHandle(handle, rnnDesc, miniBatchSize, timesteps);

    auto scratchView = getView(workspace, workSpaceSizeInBytes,
        activationsView.precision());

    auto reserveView = getView(reserveSpace, reserveSpaceSizeInBytes,
        activationsView.precision());

    prnn::rnn::backPropDeltasRecurrent(deltasView, weightsView, activationsView,
        outputDeltasView, scratchView, reserveView, opsHandle);

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnRNNBackwardWeights(prnnHandle_t handle,
                                    const prnnRNNDescriptor_t rnnDesc,
                                    const int seqLength,
                                    const prnnTensorDescriptor_t* xDesc,
                                    const void* x,
                                    const prnnTensorDescriptor_t hxDesc,
                                    const void* hx,
                                    const prnnTensorDescriptor_t * yDesc,
                                    const void* y,
                                    const void* workspace,
                                    size_t workSpaceSizeInBytes,
                                    const prnnFilterDescriptor_t dwDesc,
                                    void* dw,
                                    const void* reserveSpace,
                                    size_t reserveSpaceSizeInBytes)
{
    if (!isSupported(rnnDesc))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    if (!isBackPropGradientsSupported(x, hx, dwDesc, y))
    {
        return PRNN_STATUS_NOT_SUPPORTED;
    }

    size_t timesteps     = seqLength;

    auto inputActivationsView  = constructView(*xDesc,  x, timesteps);
    auto outputActivationsView = constructView(*yDesc,  y, timesteps);
    auto weightsView           = constructWeightsView(dwDesc, dw);

    size_t miniBatchSize = inputActivationsView.size()[1];

    auto opsHandle = constructHandle(handle, rnnDesc, miniBatchSize, timesteps);

    auto scratchView = getView(workspace, workSpaceSizeInBytes,
        inputActivationsView.precision());

    auto reserveView = getView(reserveSpace, reserveSpaceSizeInBytes,
        inputActivationsView.precision());

    prnn::rnn::backPropGradientsRecurrent(weightsView, inputActivationsView, outputActivationsView,
        scratchView, reserveView, opsHandle);

    return PRNN_STATUS_SUCCESS;
}


