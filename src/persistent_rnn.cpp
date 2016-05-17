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

// Standard Library Includes
#include <new>

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
    cudaStream_t stream;
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

prnnStatus_t prnnSetStream(prnnHandle_t handle, cudaStream_t streamId)
{
    handle->stream = streamId;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetStream(prnnHandle_t handle, cudaStream_t* streamId)
{
    *streamId = handle->stream;

    return PRNN_STATUS_SUCCESS;
}

struct prnnTensorStruct
{
public:
    prnnTensorFormat_t format;
    prnnDataType_t     dataType;

public:
    prnn::matrix::Dimension size;
    prnn::matrix::Dimension stride;
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

prnnStatus_t prnnSetTensor4dDescriptor(prnnTensorDescriptor_t descriptor,
                                       prnnTensorFormat_t     format,
                                       prnnDataType_t         dataType,
                                       int                    n,
                                       int                    c,
                                       int                    h,
                                       int                    w)
{
    descriptor->format   = format;
    descriptor->dataType = dataType;
    descriptor->size     = prnn::matrix::Dimension(w, h, c, n);
    descriptor->stride   = prnn::matrix::linearStride(descriptor->size);

    return PRNN_STATUS_SUCCESS;
}


prnnStatus_t prnnSetTensor4dDescriptorEx(prnnTensorDescriptor_t descriptor,
                                         prnnDataType_t         dataType,
                                         int                    n,
                                         int                    c,
                                         int                    h,
                                         int                    w,
                                         int                    nStride,
                                         int                    cStride,
                                         int                    hStride,
                                         int                    wStride)
{
    descriptor->format   = PRNN_TENSOR_NCHW;
    descriptor->dataType = dataType;
    descriptor->size     = prnn::matrix::Dimension(w, h, c, n);
    descriptor->stride   = prnn::matrix::Dimension(wStride, hStride, cStride, nStride);

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetTensor4dDescriptor(const prnnTensorDescriptor_t descriptor,
                                       prnnDataType_t*              dataType,
                                       int*                         n,
                                       int*                         c,
                                       int*                         h,
                                       int*                         w,
                                       int*                         nStride,
                                       int*                         cStride,
                                       int*                         hStride,
                                       int*                         wStride)
{
    *dataType = descriptor->dataType;

    *w = descriptor->size[0];
    *h = descriptor->size[1];
    *c = descriptor->size[2];
    *n = descriptor->size[3];

    *wStride = descriptor->stride[0];
    *hStride = descriptor->stride[1];
    *cStride = descriptor->stride[2];
    *nStride = descriptor->stride[3];

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnSetTensorNdDescriptor(prnnTensorDescriptor_t descriptor,
                                       prnnDataType_t         dataType,
                                       int                    nbDims,
                                       const int*             dimA,
                                       const int*             strideA)
{
    descriptor->dataType = dataType;

    if (nbDims < 0 || nbDims > PRNN_DIM_MAX)
    {
        return PRNN_STATUS_INVALID_VALUE;
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

    *dataType = descriptor->dataType;

    if (nbDimsRequested < 0 || nbDimsRequested > PRNN_DIM_MAX)
    {
        return PRNN_STATUS_INVALID_VALUE;
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
                                  int sequenceLength,
                                  int numberOfLayers,
                                  prnnDropoutDescriptor_t dropoutDescriptor,
                                  prnnRNNInputMode_t inputMode,
                                  prnnDirectionMode_t direction,
                                  prnnRNNMode_t mode,
                                  prnnDataType_t dataType)
{
    rnnDescriptor->hiddenSize     = hiddenSize;
    rnnDescriptor->sequenceLength = sequenceLength;
    rnnDescriptor->numberOfLayers = numberOfLayers;

    rnnDescriptor->inputMode = inputMode;
    rnnDescriptor->direction = direction;
    rnnDescriptor->mode      = mode;
    rnnDescriptor->dataType  = dataType;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetRNNWorkspaceSize(prnnHandle_t handle,
                                     const prnnRNNDescriptor_t rnnDesc,
                                     const prnnTensorDescriptor_t* xDesc,
                                     size_t* sizeInBytes)
{
    return PRNN_STATUS_NOT_SUPPORTED;
}

prnnStatus_t prnnGetRNNTrainingReserveSize(prnnHandle_t handle,
                                           const prnnRNNDescriptor_t rnnDesc,
                                           const prnnTensorDescriptor_t* xDesc,
                                           size_t* sizeInBytes)
{
    return PRNN_STATUS_NOT_SUPPORTED;
}


prnnStatus_t prnnGetRNNParamsSize(prnnHandle_t handle,
                                  const prnnRNNDescriptor_t rnnDesc,
                                  const prnnTensorDescriptor_t* xDesc,
                                  size_t* sizeInBytes)
{
    return PRNN_STATUS_NOT_SUPPORTED;
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

static bool isSupported(prnnRNNDescriptor_t desc)
{
    if(desc->hiddenSize > prnn::rnn::getMaximumSizeRNNForThisGPU())
    {
        return false;
    }

    if(desc->numberOfLayers > 1)
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

static prnn::matrix::DynamicView constructView(const prnnTensorDescriptor_t descriptor,
    void* data)
{
    return prnn::matrix::DynamicView(data, descriptor->size, descriptor->stride);
}

static prnn::matrix::ConstDynamicView constructView(const prnnTensorDescriptor_t descriptor,
    const void* data)
{
    return prnn::matrix::ConstDynamicView(data, descriptor->size, descriptor->stride);
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
    return prnn::RECURRENT_FORWARD;
}

static prnn::RecurrentOpsHandle constructHandle(prnnHandle_t handle,
    const prnnRNNDescriptor_t rnnDesc, size_t miniBatchSize)
{
    return prnn::RecurrentOpsHandle(rnnDesc->hiddenSize, miniBatchSize,
        getActivationFunction(rnnDesc->mode),
        getDirection(rnnDesc->direction));
}

static prnn::matrix::DynamicView getScratchView(void* workspace, size_t size)
{
    return prnn::matrix::DynamicView(workspace, {size}, {1});
}

prnnStatus_t prnnRNNForward(prnnHandle_t handle,
                            const prnnRNNDescriptor_t rnnDesc,
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

    auto activationsView = constructView(*yDesc, y);
    auto weightsView     = constructView( wDesc, w);

    size_t miniBatchSize = activationsView.size()[3];

    auto opsHandle = constructHandle(handle, rnnDesc, miniBatchSize);

    auto scratchView = getScratchView(workspace, workSpaceSizeInBytes);

    prnn::rnn::forwardPropRecurrent(activationsView, weightsView, scratchView, opsHandle);

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnRNNBackwardData(prnnHandle_t handle,
                                 const prnnRNNDescriptor_t rnnDesc,
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
                                 const void* reserveSpace,
                                 size_t reserveSpaceSizeInBytes)
{
    return PRNN_STATUS_NOT_SUPPORTED;
}

prnnStatus_t prnnRNNBackwardWeights(prnnHandle_t handle,
                                    const prnnRNNDescriptor_t rnnDesc,
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
    return PRNN_STATUS_NOT_SUPPORTED;
}


