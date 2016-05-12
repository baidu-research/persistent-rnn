/*! \file  persistent_rnn.cpp
    \date  May 10, 2016
    \brief C language interface to persistent RNN kernels, modeled after the CUDNNv5 RNN interface
           for maximum compatibility.
*/

#include <prnn/persistent_rnn.h>

size_t prnnGetVersion(void) {
    return PRNN_VERSION;
}

const char* prnnGetErrorString(prnnStatus_t status) {
    switch(status) {
    case PRNN_STATUS_SUCCESS: {
        return "success";
    }
    case PRNN_STATUS_NOT_INITIALIZED: {
        return "not initialized";
    }
    case PRNN_STATUS_ALLOC_FAILED: {
        return "memory allocation failed";
    }
    case PRNN_STATUS_BAD_PARAM: {
        return "bad parameter";
    }
    case PRNN_STATUS_INTERNAL_ERROR: {
        return "internal error";
    }
    case PRNN_STATUS_INVALID_VALUE: {
        return "invalid value";
    }
    case PRNN_STATUS_ARCH_MISMATCH: {
        return "wrong architecture";
    }
    case PRNN_STATUS_MAPPING_ERROR: {
        return "mapping error";
    }
    case PRNN_STATUS_EXECUTION_FAILED: {
        return "kernel execution failed";
    }
    case PRNN_STATUS_NOT_SUPPORTED: {
        return "not supported";
    }
    default: {
        break;
    }
    }

    return "unknown error code";
}

prnnStatus_t prnnCreate(prnnHandle_t* handle) {
    try {
        *handle = new prnnContext;
    }
    catch(std::bad_alloc) {
        return PRNN_STATUS_ALLOC_FAILED;
    }

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnDestroy(prnnHandle_t handle) {
    delete handle;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnSetStream(prnnHandle_t handle, cudaStream_t streamId) {
    handle->stream = streamId;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnGetStream(prnnHandle_t handle, cudaStream_t* streamId) {
    *streamId = handle->stream;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnCreateTensorDescriptor(prnnTensorDescriptor_t* handle) {
    try {
        *handle = new prnnTensorStruct;
    }
    catch(std::bad_alloc) {
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
                                       int                    w) {
    descriptor->format   = format;
    descriptor->dataType = dataType;
    descriptor->size     = {w, h, c, n};
    descriptor->stride   = linearStrides(descriptor->size);

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
                                         int                    wStride) {
    descriptor->format   = format;
    descriptor->dataType = dataType;
    descriptor->size     = {w, h, c, n};
    descriptor->stride   = {wStride, hStride, cStride, nStride};

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
                                       int*                         wStride) {
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
                                       const int*             strideA) {
    descripitor->dataType = dataType;

    if (nbDims < 0 || nbDims > PRNN_DIM_MAX) {
        return PRNN_STATUS_INVALID_VALUE;
    }

    descriptor->size.clear();
    descriptor->stride.clear();

    for (int i = 0; i < nbDims; ++i) {
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

    int dims = std::min(descriptor->size.size(), nbDimsRequested);

    for (int i = 0; i < dims; ++i)
    {
        dimA[i]    = descriptor->size[i];
        strideA[i] = descriptor->stride[i];
    }

    *nbDims = dims;

    return PRNN_STATUS_SUCCESS;
}

prnnStatus_t prnnDestroyTensorDescriptor(prnnTensorDescriptor_t descriptor) {
    delete descriptor;

    return PRNN_STATUS_SUCCESS;
}

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
                                     size_t* sizeInBytes) {

}

prnnStatus_t prnnGetRNNTrainingReserveSize(prnnHandle_t handle,
                                           const prnnRNNDescriptor_t rnnDesc,
                                           const prnnTensorDescriptor_t* xDesc,
                                           size_t* sizeInBytes) {

}


prnnStatus_t prnnGetRNNParamsSize(prnnHandle_t handle,
                                  const prnnRNNDescriptor_t rnnDesc,
                                  const prnnTensorDescriptor_t* xDesc,
                                  size_t* sizeInBytes) {

}

prnnStatus_t prnnGetRNNLinLayerMatrixParams(prnnHandle_t handle,
                                            const prnnRNNDescriptor_t rnnDesc,
                                            const int layer,
                                            const prnnTensorDescriptor_t* xDesc,
                                            const prnnFilterDescriptor_t wDesc,
                                            const void* w,
                                            const int linLayerID,
                                            prnnFilterDescriptor_t linLayerMatDesc,
                                            void** linLayerMat) {

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

    MatrixView activationsView = constructView(yDesc, y);
    MatrixView weightsView = constructView(wDesc, w);

    RecurrentOpsHandle opsHandle = constructHandle(handle, rnnDesc);

    MatrixView scratchView = getScratchView(worspace, workSpaceSizeInBytes);

    forward_prop_recurrent(opsHandle, weightsView, activationsView, scratchView);
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
                                    size_t reserveSpaceSizeInBytes) {
}


