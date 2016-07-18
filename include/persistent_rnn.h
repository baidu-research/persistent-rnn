/*! \file  persistent_rnn.h
    \date  May 10, 2016
    \brief C language interface to persistent RNN kernels, modeled after the CUDNNv5 RNN interface
           for maximum compatibility.
*/

#if !defined(PRNN_H_)
#define PRNN_H_

#define PRNN_MAJOR      0
#define PRNN_MINOR      2
#define PRNN_PATCHLEVEL 0

#define PRNN_VERSION    (PRNN_MAJOR * 1000 + PRNN_MINOR * 100 + PRNN_PATCHLEVEL)

// Standard Library Includes
#include <stddef.h>

#if defined (__cplusplus)
extern "C" {
#endif

struct prnnContext;
typedef struct prnnContext* prnnHandle_t;

size_t prnnGetVersion(void);

/*
 * PRNN return codes
 */
typedef enum
{
    PRNN_STATUS_SUCCESS          = 0,
    PRNN_STATUS_NOT_INITIALIZED  = 1,
    PRNN_STATUS_ALLOC_FAILED     = 2,
    PRNN_STATUS_BAD_PARAM        = 3,
    PRNN_STATUS_INTERNAL_ERROR   = 4,
    PRNN_STATUS_INVALID_VALUE    = 5,
    PRNN_STATUS_ARCH_MISMATCH    = 6,
    PRNN_STATUS_MAPPING_ERROR    = 7,
    PRNN_STATUS_EXECUTION_FAILED = 8,
    PRNN_STATUS_NOT_SUPPORTED    = 9
} prnnStatus_t;

// human-readable error messages
const char* prnnGetErrorString(prnnStatus_t status);

prnnStatus_t prnnCreate    (prnnHandle_t* handle);
prnnStatus_t prnnDestroy   (prnnHandle_t handle);
prnnStatus_t prnnSetStream (prnnHandle_t handle, void* streamId);
prnnStatus_t prnnGetStream (prnnHandle_t handle, void** streamId);

/* Data structures to represent input data and the Neural Network Layer */
typedef struct prnnTensorStruct* prnnTensorDescriptor_t;
typedef struct prnnTensorStruct* prnnFilterDescriptor_t;
typedef struct prnnDropoutStruct* prnnDropoutDescriptor_t;

/*
* PRNN data type
*/
typedef enum
{
    PRNN_DATA_FLOAT   = 0,
    PRNN_DATA_DOUBLE  = 1,
    PRNN_DATA_HALF    = 2,
    PRNN_INVALID_DATA = 3,
} prnnDataType_t;

/* Maximum supported number of tensor dimensions */
#define PRNN_DIM_MAX 8

/* Create an instance of a generic Tensor descriptor */
prnnStatus_t prnnCreateTensorDescriptor(prnnTensorDescriptor_t* tensorDesc);

typedef enum
{
    PRNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    PRNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} prnnTensorFormat_t;

prnnStatus_t prnnSetTensorNdDescriptor(prnnTensorDescriptor_t tensorDesc,
                                       prnnDataType_t         dataType,
                                       int                    nbDims,
                                       const int*             dimA,
                                       const int*             strideA);

prnnStatus_t prnnGetTensorNdDescriptor(const prnnTensorDescriptor_t tensorDesc,
                                       int                          nbDimsRequested,
                                       prnnDataType_t*              dataType,
                                       int*                         nbDims,
                                       int*                         dimA,
                                       int*                         strideA);

/* Destroy an instance of Tensor4d descriptor */
prnnStatus_t prnnDestroyTensorDescriptor(prnnTensorDescriptor_t tensorDesc);

/* RNN API */
typedef enum
{
    PRNN_RNN_RELU = 0, // Stock RNN with ReLu activation
    PRNN_RNN_TANH = 1, // Stock RNN with tanh activation
    PRNN_LSTM     = 2, // LSTM with no peephole connections
    PRNN_GRU      = 3  // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
} prnnRNNMode_t;

typedef enum
{
   PRNN_UNIDIRECTIONAL = 0,
   PRNN_BIDIRECTIONAL  = 1,  // Using output concatination at each step. Do we also want to support output sum?
   PRNN_REVERSE        = 2
} prnnDirectionMode_t;

typedef enum
{
   PRNN_LINEAR_INPUT = 0,
   PRNN_SKIP_INPUT   = 1
} prnnRNNInputMode_t;

typedef enum
{
    PRNN_PERSISTENT_BACKEND = 0,
    PRNN_CUDNN_BACKEND = 1,
    PRNN_BEST_BACKEND = 2
} prnnBackend_t;

struct prnnRNNStruct;
typedef struct prnnRNNStruct* prnnRNNDescriptor_t;

prnnStatus_t  prnnCreateRNNDescriptor(prnnRNNDescriptor_t* rnnDesc);
prnnStatus_t prnnDestroyRNNDescriptor(prnnRNNDescriptor_t rnnDesc);

prnnStatus_t prnnSetRNNDescriptor(prnnRNNDescriptor_t rnnDesc,
                                  int hiddenSize,
                                  int numLayers,
                                  prnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                  prnnRNNInputMode_t inputMode,
                                  prnnDirectionMode_t direction,
                                  prnnRNNMode_t mode,
                                  prnnDataType_t dataType,
                                  prnnBackend_t backend);

// dataType in the RNN descriptor is used to determine math precision
// dataType in weight descriptors and input descriptors is used to describe storage

prnnStatus_t prnnGetRNNWorkspaceSize(prnnHandle_t handle,
                                     const prnnRNNDescriptor_t rnnDesc,
                                     const int seqLength,
                                     const prnnTensorDescriptor_t* xDesc,
                                     size_t* sizeInBytes
                                     );

prnnStatus_t prnnGetRNNTrainingReserveSize(prnnHandle_t handle,
                                           const prnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const prnnTensorDescriptor_t* xDesc,
                                           size_t* sizeInBytes
                                           );


prnnStatus_t prnnGetRNNParamsSize(prnnHandle_t handle,
                                  const prnnRNNDescriptor_t rnnDesc,
                                  const prnnTensorDescriptor_t* xDesc,
                                  size_t* sizeInBytes
                                  );

prnnStatus_t prnnGetRNNLinLayerMatrixParams(prnnHandle_t handle,
                                            const prnnRNNDescriptor_t rnnDesc,
                                            const int layer,
                                            const prnnTensorDescriptor_t* xDesc,
                                            const prnnFilterDescriptor_t wDesc,
                                            const void* w,
                                            const int linLayerID,
                                            prnnFilterDescriptor_t linLayerMatDesc,
                                            void** linLayerMat
                                            );

prnnStatus_t prnnGetRNNLinLayerBiasParams(prnnHandle_t handle,
                                          const prnnRNNDescriptor_t rnnDesc,
                                          const int layer,
                                          const prnnTensorDescriptor_t* xDesc,
                                          const prnnFilterDescriptor_t wDesc,
                                          const void* w,
                                          const int linLayerID,
                                          prnnFilterDescriptor_t linLayerBiasDesc,
                                          void** linLayerBias
                                          );


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
                            size_t reserveSpaceSizeInBytes);

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
                                 size_t reserveSpaceSizeInBytes);


prnnStatus_t prnnRNNBackwardWeights(prnnHandle_t handle,
                                    const prnnRNNDescriptor_t rnnDesc,
                                    const int seqLength,
                                    const prnnTensorDescriptor_t* xDesc,
                                    const void* x,
                                    const prnnTensorDescriptor_t hxDesc,
                                    const void* hx,
                                    const prnnTensorDescriptor_t* yDesc,
                                    const void* y,
                                    const void* workspace,
                                    size_t workSpaceSizeInBytes,
                                    const prnnFilterDescriptor_t dwDesc,
                                    void* dw,
                                    const void* reserveSpace,
                                    size_t reserveSpaceSizeInBytes);

#if defined (__cplusplus)
}
#endif

#endif

