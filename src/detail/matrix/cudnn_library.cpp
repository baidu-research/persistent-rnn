/*  \file   CudnnLibrary.cpp
    \date   April 23, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the CudnnLibrary class.
*/


// Standard Library Includes
#include <prnn/detail/matrix/cudnn_library.h>

#include <prnn/detail/parallel/cuda.h>

#include <prnn/detail/util/casts.h>
#include <prnn/detail/util/logger.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace prnn
{

namespace matrix
{

void CudnnLibrary::CudnnLibrary::load()
{
    _interface.load();
}

bool CudnnLibrary::loaded()
{
    load();

    return _interface.loaded();
}

bool CudnnLibrary::isSupported()
{
    return loaded();
}

void CudnnLibrary::cudnnSetStream(void* stream)
{
    _check();

    auto status = (*_interface.cudnnSetStream)(_interface.getHandle(), stream);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetStream failed: " +
            _interface.getErrorString(status));
    }
}

/* Create an instance of a generic Tensor descriptor */
void CudnnLibrary::cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateTensorDescriptor)(tensorDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateTensorDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t   tensorDesc,
                                   cudnnTensorFormat_t  format,
                                   cudnnDataType_t dataType, // image data type
                                   int n,        // number of inputs (batch size)
                                   int c,        // number of input feature maps
                                   int h,        // height of input section
                                   int w         // width of input section
                                   )
{
    _check();

    auto status = (*_interface.cudnnSetTensor4dDescriptor)(tensorDesc, format,
        dataType, n, c, h, w);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetTensor4dDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t   tensorDesc,
                                              cudnnDataType_t dataType,
                                              int                    nbDims,
                                              const int*             dimA,
                                              const int*             strideA
                                              )
{
    _check();

    auto status = (*_interface.cudnnSetTensorNdDescriptor)(tensorDesc,
        dataType, nbDims, dimA, strideA);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetTensorNdDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                              int                     nbDimsRequested,
                                              cudnnDataType_t*        dataType,
                                              int*                    nbDims,
                                              int*                    dimA,
                                              int*                    strideA)
{
    _check();

    auto status = (*_interface.cudnnGetTensorNdDescriptor)(tensorDesc,
        nbDimsRequested, dataType, nbDims, dimA, strideA);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetTensorNdDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyTensorDescriptor)(tensorDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyTensorDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnTransformTensor(const void*                      alpha,
                             const cudnnTensorDescriptor_t    srcDesc,
                             const void*                      srcData,
                             const void*                      beta,
                             const cudnnTensorDescriptor_t    destDesc,
                             void*                            destData)
{
    _check();

    auto status = (*_interface.cudnnTransformTensor)(_interface.getHandle(), alpha, srcDesc,
        srcData, beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnTransformTensor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateFilterDescriptor)(filterDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateFilterDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                   cudnnDataType_t dataType, // image data type
                                   cudnnTensorFormat_t format,
                                   int k,        // number of output feature maps
                                   int c,        // number of input feature maps
                                   int h,        // height of each input filter
                                   int w         // width of  each input fitler
                                   )
{
    _check();

    auto status = (*_interface.cudnnSetFilter4dDescriptor)(filterDesc, dataType, format, k, c, h, w);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetFilter4dDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                   cudnnDataType_t dataType, // image data type
                                   cudnnTensorFormat_t format,
                                   int dims,
                                   int* sizes)
{
    _check();

    auto status = (*_interface.cudnnSetFilterNdDescriptor)(filterDesc, dataType, format, dims, sizes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetFilterNdDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t wDesc,
                                              int nbDimsRequested,
                                              cudnnDataType_t* dataType,
                                              cudnnTensorFormat_t* format,
                                              int* nbDims,
                                              int* filterDimA)
{
    _check();

    auto status = (*_interface.cudnnGetFilterNdDescriptor)(wDesc, nbDimsRequested, dataType,
        format, nbDims, filterDimA);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetFilterNdDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyFilterDescriptor)(filterDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyFilterDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateConvolutionDescriptor)(convDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateConvolutionDescriptor failed: " + _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                        int pad_h,    // zero-padding height
                                        int pad_w,    // zero-padding width
                                        int u,        // vertical filter stride
                                        int v,        // horizontal filter stride
                                        int upscalex, // upscale the input in x-direction
                                        int upscaley, // upscale the input in y-direction
                                        cudnnConvolutionMode_t mode)
{
    _check();

    auto status = (*_interface.cudnnSetConvolution2dDescriptor)(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetConvolution2dDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyConvolutionDescriptor)(convDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyConvolutionDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc)
{
    _check();

    auto status = (*_interface.cudnnCreatePoolingDescriptor)(poolingDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreatePoolingDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                    cudnnPoolingMode_t mode,
                                                    int windowHeight,
                                                    int windowWidth,
                                                    int verticalPadding,
                                                    int horizontalPadding,
                                                    int verticalStride,
                                                    int horizontalStride
                                               )
{
    _check();

    auto status = (*_interface.cudnnSetPooling2dDescriptor)(
        poolingDesc,
        mode,
        windowHeight,
        windowWidth,
        verticalPadding,
        horizontalPadding,
        verticalStride,
        horizontalStride
        );

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetPooling2dDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyPoolingDescriptor)(poolingDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyPoolingDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                             const cudnnTensorDescriptor_t inputTensorDesc,
                                             int* outN,
                                             int* outC,
                                             int* outH,
                                             int* outW)
{
    _check();

    auto status = (*_interface.cudnnGetPooling2dForwardOutputDim)(poolingDesc,
        inputTensorDesc,
        outN,
        outC,
        outH,
        outW);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetPooling2dForwardOutputDim failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(const cudnnTensorDescriptor_t      srcDesc,
                                            const cudnnFilterDescriptor_t      filterDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t      destDesc,
                                            cudnnConvolutionFwdPreference_t    preference,
                                            size_t                             memoryLimitInbytes,
                                            cudnnConvolutionFwdAlgo_t*         algo)
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionForwardAlgorithm)(_interface.getHandle(),
        srcDesc, filterDesc, convDesc,
        destDesc, preference, memoryLimitInbytes, algo);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetConvolutionForwardAlgorithm failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(const cudnnTensorDescriptor_t srcDesc,
                                                const cudnnFilterDescriptor_t      filterDesc,
                                                const cudnnConvolutionDescriptor_t convDesc,
                                                const cudnnTensorDescriptor_t      destDesc,
                                                cudnnConvolutionFwdAlgo_t          algo,
                                                size_t*                            sizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionForwardWorkspaceSize)(_interface.getHandle(),
        srcDesc, filterDesc, convDesc, destDesc, algo, sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetConvolutionForwardWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }

}


void CudnnLibrary::cudnnConvolutionForward(const void* alpha,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const void*                        srcData,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const void*                        filterData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                cudnnConvolutionFwdAlgo_t          algo,
                                void*                              workSpace,
                                size_t                             workSpaceSizeInBytes,
                                const void*                        beta,
                                const cudnnTensorDescriptor_t      destDesc,
                                void*                              destData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionForward)(_interface.getHandle(),
        alpha, srcDesc, srcData, filterDesc, filterData, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionForward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetConvolutionBackwardDataAlgorithm(
                            const cudnnFilterDescriptor_t       wDesc,
                            const cudnnTensorDescriptor_t       dyDesc,
                            const cudnnConvolutionDescriptor_t  convDesc,
                            const cudnnTensorDescriptor_t       dxDesc,
                            cudnnConvolutionBwdDataPreference_t preference,
                            size_t                              memoryLimitInBytes,
                            cudnnConvolutionBwdDataAlgo_t*      algo )
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionBackwardDataAlgorithm)(
        _interface.getHandle(),
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        preference,
        memoryLimitInBytes,
        algo);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardDataAlgorithm failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionBackwardDataWorkspaceSize(
                            const cudnnFilterDescriptor_t       wDesc,
                            const cudnnTensorDescriptor_t       dyDesc,
                            const cudnnConvolutionDescriptor_t  convDesc,
                            const cudnnTensorDescriptor_t       dxDesc,
                            cudnnConvolutionBwdDataAlgo_t       algo,
                            size_t*                             sizeInBytes )
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionBackwardDataWorkspaceSize)(
        _interface.getHandle(),
        wDesc,
        dyDesc,
        convDesc,
        dxDesc,
        algo,
        sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardDataWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnConvolutionBackwardData(const void*        alpha,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const void*                        filterData,
                                const cudnnTensorDescriptor_t      diffDesc,
                                const void*                        diffData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                cudnnConvolutionBwdDataAlgo_t      algo,
                                void*                              workSpace,
                                size_t                             workSpaceSizeInBytes,
                                const void*                        beta,
                                const cudnnTensorDescriptor_t      gradDesc,
                                void*                              gradData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionBackwardData)(_interface.getHandle(),
        alpha, filterDesc, filterData, diffDesc, diffData, convDesc,
        algo, workSpace, workSpaceSizeInBytes,
        beta, gradDesc, gradData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardData failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetConvolutionBackwardFilterAlgorithm(
                        const cudnnTensorDescriptor_t         xDesc,
                        const cudnnTensorDescriptor_t         dyDesc,
                        const cudnnConvolutionDescriptor_t    convDesc,
                        const cudnnFilterDescriptor_t         dwDesc,
                        cudnnConvolutionBwdFilterPreference_t preference,
                        size_t                                memoryLimitInBytes,
                        cudnnConvolutionBwdFilterAlgo_t*      algo )
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionBackwardFilterAlgorithm)(
        _interface.getHandle(),
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        preference,
        memoryLimitInBytes,
        algo);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardFilterAlgorithm failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        const cudnnTensorDescriptor_t       xDesc,
                        const cudnnTensorDescriptor_t       dyDesc,
                        const cudnnConvolutionDescriptor_t  convDesc,
                        const cudnnFilterDescriptor_t       gradDesc,
                        cudnnConvolutionBwdFilterAlgo_t     algo,
                        size_t*                             sizeInBytes )
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionBackwardFilterWorkspaceSize)(
        _interface.getHandle(),
        xDesc,
        dyDesc,
        convDesc,
        gradDesc,
        algo,
        sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardFilterWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnConvolutionBackwardFilter(const void*      alpha,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const void*                        srcData,
                                const cudnnTensorDescriptor_t      diffDesc,
                                const void*                        diffData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                cudnnConvolutionBwdFilterAlgo_t    algo,
                                void*                              workSpace,
                                size_t                             workSpaceSizeInBytes,
                                const void*                        beta,
                                const cudnnFilterDescriptor_t      gradDesc,
                                void*                              gradData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionBackwardFilter)(_interface.getHandle(),
        alpha, srcDesc, srcData, diffDesc, diffData, convDesc,
        algo, workSpace, workSpaceSizeInBytes,
        beta, gradDesc, gradData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardFilter failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnPoolingForward(const cudnnPoolingDescriptor_t   poolingDesc,
                                       const void*                      alpha,
                                       const cudnnTensorDescriptor_t    srcDesc,
                                       const void*                      srcData,
                                       const void*                      beta,
                                       const cudnnTensorDescriptor_t    destDesc,
                                       void*                            destData
                                             )
{
    _check();

    auto status = (*_interface.cudnnPoolingForward)(_interface.getHandle(),
        poolingDesc,
        alpha, srcDesc, srcData,
        beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnPoolingForward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnPoolingBackward(const cudnnPoolingDescriptor_t  poolingDesc,
                                        const void*                     alpha,
                                        const cudnnTensorDescriptor_t   srcDesc,
                                        const void*                     srcData,
                                        const cudnnTensorDescriptor_t   srcDiffDesc,
                                        const void*                     srcDiffData,
                                        const cudnnTensorDescriptor_t   destDesc,
                                        const void*                     destData,
                                        const void*                     beta,
                                        const cudnnTensorDescriptor_t   destDiffDesc,
                                        void*                           destDiffData
                                              )
{
    _check();

    auto status = (*_interface.cudnnPoolingBackward)(_interface.getHandle(),
        poolingDesc,
        alpha,
        srcDesc,
        srcData,
        srcDiffDesc,
        srcDiffData,
        destDesc,
        destData,
        beta,
        destDiffDesc,
        destDiffData
        );

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnPoolingBackward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateDropoutDescriptor)(dropoutDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateDropoutDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyDropoutDescriptor)(dropoutDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyDropoutDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                             float dropout,
                                             void* states,
                                             size_t stateSizeInBytes,
                                             unsigned long long seed)
{
    _check();

    auto status = (*_interface.cudnnSetDropoutDescriptor)(dropoutDesc,
                                                          _interface.getHandle(),
                                                          dropout,
                                                          states,
                                                          stateSizeInBytes,
                                                          seed);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetDropoutDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateRNNDescriptor)(rnnDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateRNNDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyRNNDescriptor)(rnnDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyRNNDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetRNNDescriptor(cudnnRNNDescriptor_t rnnDesc,
                                         int hiddenSize,
                                         int numLayers,
                                         cudnnDropoutDescriptor_t dropoutDesc,
                                         cudnnRNNInputMode_t inputMode,
                                         cudnnDirectionMode_t direction,
                                         cudnnRNNMode_t mode,
                                         cudnnDataType_t dataType)
{
    _check();

    auto status = (*_interface.cudnnSetRNNDescriptor)(rnnDesc,
                                                      hiddenSize,
                                                      numLayers,
                                                      dropoutDesc,
                                                      inputMode,
                                                      direction,
                                                      mode,
                                                      dataType);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetRNNDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetRNNWorkspaceSize(const cudnnRNNDescriptor_t rnnDesc,
                                            int seqLength,
                                            const cudnnTensorDescriptor_t* xDesc,
                                            size_t* sizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnGetRNNWorkspaceSize)(_interface.getHandle(),
                                                         rnnDesc,
                                                         seqLength,
                                                         xDesc,
                                                         sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetRNNWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetRNNTrainingReserveSize(const cudnnRNNDescriptor_t rnnDesc,
                                                  const int seqLength,
                                                  const cudnnTensorDescriptor_t* xDesc,
                                                  size_t* sizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnGetRNNTrainingReserveSize)(_interface.getHandle(),
                                                               rnnDesc,
                                                               seqLength,
                                                               xDesc,
                                                               sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetRNNTrainingReserveSize failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetRNNParamsSize(const cudnnRNNDescriptor_t rnnDesc,
                                         const cudnnTensorDescriptor_t xDesc,
                                         size_t* sizeInBytes,
                                         cudnnDataType_t dataType)
{
    _check();

    auto status = (*_interface.cudnnGetRNNParamsSize)(_interface.getHandle(),
                                                      rnnDesc,
                                                      xDesc,
                                                      sizeInBytes,
                                                      dataType);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetRNNParamsSize failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetRNNLinLayerMatrixParams(const cudnnRNNDescriptor_t rnnDesc,
                                                   const int layer,
                                                   const cudnnTensorDescriptor_t xDesc,
                                                   const cudnnFilterDescriptor_t wDesc,
                                                   const void* w,
                                                   const int linLayerID,
                                                   cudnnFilterDescriptor_t linLayerMatDesc,
                                                   void** linLayerMat)
{
    _check();

    auto status = (*_interface.cudnnGetRNNLinLayerMatrixParams)(_interface.getHandle(),
                                                                rnnDesc,
                                                                layer,
                                                                xDesc,
                                                                wDesc,
                                                                w,
                                                                linLayerID,
                                                                linLayerMatDesc,
                                                                linLayerMat);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetRNNLinLayerMatrixParams failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetRNNLinLayerBiasParams(const cudnnRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 const cudnnTensorDescriptor_t xDesc,
                                                 const cudnnFilterDescriptor_t wDesc,
                                                 const void* w,
                                                 const int linLayerID,
                                                 cudnnFilterDescriptor_t linLayerBiasDesc,
                                                 void** linLayerBias)
{
    _check();

    auto status = (*_interface.cudnnGetRNNLinLayerBiasParams)(_interface.getHandle(),
                                                              rnnDesc,
                                                              layer,
                                                              xDesc,
                                                              wDesc,
                                                              w,
                                                              linLayerID,
                                                              linLayerBiasDesc,
                                                              linLayerBias);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetRNNLinLayerBiasParams failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnRNNForwardTraining(const cudnnRNNDescriptor_t rnnDesc,
                                           int seqLength,
                                           const cudnnTensorDescriptor_t* xDesc,
                                           const void* x,
                                           const cudnnTensorDescriptor_t hxDesc,
                                           const void* hx,
                                           const cudnnTensorDescriptor_t cxDesc,
                                           const void* cx,
                                           const cudnnFilterDescriptor_t wDesc,
                                           const void* w,
                                           const cudnnTensorDescriptor_t* yDesc,
                                           void* y,
                                           const cudnnTensorDescriptor_t hyDesc,
                                           void* hy,
                                           const cudnnTensorDescriptor_t cyDesc,
                                           void* cy,
                                           void* workspace,
                                           size_t workSpaceSizeInBytes,
                                           void* reserveSpace,
                                           size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnRNNForwardTraining)(_interface.getHandle(),
                                                        rnnDesc,
                                                        seqLength,
                                                        xDesc,
                                                        x,
                                                        hxDesc,
                                                        hx,
                                                        cxDesc,
                                                        cx,
                                                        wDesc,
                                                        w,
                                                        yDesc,
                                                        y,
                                                        hyDesc,
                                                        hy,
                                                        cyDesc,
                                                        cy,
                                                        workspace,
                                                        workSpaceSizeInBytes,
                                                        reserveSpace,
                                                        reserveSpaceSizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnRNNForwardTraining failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnRNNBackwardData(const cudnnRNNDescriptor_t rnnDesc,
                                 const int seqLength,
                                 const cudnnTensorDescriptor_t* yDesc,
                                 const void* y,
                                 const cudnnTensorDescriptor_t* dyDesc,
                                 const void* dy,
                                 const cudnnTensorDescriptor_t dhyDesc,
                                 const void* dhy,
                                 const cudnnTensorDescriptor_t dcyDesc,
                                 const void* dcy,
                                 const cudnnFilterDescriptor_t wDesc,
                                 const void* w,
                                 const cudnnTensorDescriptor_t hxDesc,
                                 const void* hx,
                                 const cudnnTensorDescriptor_t cxDesc,
                                 const void* cx,
                                 const cudnnTensorDescriptor_t* dxDesc,
                                 void* dx,
                                 const cudnnTensorDescriptor_t dhxDesc,
                                 void* dhx,
                                 const cudnnTensorDescriptor_t dcxDesc,
                                 void* dcx,
                                 void* workspace,
                                 size_t workSpaceSizeInBytes,
                                 const void* reserveSpace,
                                 size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnRNNBackwardData)(_interface.getHandle(),
                                                     rnnDesc,
                                                     seqLength,
                                                     yDesc,
                                                     y,
                                                     dyDesc,
                                                     dy,
                                                     dhyDesc,
                                                     dhy,
                                                     dcyDesc,
                                                     dcy,
                                                     wDesc,
                                                     w,
                                                     hxDesc,
                                                     hx,
                                                     cxDesc,
                                                     cx,
                                                     dxDesc,
                                                     dx,
                                                     dhxDesc,
                                                     dhx,
                                                     dcxDesc,
                                                     dcx,
                                                     workspace,
                                                     workSpaceSizeInBytes,
                                                     reserveSpace,
                                                     reserveSpaceSizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnRNNBackwardData failed: " +
            _interface.getErrorString(status));
    }
}


void CudnnLibrary::cudnnRNNBackwardWeights(const cudnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const cudnnTensorDescriptor_t* xDesc,
                                           const void* x,
                                           const cudnnTensorDescriptor_t hxDesc,
                                           const void* hx,
                                           const cudnnTensorDescriptor_t* yDesc,
                                           const void* y,
                                           const void* workspace,
                                           size_t workSpaceSizeInBytes,
                                           const cudnnFilterDescriptor_t dwDesc,
                                           void* dw,
                                           const void* reserveSpace,
                                           size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnRNNBackwardWeights)(_interface.getHandle(),
                                                        rnnDesc,
                                                        seqLength,
                                                        xDesc,
                                                        x,
                                                        hxDesc,
                                                        hx,
                                                        yDesc,
                                                        y,
                                                        workspace,
                                                        workSpaceSizeInBytes,
                                                        dwDesc,
                                                        dw,
                                                        reserveSpace,
                                                        reserveSpaceSizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnRNNBackwardWeights failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call CUDNN function when "
            "the library is not loaded. Loading library failed, consider "
            "installing CUDNN.");
    }
}

static void checkFunction(void* pointer, const std::string& name)
{
    if(pointer == nullptr)
    {
        throw std::runtime_error("Failed to load function '" + name +
            "' from dynamic library.");
    }
}

CudnnLibrary::Interface::Interface() : _library(nullptr), _failed(false), _handle(nullptr)
{

}

CudnnLibrary::Interface::~Interface()
{
    unload();
}

void CudnnLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    const char* libraryName = "libcudnn.dylib";
    #else
    const char* libraryName = "libcudnn.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CudnnLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("Cudnnlibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {
        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(cudnnSetStream);

        DynLink(cudnnGetErrorString);
        DynLink(cudnnCreate);
        DynLink(cudnnDestroy);

        DynLink(cudnnCreateTensorDescriptor);
        DynLink(cudnnSetTensor4dDescriptor);
        DynLink(cudnnSetTensorNdDescriptor);
        DynLink(cudnnGetTensorNdDescriptor);
        DynLink(cudnnDestroyTensorDescriptor);
        DynLink(cudnnTransformTensor);

        DynLink(cudnnCreateFilterDescriptor);
        DynLink(cudnnSetFilter4dDescriptor);
        DynLink(cudnnSetFilterNdDescriptor);
        DynLink(cudnnGetFilterNdDescriptor);
        DynLink(cudnnDestroyFilterDescriptor);

        DynLink(cudnnCreateConvolutionDescriptor);
        DynLink(cudnnSetConvolution2dDescriptor);
        DynLink(cudnnDestroyConvolutionDescriptor);

        DynLink(cudnnCreatePoolingDescriptor);
        DynLink(cudnnSetPooling2dDescriptor);
        DynLink(cudnnDestroyPoolingDescriptor);
        DynLink(cudnnGetPooling2dForwardOutputDim);

        DynLink(cudnnGetConvolutionForwardAlgorithm);
        DynLink(cudnnGetConvolutionForwardWorkspaceSize);
        DynLink(cudnnConvolutionForward);

        DynLink(cudnnGetConvolutionBackwardDataAlgorithm);
        DynLink(cudnnGetConvolutionBackwardDataWorkspaceSize);
        DynLink(cudnnConvolutionBackwardData);

        DynLink(cudnnGetConvolutionBackwardFilterAlgorithm);
        DynLink(cudnnGetConvolutionBackwardFilterWorkspaceSize);
        DynLink(cudnnConvolutionBackwardFilter);

        DynLink(cudnnPoolingForward);
        DynLink(cudnnPoolingBackward);

        DynLink(cudnnCreateDropoutDescriptor);
        DynLink(cudnnDestroyDropoutDescriptor);
        DynLink(cudnnSetDropoutDescriptor);

        DynLink(cudnnCreateRNNDescriptor);
        DynLink(cudnnDestroyRNNDescriptor);

        DynLink(cudnnSetRNNDescriptor);
        DynLink(cudnnGetRNNWorkspaceSize);

        DynLink(cudnnGetRNNTrainingReserveSize);
        DynLink(cudnnGetRNNParamsSize);
        DynLink(cudnnGetRNNLinLayerMatrixParams);
        DynLink(cudnnGetRNNLinLayerBiasParams);

        DynLink(cudnnRNNForwardTraining);

        DynLink(cudnnRNNBackwardData);
        DynLink(cudnnRNNBackwardWeights);

        #undef DynLink

        auto status = (*cudnnCreate)(&_handle);

        if(status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("cudnnCreate failed: " + getErrorString(status));
        }

        util::log("Cudnnlibrary") << " Loaded library '" << libraryName
            << "' successfully\n";
    }
    catch(...)
    {
        unload();
        throw;
    }
}

bool CudnnLibrary::Interface::loaded() const
{
    return !_failed && (_library != nullptr);
}

void CudnnLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

CudnnLibrary::cudnnHandle_t CudnnLibrary::Interface::getHandle()
{
    return _handle;
}

std::string CudnnLibrary::Interface::getErrorString(cudnnStatus_t status)
{
    _check();

    return (*cudnnGetErrorString)(status);
}

CudnnLibrary::Interface CudnnLibrary::_interface;

}

}




