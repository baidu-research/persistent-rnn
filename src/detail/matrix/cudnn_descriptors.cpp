/*  \file   CudnnDescriptors.cpp
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the cudnn descriptor C++ wrappers.
*/

// PRNN Library Includes
#include <prnn/detail/matrix/cudnn_descriptors.h>
#include <prnn/detail/matrix/cudnn_library.h>
#include <prnn/detail/matrix/precision.h>
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/dimension.h>
#include <prnn/detail/matrix/allocation.h>

#include <prnn/detail/util/memory.h>

// Standard Library Includes
#include <cassert>
#include <vector>

namespace prnn
{

namespace matrix
{

static CudnnLibrary::cudnnDataType_t getCudnnDataType(const Precision& precision)
{
    if(precision == DoublePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_DOUBLE;
    }

    assert(precision == SinglePrecision());

    return CudnnLibrary::CUDNN_DATA_FLOAT;
}

CudnnFilterDescriptor::CudnnFilterDescriptor(const Matrix& filter)
: _filter(std::make_unique<Matrix>(filter))
{
    CudnnLibrary::cudnnCreateFilterDescriptor(&_descriptor);

    int dim[4] = {1, 1, 1, 1};

    if(filter.size().size() >= 1)
    {
        dim[0] = filter.size()[0];
    }
    if(filter.size().size() >= 2)
    {
        dim[1] = filter.size()[1];
    }
    if(filter.size().size() >= 3)
    {
        dim[2] = filter.size()[2];
    }
    if(filter.size().size() >= 4)
    {
        dim[3] = filter.size()[3];
    }

    CudnnLibrary::cudnnSetFilter4dDescriptor(_descriptor,
        getCudnnDataType(filter.precision()),
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        dim[3],
        dim[2],
        dim[1],
        dim[0]);
}

CudnnFilterDescriptor::~CudnnFilterDescriptor()
{
    CudnnLibrary::cudnnDestroyFilterDescriptor(_descriptor);
}

cudnnFilterDescriptor_t CudnnFilterDescriptor::descriptor() const
{
    return _descriptor;
}

void* CudnnFilterDescriptor::data()
{
    return _filter->data();
}

CudnnFilterConstViewDescriptor::CudnnFilterConstViewDescriptor(const void* data,
    const Dimension& size, const Precision& precision)
: _descriptor(nullptr), _data(data)
{
    CudnnLibrary::cudnnCreateFilterDescriptor(&_descriptor);

    std::vector<int> sizes(size.begin(), size.end());

    CudnnLibrary::cudnnSetFilterNdDescriptor(_descriptor,
        getCudnnDataType(precision),
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        sizes.size(),
        sizes.data());
}

CudnnFilterConstViewDescriptor::~CudnnFilterConstViewDescriptor()
{
    CudnnLibrary::cudnnDestroyFilterDescriptor(_descriptor);
}

cudnnFilterDescriptor_t CudnnFilterConstViewDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnFilterDescriptor_t& CudnnFilterConstViewDescriptor::descriptor()
{
    return _descriptor;
}

Dimension CudnnFilterConstViewDescriptor::getDimensions() const
{
    int sizes[4];
    int dimensions = 0;

    CudnnLibrary::cudnnTensorFormat_t format;
    CudnnLibrary::cudnnDataType_t dataType;


    CudnnLibrary::cudnnGetFilterNdDescriptor(_descriptor,
                                             4,
                                             &dataType,
                                             &format,
                                             &dimensions,
                                             sizes);

    Dimension result;

    for(int i = 0; i < dimensions; ++i)
    {
        result.push_back(sizes[i]);
    }

    return result;
}

std::string CudnnFilterConstViewDescriptor::toString() const
{
    return getDimensions().toString();
}

const void* CudnnFilterConstViewDescriptor::data() const
{
    return _data;
}

CudnnFilterViewDescriptor::CudnnFilterViewDescriptor(void* data, const Dimension& size,
    const Precision& precision)
: _descriptor(nullptr), _data(data)
{
    CudnnLibrary::cudnnCreateFilterDescriptor(&_descriptor);

    std::vector<int> sizes(size.begin(), size.end());

    CudnnLibrary::cudnnSetFilterNdDescriptor(_descriptor,
        getCudnnDataType(precision),
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        sizes.size(),
        sizes.data());
}

CudnnFilterViewDescriptor::~CudnnFilterViewDescriptor()
{
    CudnnLibrary::cudnnDestroyFilterDescriptor(_descriptor);
}

cudnnFilterDescriptor_t CudnnFilterViewDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnFilterDescriptor_t& CudnnFilterViewDescriptor::descriptor()
{
    return _descriptor;
}

Dimension CudnnFilterViewDescriptor::getDimensions() const
{
    int sizes[8];
    int dimensions = 0;

    CudnnLibrary::cudnnTensorFormat_t format;
    CudnnLibrary::cudnnDataType_t dataType;


    CudnnLibrary::cudnnGetFilterNdDescriptor(_descriptor,
                                             8,
                                             &dataType,
                                             &format,
                                             &dimensions,
                                             sizes);

    Dimension result;

    for(int i = 0; i < dimensions; ++i)
    {
        result.push_back(sizes[i]);
    }

    return result;
}

std::string CudnnFilterViewDescriptor::toString() const
{
    return getDimensions().toString();
}

void* CudnnFilterViewDescriptor::data() const
{
    return _data;
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Matrix& tensor)
: _tensor(std::make_unique<Matrix>(tensor))
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetTensor4dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        getCudnnDataType(_tensor->precision()), // image data type
        _tensor->size()[3],        // number of inputs (batch size)
        _tensor->size()[2],        // number of input feature maps
        _tensor->size()[1],        // height of input section
        _tensor->size()[0]         // width of input section
    );
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Dimension& size)
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetTensor4dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        CudnnLibrary::CUDNN_DATA_FLOAT, // image data type
        size[3],        // number of inputs (batch size)
        size[2],        // number of input feature maps
        size[1],        // height of input section
        size[0]         // width of input section
    );
}

CudnnTensorDescriptor::~CudnnTensorDescriptor()
{
    CudnnLibrary::cudnnDestroyTensorDescriptor(_descriptor);
}

cudnnTensorDescriptor_t CudnnTensorDescriptor::descriptor() const
{
    return _descriptor;
}

void* CudnnTensorDescriptor::data()
{
    return _tensor->data();
}

size_t CudnnTensorDescriptor::bytes() const
{
    return _tensor->elements() * _tensor->precision().size();
}

static std::vector<int> getDimensions(const Dimension& size)
{
    return std::vector<int>(size.begin(), size.end());
}

static std::vector<int> getStrides(const Dimension& strides)
{
    return getDimensions(strides);
}

CudnnTensorViewDescriptor::CudnnTensorViewDescriptor(
    void* data, const Dimension& size, const Dimension& stride,
    const Precision& precision)
: _descriptor(nullptr), _data(data)
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    auto dimensions = prnn::matrix::getDimensions(size);
    auto strides    = getStrides(stride);

    CudnnLibrary::cudnnSetTensorNdDescriptor(_descriptor,
        getCudnnDataType(precision),
        size.size(),
        dimensions.data(),
        strides.data()
    );

}

CudnnTensorViewDescriptor::CudnnTensorViewDescriptor()
: _descriptor(nullptr), _data(nullptr)
{

}

CudnnTensorViewDescriptor::~CudnnTensorViewDescriptor()
{
    CudnnLibrary::cudnnDestroyTensorDescriptor(_descriptor);
}

cudnnTensorDescriptor_t CudnnTensorViewDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnTensorDescriptor_t& CudnnTensorViewDescriptor::descriptor()
{
    return _descriptor;
}

void* CudnnTensorViewDescriptor::data() const
{
    return _data;
}

Dimension CudnnTensorViewDescriptor::getDimensions() const
{
    int sizes[4];
    int strides[4];
    int dimensions = 0;

    CudnnLibrary::cudnnDataType_t dataType;

    CudnnLibrary::cudnnGetTensorNdDescriptor(_descriptor,
                                             4,
                                             &dataType,
                                             &dimensions,
                                             sizes,
                                             strides);

    Dimension result;

    for(int i = 0; i < dimensions; ++i)
    {
        result.push_back(sizes[i]);
    }

    return result;
}

std::string CudnnTensorViewDescriptor::toString() const
{
    return getDimensions().toString();
}

CudnnTensorViewDescriptorArray::CudnnTensorViewDescriptorArray(void* data, const Dimension& size,
    const Dimension& stride, size_t timesteps, const Precision& precision)
: _data(data)
{
    _descriptors.resize(timesteps);

    auto dimensions = prnn::matrix::getDimensions(size);
    auto strides    = getStrides(stride);

    for(size_t i = 0; i < timesteps; ++i)
    {
        CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptors[i]);
        CudnnLibrary::cudnnSetTensorNdDescriptor(_descriptors[i],
            getCudnnDataType(precision),
            size.size(),
            dimensions.data(),
            strides.data()
        );
    }
}

CudnnTensorViewDescriptorArray::~CudnnTensorViewDescriptorArray()
{
    for(auto descriptor : _descriptors)
    {
        CudnnLibrary::cudnnDestroyTensorDescriptor(descriptor);
    }
}

cudnnTensorDescriptor_t* CudnnTensorViewDescriptorArray::descriptors()
{
    return _descriptors.data();
}

Dimension CudnnTensorViewDescriptorArray::getDimensions() const
{
    int sizes[4];
    int strides[4];
    int dimensions = 0;

    CudnnLibrary::cudnnDataType_t dataType;

    CudnnLibrary::cudnnGetTensorNdDescriptor(_descriptors[0],
                                             4,
                                             &dataType,
                                             &dimensions,
                                             sizes,
                                             strides);

    Dimension result;

    for(int i = 0; i < dimensions; ++i)
    {
        result.push_back(sizes[i]);
    }

    return result;
}

std::string CudnnTensorViewDescriptorArray::toString() const
{
    return getDimensions().toString();
}

void* CudnnTensorViewDescriptorArray::data() const
{
    return _data;
}

CudnnTensorConstViewDescriptorArray::CudnnTensorConstViewDescriptorArray(const void* data,
    const Dimension& size, const Dimension& stride, size_t timesteps, const Precision& precision)
: _data(data)
{
    _descriptors.resize(timesteps);

    auto dimensions = prnn::matrix::getDimensions(size);
    auto strides    = getStrides(stride);

    for(size_t i = 0; i < timesteps; ++i)
    {
        CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptors[i]);
        CudnnLibrary::cudnnSetTensorNdDescriptor(_descriptors[i],
            getCudnnDataType(precision),
            size.size(),
            dimensions.data(),
            strides.data()
        );
    }
}

CudnnTensorConstViewDescriptorArray::~CudnnTensorConstViewDescriptorArray()
{
    for(auto descriptor : _descriptors)
    {
        CudnnLibrary::cudnnDestroyTensorDescriptor(descriptor);
    }
}

cudnnTensorDescriptor_t* CudnnTensorConstViewDescriptorArray::descriptors()
{
    return _descriptors.data();
}

const void* CudnnTensorConstViewDescriptorArray::data() const
{
    return _data;
}

Dimension CudnnTensorConstViewDescriptorArray::getDimensions() const
{
    int sizes[4];
    int strides[4];
    int dimensions = 0;

    CudnnLibrary::cudnnDataType_t dataType;

    CudnnLibrary::cudnnGetTensorNdDescriptor(_descriptors[0],
                                             4,
                                             &dataType,
                                             &dimensions,
                                             sizes,
                                             strides);

    Dimension result;

    for(int i = 0; i < dimensions; ++i)
    {
        result.push_back(sizes[i]);
    }

    return result;
}

std::string CudnnTensorConstViewDescriptorArray::toString() const
{
    return getDimensions().toString();
}

CudnnTensorConstViewDescriptor::CudnnTensorConstViewDescriptor(
    const void* data, const Dimension& size, const Dimension& stride, const Precision& precision)
: _descriptor(nullptr), _data(data)
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    auto dimensions = matrix::getDimensions(size);
    auto strides    = getStrides(stride);

    CudnnLibrary::cudnnSetTensorNdDescriptor(_descriptor,
        getCudnnDataType(precision),
        size.size(),
        dimensions.data(),
        strides.data()
    );

}

CudnnTensorConstViewDescriptor::CudnnTensorConstViewDescriptor()
: _descriptor(nullptr), _data(nullptr)
{

}

CudnnTensorConstViewDescriptor::~CudnnTensorConstViewDescriptor()
{
    CudnnLibrary::cudnnDestroyTensorDescriptor(_descriptor);
}

cudnnTensorDescriptor_t CudnnTensorConstViewDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnTensorDescriptor_t& CudnnTensorConstViewDescriptor::descriptor()
{
    return _descriptor;
}

const void* CudnnTensorConstViewDescriptor::data() const
{
    return _data;
}

CudnnScalar::CudnnScalar(double value, const Precision& p)
: _doubleValue(value), _floatValue(value), _precision(std::make_unique<Precision>(p))
{

}

CudnnScalar::~CudnnScalar()
{

}

void* CudnnScalar::data()
{
    if(*_precision == SinglePrecision())
    {
        return &_floatValue;
    }
    else
    {
        return &_doubleValue;
    }
}

CudnnForwardWorkspace::CudnnForwardWorkspace(const CudnnTensorDescriptor& source,
    const CudnnFilterDescriptor& filter,
    cudnnConvolutionDescriptor_t convolutionDescriptor, const CudnnTensorDescriptor& result)
{
    CudnnLibrary::cudnnConvolutionFwdAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(
        source.descriptor(),
        filter.descriptor(),
        convolutionDescriptor,
        result.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        source.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(
        source.descriptor(),
        filter.descriptor(),
        convolutionDescriptor,
        result.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnForwardWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnForwardWorkspace::data()
{
    return _data->data();
}

size_t CudnnForwardWorkspace::size() const
{
    return _data->size();
}

CudnnBackwardDataWorkspace::CudnnBackwardDataWorkspace(const CudnnFilterDescriptor& filter,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnTensorDescriptor& inputDeltas)
{
    CudnnLibrary::cudnnConvolutionBwdDataAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionBackwardDataAlgorithm(
        filter.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        inputDeltas.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        inputDeltas.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionBackwardDataWorkspaceSize(
        filter.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        inputDeltas.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnBackwardDataWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnBackwardDataWorkspace::data()
{
    return _data->data();
}

size_t CudnnBackwardDataWorkspace::size() const
{
    return _data->size();
}

CudnnBackwardFilterWorkspace::CudnnBackwardFilterWorkspace(const CudnnTensorDescriptor& input,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnFilterDescriptor& filterGradient)
{
    CudnnLibrary::cudnnConvolutionBwdFilterAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionBackwardFilterAlgorithm(
        input.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        filterGradient.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        input.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionBackwardFilterWorkspaceSize(
        input.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        filterGradient.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnBackwardFilterWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnBackwardFilterWorkspace::data()
{
    return _data->data();
}

size_t CudnnBackwardFilterWorkspace::size() const
{
    return _data->size();
}

CudnnPooling2dDescriptor::CudnnPooling2dDescriptor(size_t inputW, size_t inputH, size_t padW,
    size_t padH, size_t poolW, size_t poolH)
{
    CudnnLibrary::cudnnCreatePoolingDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetPooling2dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_POOLING_MAX,
        inputH,
        inputW,
        padH,
        padW,
        poolH,
        poolW);
}

CudnnPooling2dDescriptor::~CudnnPooling2dDescriptor()
{
    CudnnLibrary::cudnnDestroyPoolingDescriptor(_descriptor);
}

cudnnPoolingDescriptor_t CudnnPooling2dDescriptor::descriptor() const
{
    return _descriptor;
}

CudnnRNNDescriptor::CudnnRNNDescriptor(int hiddenSize, int numLayers, int inputMode,
    int direction, int mode, int dataType)
: _descriptor(nullptr), _dropoutDescriptor(nullptr)
{
    CudnnLibrary::cudnnCreateDropoutDescriptor(&_dropoutDescriptor);

    CudnnLibrary::cudnnSetDropoutDescriptor(_dropoutDescriptor,
                                            1.0f,
                                            nullptr,
                                            0,
                                            0);

    CudnnLibrary::cudnnCreateRNNDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetRNNDescriptor(_descriptor,
        hiddenSize,
        numLayers,
        _dropoutDescriptor,
        static_cast<CudnnLibrary::cudnnRNNInputMode_t>(inputMode),
        static_cast<CudnnLibrary::cudnnDirectionMode_t>(direction),
        static_cast<CudnnLibrary::cudnnRNNMode_t>(mode),
        static_cast<CudnnLibrary::cudnnDataType_t>(dataType));
}

CudnnRNNDescriptor::~CudnnRNNDescriptor()
{
    CudnnLibrary::cudnnDestroyRNNDescriptor(_descriptor);
    CudnnLibrary::cudnnDestroyDropoutDescriptor(_dropoutDescriptor);
}

cudnnRNNDescriptor_t CudnnRNNDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnRNNDescriptor_t& CudnnRNNDescriptor::descriptor()
{
    return _descriptor;
}

}

}

