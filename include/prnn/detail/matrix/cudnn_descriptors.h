/*  \file   CudnnDescriptors.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the cudnn descriptor C++ wrappers.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <vector>

// Forward Declarations
namespace prnn { namespace matrix { class Matrix;     } }
namespace prnn { namespace matrix { class Dimension;  } }
namespace prnn { namespace matrix { class Precision;  } }
namespace prnn { namespace matrix { class Allocation; } }

typedef struct cudnnTensorStruct*      cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct*      cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*     cudnnPoolingDescriptor_t;
typedef struct cudnnRNNStruct*         cudnnRNNDescriptor_t;
typedef struct cudnnDropoutStruct*     cudnnDropoutDescriptor_t;

namespace prnn
{

namespace matrix
{

class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor(const Matrix& filter);

    ~CudnnFilterDescriptor();

public:
    cudnnFilterDescriptor_t descriptor() const;
    cudnnFilterDescriptor_t& descriptor();

public:
    void* data();

private:
    cudnnFilterDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _filter;

};

class CudnnFilterConstViewDescriptor
{
public:
    CudnnFilterConstViewDescriptor(const void* data, const Dimension& size,
        const Precision& precision);

    ~CudnnFilterConstViewDescriptor();

public:
    cudnnFilterDescriptor_t descriptor() const;
    cudnnFilterDescriptor_t& descriptor();

public:
    Dimension getDimensions() const;

public:
    const void* data() const;

public:
    std::string toString() const;

public:
    CudnnFilterConstViewDescriptor& operator=(const CudnnFilterConstViewDescriptor&) = delete;
    CudnnFilterConstViewDescriptor(const CudnnFilterConstViewDescriptor&) = delete;

private:
    cudnnFilterDescriptor_t _descriptor;

private:
    const void* _data;
};

class CudnnFilterViewDescriptor
{
public:
    CudnnFilterViewDescriptor(void* data, const Dimension& size, const Precision& precision);

    ~CudnnFilterViewDescriptor();

public:
    cudnnFilterDescriptor_t descriptor() const;
    cudnnFilterDescriptor_t& descriptor();

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

public:
    void* data() const;

public:
    CudnnFilterViewDescriptor& operator=(const CudnnFilterViewDescriptor&) = delete;
    CudnnFilterViewDescriptor(const CudnnFilterViewDescriptor&) = delete;

private:
    cudnnFilterDescriptor_t _descriptor;

private:
    void* _data;
};

class CudnnTensorDescriptor
{
public:
    CudnnTensorDescriptor(const Matrix& tensor);
    CudnnTensorDescriptor(const Dimension& size);
    ~CudnnTensorDescriptor();

public:
    cudnnTensorDescriptor_t descriptor() const;
    cudnnTensorDescriptor_t& descriptor();

public:
    void* data();
    size_t bytes() const;

private:
    cudnnTensorDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _tensor;

};

class CudnnTensorViewDescriptor
{
public:
    CudnnTensorViewDescriptor(void* data, const Dimension& size, const Dimension& strides,
        const Precision& precision);
    CudnnTensorViewDescriptor();
    ~CudnnTensorViewDescriptor();

public:
    cudnnTensorDescriptor_t descriptor() const;
    cudnnTensorDescriptor_t& descriptor();

public:
    CudnnTensorViewDescriptor& operator=(const CudnnTensorViewDescriptor&) = delete;
    CudnnTensorViewDescriptor(const CudnnTensorViewDescriptor&) = delete;

public:
    void* data() const;

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

private:
    cudnnTensorDescriptor_t _descriptor;

private:
    void* _data;

};

class CudnnTensorViewDescriptorArray
{
public:
    CudnnTensorViewDescriptorArray(void* data, const Dimension& size, const Dimension& strides,
        size_t timesteps, const Precision& precision);
    ~CudnnTensorViewDescriptorArray();

public:
    cudnnTensorDescriptor_t* descriptors();

public:
    void* data() const;

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

public:
    CudnnTensorViewDescriptorArray& operator=(const CudnnTensorViewDescriptorArray&) = delete;
    CudnnTensorViewDescriptorArray(const CudnnTensorViewDescriptorArray&) = delete;

private:
    std::vector<cudnnTensorDescriptor_t> _descriptors;

private:
    void* _data;

};

class CudnnTensorConstViewDescriptorArray
{
public:
    CudnnTensorConstViewDescriptorArray(const void* data, const Dimension& size,
        const Dimension& strides,
        size_t timesteps, const Precision& precision);
    ~CudnnTensorConstViewDescriptorArray();

public:
    cudnnTensorDescriptor_t* descriptors();

public:
    const void* data() const;

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

public:
    CudnnTensorConstViewDescriptorArray& operator=(const CudnnTensorConstViewDescriptorArray&) = delete;
    CudnnTensorConstViewDescriptorArray(const CudnnTensorConstViewDescriptorArray&) = delete;

private:
    std::vector<cudnnTensorDescriptor_t> _descriptors;

private:
    const void* _data;

};

class CudnnTensorConstViewDescriptor
{
public:
    CudnnTensorConstViewDescriptor(const void* data,
        const Dimension& size, const Dimension& strides, const Precision& precision);
    CudnnTensorConstViewDescriptor();
    ~CudnnTensorConstViewDescriptor();

public:
    cudnnTensorDescriptor_t descriptor() const;
    cudnnTensorDescriptor_t& descriptor();

public:
    const void* data() const;

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

public:
    CudnnTensorConstViewDescriptor& operator=(const CudnnTensorConstViewDescriptor&) = delete;
    CudnnTensorConstViewDescriptor(const CudnnTensorConstViewDescriptor&) = delete;

private:
    cudnnTensorDescriptor_t _descriptor;

private:
    const void* _data;

};

class CudnnScalar
{
public:
    CudnnScalar(double value, const Precision& p);
    ~CudnnScalar();

public:
    void* data();

private:
    double _doubleValue;
    float  _floatValue;

private:
    std::unique_ptr<Precision> _precision;

};

class CudnnForwardWorkspace
{
public:
    CudnnForwardWorkspace(const CudnnTensorDescriptor& source, const CudnnFilterDescriptor& filter,
        cudnnConvolutionDescriptor_t convolutionDescriptor, const CudnnTensorDescriptor& result);

public:
    int algorithm() const;
    void* data();
    size_t size() const;

private:
    int _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

class CudnnBackwardDataWorkspace
{
public:
    CudnnBackwardDataWorkspace(const CudnnFilterDescriptor& filter,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnTensorDescriptor& inputDeltas);

public:
    int algorithm() const;
    void* data();
    size_t size() const;

private:
    int _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

class CudnnBackwardFilterWorkspace
{
public:
    CudnnBackwardFilterWorkspace(const CudnnTensorDescriptor& source,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnFilterDescriptor& filterGradient
        );

public:
    int algorithm() const;
    void* data();
    size_t size() const;

private:
    int _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

class CudnnPooling2dDescriptor
{
public:
    CudnnPooling2dDescriptor(size_t inputW, size_t inputH, size_t padW, size_t padH,
        size_t poolW, size_t poolH);
    ~CudnnPooling2dDescriptor();

public:
    cudnnPoolingDescriptor_t descriptor() const;
    cudnnPoolingDescriptor_t& descriptor();

public:
    CudnnPooling2dDescriptor& operator=(const CudnnPooling2dDescriptor&) = delete;
    CudnnPooling2dDescriptor(const CudnnPooling2dDescriptor&) = delete;

private:
    cudnnPoolingDescriptor_t _descriptor;

};

class CudnnRNNDescriptor
{
public:
    CudnnRNNDescriptor(int hiddenSize, int numLayers, int inputMode,
        int direction, int mode, int dataType);
    ~CudnnRNNDescriptor();

public:
    cudnnRNNDescriptor_t descriptor() const;
    cudnnRNNDescriptor_t& descriptor();

public:
    CudnnRNNDescriptor& operator=(const CudnnRNNDescriptor&) = delete;
    CudnnRNNDescriptor(const CudnnRNNDescriptor&) = delete;

private:
    cudnnRNNDescriptor_t     _descriptor;
    cudnnDropoutDescriptor_t _dropoutDescriptor;


};

}

}

