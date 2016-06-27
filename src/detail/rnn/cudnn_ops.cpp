
// PRNN Includes
#include <prnn/detail/rnn/cudnn_ops.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>

#include <prnn/detail/matrix/cudnn_library.h>
#include <prnn/detail/matrix/cudnn_descriptors.h>
#include <prnn/detail/matrix/matrix_view.h>

#include <prnn/detail/util/memory.h>

namespace prnn
{

namespace rnn
{

using CudnnLibrary = prnn::matrix::CudnnLibrary;
using CudnnRNNDescriptor = prnn::matrix::CudnnRNNDescriptor;
using CudnnFilterDescriptor = prnn::matrix::CudnnFilterDescriptor;
using CudnnFilterConstViewDescriptor = prnn::matrix::CudnnFilterConstViewDescriptor;
using CudnnFilterViewDescriptor = prnn::matrix::CudnnFilterViewDescriptor;
using CudnnTensorDescriptor = prnn::matrix::CudnnTensorDescriptor;
using CudnnTensorViewDescriptor = prnn::matrix::CudnnTensorViewDescriptor;
using CudnnTensorConstViewDescriptor = prnn::matrix::CudnnTensorConstViewDescriptor;

CudnnLibrary::cudnnDirectionMode_t convertDirection(const prnn::RecurrentLayerDirection direction)
{
    switch(direction)
    {
    case RECURRENT_FORWARD:
    {
        return CudnnLibrary::CUDNN_UNIDIRECTIONAL;
    }
    case RECURRENT_BIDIRECTIONAL:
    {
        return CudnnLibrary::CUDNN_BIDIRECTIONAL;
    }
    case RECURRENT_REVERSE:
    {
        throw std::invalid_argument("No support for reverse RNN in cudnn.");
    }
    }

    assert(false);
}

CudnnLibrary::cudnnRNNMode_t convertLayerType(const RecurrentLayerType& type)
{
    switch(type)
    {
    case RECURRENT_SIMPLE_TYPE:
    {
        return CudnnLibrary::CUDNN_RNN_RELU;
    }
    case RECURRENT_GRU_TYPE:
    {
        return CudnnLibrary::CUDNN_GRU;
    }
    case RECURRENT_LSTM_TYPE:
    {
        return CudnnLibrary::CUDNN_LSTM;
    }
    }

    assert(false);
}

CudnnLibrary::cudnnRNNInputMode_t convertInputMode(const RecurrentLayerInputMode& mode)
{
    switch(mode)
    {
    case RECURRENT_LINEAR_INPUT:
    {
        return CudnnLibrary::CUDNN_LINEAR_INPUT;
    }
    case RECURRENT_SKIP_INPUT:
    {
        return CudnnLibrary::CUDNN_SKIP_INPUT;
    }
    }

    assert(false);
}

CudnnLibrary::cudnnDataType_t convertPrecision(const matrix::Precision& precision)
{
    if(precision == matrix::SinglePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_FLOAT;
    }
    else if(precision == matrix::DoublePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_DOUBLE;
    }
    else if(precision == matrix::HalfPrecision())
    {
        return CudnnLibrary::CUDNN_DATA_HALF;
    }

    assert(false);
}

std::unique_ptr<CudnnRNNDescriptor> createRnnDescriptor(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return std::make_unique<CudnnRNNDescriptor>(handle.layerSize, handle.timesteps, handle.layers,
        convertDirection(handle.direction), convertLayerType(handle.layerType),
        convertInputMode(handle.inputMode), convertPrecision(precision));
}

std::unique_ptr<CudnnFilterConstViewDescriptor> getFilterDescriptor(const matrix::ConstDynamicView& view)
{
    return std::make_unique<CudnnFilterConstViewDescriptor>(view.data<void>(), view.size(),
        view.precision());
}

std::unique_ptr<CudnnFilterViewDescriptor> getFilterDescriptor(const matrix::DynamicView& view)
{
    return std::make_unique<CudnnFilterViewDescriptor>(view.data<void>(), view.size(),
        view.precision());
}

std::unique_ptr<CudnnTensorViewDescriptor> unpack(const matrix::DynamicView& activations, size_t index)
{
    if(activations.size().size() > 3 && index < activations.size()[3])
    {
        matrix::Dimension begin({0,0,0,index});
        matrix::Dimension end = activations.size();

        end[3] = index + 1;

        auto pack = slice(activations, begin, end);

        return std::make_unique<CudnnTensorViewDescriptor>(pack.data<void>(),
            pack.size(), pack.precision());
    }

    return std::make_unique<CudnnTensorViewDescriptor>();
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpack(const matrix::ConstDynamicView& activations,
    size_t index)
{
    if(activations.size().size() > 3 && index < activations.size()[3])
    {
        matrix::Dimension begin({0,0,0,index});
        matrix::Dimension end = activations.size();

        end[3] = index + 1;

        auto pack = slice(activations, begin, end);

        return std::make_unique<CudnnTensorConstViewDescriptor>(pack.data<void>(), pack.size(),
            pack.precision());
    }

    return std::make_unique<CudnnTensorConstViewDescriptor>();
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackX(const matrix::DynamicView& activations)
{
    return unpack(activations, 0);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackX(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 0);
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackHX(const matrix::DynamicView& activations)
{
    return unpack(activations, 1);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackHX(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 1);
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackCX(const matrix::DynamicView& activations)
{
    return unpack(activations, 2);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackCX(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 2);
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackY(const matrix::DynamicView& activations)
{
    return unpack(activations, 3);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackY(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 3);
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackHY(const matrix::DynamicView& activations)
{
    return unpack(activations, 4);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackHY(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 4);
}

std::unique_ptr<CudnnTensorViewDescriptor> unpackCY(const matrix::DynamicView& activations)
{
    return unpack(activations, 4);
}

std::unique_ptr<CudnnTensorConstViewDescriptor> unpackCY(
    const matrix::ConstDynamicView& activations)
{
    return unpack(activations, 4);
}

void cudnnForwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, weights.precision());

    auto xDescriptor  = unpackX(activations);
    auto hxDescriptor = unpackHX(activations);
    auto cxDescriptor = unpackCX(activations);

    auto yDescriptor  = unpackY(activations);
    auto hyDescriptor = unpackHY(activations);
    auto cyDescriptor = unpackCY(activations);

    auto weightsDescriptor = getFilterDescriptor(weights);

    CudnnLibrary::cudnnRNNForward(rnnDescriptor->descriptor(),
                                  &xDescriptor->descriptor(),
                                  xDescriptor->data(),
                                  hxDescriptor->descriptor(),
                                  hxDescriptor->data(),
                                  cxDescriptor->descriptor(),
                                  cxDescriptor->data(),
                                  weightsDescriptor->descriptor(),
                                  weightsDescriptor->data(),
                                  &yDescriptor->descriptor(),
                                  yDescriptor->data(),
                                  hyDescriptor->descriptor(),
                                  hyDescriptor->data(),
                                  cyDescriptor->descriptor(),
                                  cyDescriptor->data(),
                                  scratch.data<void>(),
                                  scratch.elements() * scratch.precision().size(),
                                  reserve.data<void>(),
                                  reserve.elements() * reserve.precision().size());
}

void cudnnBackPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, weights.precision());

    auto yDescriptor  = unpackY(activations);
    auto dyDescriptor = unpackY(deltas);

    auto dhyDescriptor = unpackHY(deltas);
    auto dcyDescriptor = unpackCY(deltas);

    auto weightsDescriptor = getFilterDescriptor(weights);

    auto hxDescriptor  = unpackHX(activations);
    auto dhxDescriptor = unpackHX(deltas);

    auto cxDescriptor  = unpackCX(activations);
    auto dcxDescriptor = unpackCX(deltas);

    auto dxDescriptor = unpackX(deltas);

    CudnnLibrary::cudnnRNNBackwardData(rnnDescriptor->descriptor(),
                                       &yDescriptor->descriptor(),
                                       yDescriptor->data(),
                                       &dyDescriptor->descriptor(),
                                       dyDescriptor->data(),
                                       dhyDescriptor->descriptor(),
                                       dhyDescriptor->data(),
                                       dcyDescriptor->descriptor(),
                                       dcyDescriptor->data(),
                                       weightsDescriptor->descriptor(),
                                       weightsDescriptor->data(),
                                       hxDescriptor->descriptor(),
                                       hxDescriptor->data(),
                                       cxDescriptor->descriptor(),
                                       cxDescriptor->data(),
                                       &dxDescriptor->descriptor(),
                                       dxDescriptor->data(),
                                       dhxDescriptor->descriptor(),
                                       dhxDescriptor->data(),
                                       dcxDescriptor->descriptor(),
                                       dcxDescriptor->data(),
                                       scratch.data<void>(),
                                       scratch.elements() * scratch.precision().size(),
                                       reserve.data<void>(),
                                       reserve.elements() * reserve.precision().size());

}

void cudnnBackPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& deltas,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, dWeights.precision());

    auto xDescriptor  = unpackX(outputActivations);
    auto hxDescriptor = unpackHX(outputActivations);
    auto yDescriptor  = unpackY(outputActivations);

    auto weightsDescriptor = getFilterDescriptor(dWeights);

    CudnnLibrary::cudnnRNNBackwardWeights(rnnDescriptor->descriptor(),
                                          &xDescriptor->descriptor(),
                                          xDescriptor->data(),
                                          hxDescriptor->descriptor(),
                                          hxDescriptor->data(),
                                          &yDescriptor->descriptor(),
                                          yDescriptor->data(),
                                          scratch.data<void>(),
                                          scratch.elements() * scratch.precision().size(),
                                          weightsDescriptor->descriptor(),
                                          weightsDescriptor->data(),
                                          reserve.data<void>(),
                                          reserve.elements() * reserve.precision().size());

}

}

}



