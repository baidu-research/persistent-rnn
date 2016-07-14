
// PRNN Includes
#include <prnn/detail/rnn/cudnn_ops.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>

#include <prnn/detail/matrix/cudnn_library.h>
#include <prnn/detail/matrix/cudnn_descriptors.h>
#include <prnn/detail/matrix/matrix_view.h>

#include <prnn/detail/util/memory.h>
#include <prnn/detail/util/logger.h>

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
using CudnnTensorViewDescriptorArray = prnn::matrix::CudnnTensorViewDescriptorArray;
using CudnnTensorConstViewDescriptor = prnn::matrix::CudnnTensorConstViewDescriptor;
using CudnnTensorConstViewDescriptorArray = prnn::matrix::CudnnTensorConstViewDescriptorArray;

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
    return std::make_unique<CudnnRNNDescriptor>(handle.layerSize, handle.layers,
        convertInputMode(handle.inputMode), convertDirection(handle.direction),
        convertLayerType(handle.layerType), convertPrecision(precision));
}

std::unique_ptr<CudnnFilterConstViewDescriptor> getFilterDescriptor(
    const matrix::ConstDynamicView& view)
{
    return std::make_unique<CudnnFilterConstViewDescriptor>(view.data<void>(), view.size(),
        view.precision());
}

std::unique_ptr<CudnnFilterViewDescriptor> getFilterDescriptor(const matrix::DynamicView& view)
{
    return std::make_unique<CudnnFilterViewDescriptor>(view.data<void>(), view.size(),
        view.precision());
}

std::unique_ptr<CudnnTensorConstViewDescriptorArray> getActivationsDescriptors(
    const matrix::ConstDynamicView& activations)
{
    matrix::Dimension dimensions = {activations.size()[1],   activations.size()[0],   1};
    matrix::Dimension strides    = {activations.stride()[1], activations.stride()[0], 1};

    size_t timesteps = activations.size()[2];

    return std::make_unique<CudnnTensorConstViewDescriptorArray>(activations.data<void>(),
        dimensions, strides, timesteps, activations.precision());
}

std::unique_ptr<CudnnTensorViewDescriptorArray> getActivationsDescriptors(
    const matrix::DynamicView& activations)
{
    matrix::Dimension dimensions = {activations.size()[1],   activations.size()[0],   1};
    matrix::Dimension strides    = {activations.stride()[1], activations.stride()[0], 1};

    size_t timesteps = activations.size()[2];

    return std::make_unique<CudnnTensorViewDescriptorArray>(activations.data<void>(), dimensions,
        strides, timesteps, activations.precision());
}

std::unique_ptr<CudnnTensorViewDescriptor> getEmptyLayerInputDescriptor(
    const RecurrentOpsHandle& handle, const matrix::Precision& precision)
{
    matrix::Dimension dimensions = {handle.layers, handle.miniBatchSize, handle.layerSize};
    matrix::Dimension strides    = {handle.layerSize * handle.miniBatchSize, handle.layerSize, 1};

    return std::make_unique<CudnnTensorViewDescriptor>(nullptr, dimensions,
        strides, precision);
}

static std::string toString(RecurrentLayerBackend backend)
{
    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        return "cudnn backend";
    }
    else if(backend == RECURRENT_PERSISTENT_BACKEND)
    {
        return "persistent backend";
    }
    else
    {
        return "generic backend";
    }
}

void cudnnForwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, weights.precision());

    auto xDescriptor  = getActivationsDescriptors(inputActivations);
    auto hxDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());
    auto cxDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());

    auto yDescriptor  = getActivationsDescriptors(activations);
    auto hyDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());
    auto cyDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());

    auto weightsDescriptor = getFilterDescriptor(weights);

    prnn::util::log("CudnnOps") << "Running cudnnRNNForwardTraining\n";
    prnn::util::log("CudnnOps") << "Inputs         " << xDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Outputs        " << yDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "hx             " << hxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "cx             " << cxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "hy             " << hyDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "cy             " << cyDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Input Weights  " << weightsDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Handle         " << handle.toString() << "\n";
    prnn::util::log("CudnnOps") << "Backend        " << toString(getBackend(handle, inputActivations.precision())) << "\n";
    prnn::util::log("CudnnOps") << "Scratch size   " << scratch.elements() * scratch.precision().size() << "\n";
    prnn::util::log("CudnnOps") << "Reserve size   " << reserve.elements() * reserve.precision().size() << "\n";

    CudnnLibrary::cudnnSetStream(handle.stream);
    CudnnLibrary::cudnnRNNForwardTraining(rnnDescriptor->descriptor(),
                                          handle.timesteps,
                                          xDescriptor->descriptors(),
                                          xDescriptor->data(),
                                          hxDescriptor->descriptor(),
                                          hxDescriptor->data(),
                                          cxDescriptor->descriptor(),
                                          cxDescriptor->data(),
                                          weightsDescriptor->descriptor(),
                                          weightsDescriptor->data(),
                                          yDescriptor->descriptors(),
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
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& outputDeltas,
    const matrix::DynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, weights.precision());

    auto yDescriptor  = getActivationsDescriptors(outputActivations);
    auto dyDescriptor = getActivationsDescriptors(outputDeltas);

    auto dhyDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());
    auto dcyDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());

    auto weightsDescriptor = getFilterDescriptor(weights);

    auto hxDescriptor  = getEmptyLayerInputDescriptor(handle, weights.precision());
    auto dhxDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());

    auto cxDescriptor  = getEmptyLayerInputDescriptor(handle, weights.precision());
    auto dcxDescriptor = getEmptyLayerInputDescriptor(handle, weights.precision());

    auto dxDescriptor = getActivationsDescriptors(deltas);

    prnn::util::log("CudnnOps") << "Running cudnnRNNBackwardData\n";
    prnn::util::log("CudnnOps") << "y              " << yDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dy             " << dyDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dhy            " << dhyDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dcy            " << dcyDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Input Weights  " << weightsDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "hx             " << hxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dhx            " << dhxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "cx             " << cxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dcx            " << dcxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "dx             " << dxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Handle         " << handle.toString() << "\n";
    prnn::util::log("CudnnOps") << "Backend        " << toString(getBackend(handle, deltas.precision())) << "\n";
    prnn::util::log("CudnnOps") << "Scratch size   " << scratch.elements() * scratch.precision().size() << "\n";
    prnn::util::log("CudnnOps") << "Reserve size   " << reserve.elements() * reserve.precision().size() << "\n";

    CudnnLibrary::cudnnSetStream(handle.stream);

    CudnnLibrary::cudnnRNNBackwardData(rnnDescriptor->descriptor(),
                                       handle.timesteps,
                                       yDescriptor->descriptors(),
                                       yDescriptor->data(),
                                       dyDescriptor->descriptors(),
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
                                       dxDescriptor->descriptors(),
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
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto rnnDescriptor = createRnnDescriptor(handle, dWeights.precision());

    auto xDescriptor  = getActivationsDescriptors(inputActivations);
    auto hxDescriptor = getEmptyLayerInputDescriptor(handle, dWeights.precision());
    auto yDescriptor  = getActivationsDescriptors(outputActivations);

    auto weightsDescriptor = getFilterDescriptor(dWeights);

    prnn::util::log("CudnnOps") << "Running cudnnRNNBackwardData\n";
    prnn::util::log("CudnnOps") << "y              " << yDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Input Weights  " << weightsDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "hx             " << hxDescriptor->toString() << "\n";
    prnn::util::log("CudnnOps") << "Handle         " << handle.toString() << "\n";
    prnn::util::log("CudnnOps") << "Backend        " << toString(getBackend(handle, dWeights.precision())) << "\n";
    prnn::util::log("CudnnOps") << "Scratch size   " << scratch.elements() * scratch.precision().size() << "\n";
    prnn::util::log("CudnnOps") << "Reserve size   " << reserve.elements() * reserve.precision().size() << "\n";

    CudnnLibrary::cudnnSetStream(handle.stream);
    CudnnLibrary::cudnnRNNBackwardWeights(rnnDescriptor->descriptor(),
                                          handle.timesteps,
                                          xDescriptor->descriptors(),
                                          xDescriptor->data(),
                                          hxDescriptor->descriptor(),
                                          hxDescriptor->data(),
                                          yDescriptor->descriptors(),
                                          yDescriptor->data(),
                                          scratch.data<void>(),
                                          scratch.elements() * scratch.precision().size(),
                                          weightsDescriptor->descriptor(),
                                          weightsDescriptor->data(),
                                          reserve.data<void>(),
                                          reserve.elements() * reserve.precision().size());

}

size_t cudnnGetReserveSize(
    const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto rnnDescriptor = createRnnDescriptor(handle, precision);

    CudnnTensorViewDescriptorArray xDescriptor(nullptr, {handle.miniBatchSize,
        handle.layerSize, 1}, {handle.layerSize, 1, 1}, handle.timesteps, precision);

    size_t size = 0;

    CudnnLibrary::cudnnGetRNNTrainingReserveSize(rnnDescriptor->descriptor(),
        handle.timesteps,
        xDescriptor.descriptors(), &size);

    return size;

}

size_t cudnnGetScratchSize(
    const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto rnnDescriptor = createRnnDescriptor(handle, precision);

    CudnnTensorViewDescriptorArray xDescriptor(nullptr, {handle.miniBatchSize,
        handle.layerSize, 1}, {handle.layerSize, 1, 1}, handle.timesteps, precision);

    size_t size = 0;

    CudnnLibrary::cudnnGetRNNWorkspaceSize(rnnDescriptor->descriptor(),
        handle.timesteps,
        xDescriptor.descriptors(), &size);

    return size;
}

size_t cudnnGetWeightsSize(
    const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto rnnDescriptor = createRnnDescriptor(handle, precision);

    CudnnTensorViewDescriptorArray xDescriptor(nullptr, {handle.miniBatchSize,
        handle.layerSize, 1}, {handle.layerSize, 1, 1}, handle.timesteps, precision);

    size_t size = 0;

    CudnnLibrary::cudnnGetRNNParamsSize(rnnDescriptor->descriptor(),
        xDescriptor.descriptors()[0], &size, convertPrecision(precision));

    return size;
}

static size_t getArraysPerLayer(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE)
    {
        return 4;
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        return 12;
    }
    else
    {
        return 16;
    }
}

static size_t getLayer(const RecurrentOpsHandle& handle, size_t index)
{
    return index / getArraysPerLayer(handle);
}

static size_t isLinearLayer(const RecurrentOpsHandle& handle, size_t index)
{
    return index % 2 == 0;
}

static size_t getIdInLayer(const RecurrentOpsHandle& handle, size_t index)
{
    return (index % getArraysPerLayer(handle)) / 2;
}

void getOffsetAndDimensions(size_t& offset, matrix::Dimension& dimensions,
    const RecurrentOpsHandle& handle, const matrix::Precision& precision, size_t index)
{
    auto rnnDescriptor = createRnnDescriptor(handle, precision);

    CudnnTensorViewDescriptorArray xDescriptor(nullptr, {handle.miniBatchSize,
        handle.layerSize, 1}, {handle.layerSize, 1, 1}, handle.timesteps, precision);

    CudnnFilterViewDescriptor wDescriptor(nullptr,
        {cudnnGetWeightsSize(handle, precision) / precision.size(), 1, 1},
        precision);

    CudnnFilterViewDescriptor filter(nullptr, {1, 1, 1}, precision);

    void* address = nullptr;

    if(isLinearLayer(handle, index))
    {
        CudnnLibrary::cudnnGetRNNLinLayerMatrixParams(rnnDescriptor->descriptor(),
                                                      getLayer(handle, index),
                                                      *xDescriptor.descriptors(),
                                                      wDescriptor.descriptor(),
                                                      nullptr,
                                                      getIdInLayer(handle, index),
                                                      filter.descriptor(),
                                                      &address);
    }
    else
    {
        CudnnLibrary::cudnnGetRNNLinLayerBiasParams(rnnDescriptor->descriptor(),
                                                    getLayer(handle, index),
                                                    *xDescriptor.descriptors(),
                                                    wDescriptor.descriptor(),
                                                    nullptr,
                                                    getIdInLayer(handle, index),
                                                    filter.descriptor(),
                                                    &address);
    }

    offset = reinterpret_cast<size_t>(address) / precision.size();

    dimensions = filter.getDimensions();
}

size_t cudnnGetWeightsBegin(const RecurrentOpsHandle& handle, const matrix::Precision& precision,
    size_t index)
{
    size_t offset = 0;
    matrix::Dimension dimensions;

    getOffsetAndDimensions(offset, dimensions, handle, precision, index);

    return offset;
}

size_t cudnnGetWeightsEnd(const RecurrentOpsHandle& handle, const matrix::Precision& precision,
    size_t index)
{
    size_t offset = 0;
    matrix::Dimension dimensions;

    getOffsetAndDimensions(offset, dimensions, handle, precision, index);

    return offset + dimensions.product();
}

}

}



