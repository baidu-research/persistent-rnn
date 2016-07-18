/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

// Persistent RNN Includes
#include <persistent_rnn_high_level.h>

#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/copy_operations.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

namespace prnn
{

matrix::Matrix createReserveRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto size = prnn::rnn::getReserveDimensions(handle, precision);

    return matrix::Matrix(size, precision);
}

matrix::Matrix createWeightsRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    auto size = prnn::rnn::getWeightDimensions(handle, precision);

    return matrix::Matrix(size, precision);
}

matrix::Matrix sliceLayerWeights(const matrix::Matrix& weights, const RecurrentOpsHandle& handle,
    size_t index)
{
    matrix::Dimension begin;
    matrix::Dimension end;

    prnn::rnn::getWeightsRange(begin, end, handle, weights.precision(), index);

    return slice(weights, begin, end);
}

void forwardPropRecurrent(matrix::Matrix& activations,
    matrix::Matrix& reserve,
    const matrix::Matrix& weights,
    const RecurrentOpsHandle& handle)
{
    auto scratch = prnn::rnn::getForwardPropScratch(handle, activations.precision());

    prnn::rnn::forwardPropRecurrent(matrix::DynamicView(activations),
        matrix::ConstDynamicView(copy(activations)),
        matrix::ConstDynamicView(weights),
        matrix::DynamicView(scratch),
        matrix::DynamicView(reserve),
        handle);
}

void backPropDeltasRecurrent(matrix::Matrix& deltas,
    const matrix::Matrix& weights,
    const matrix::Matrix& activationsData,
    matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle)
{
    auto activations = activationsData;

    auto scratch = prnn::rnn::getBackPropDeltasScratch(handle, activations.precision());

    prnn::rnn::backPropDeltasRecurrent(matrix::DynamicView(deltas),
        matrix::ConstDynamicView(weights),
        matrix::ConstDynamicView(activations),
        matrix::ConstDynamicView(copy(deltas)),
        matrix::DynamicView(scratch),
        matrix::DynamicView(reserve),
        handle);
}

void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& activations,
    const matrix::Matrix& outputActivations,
    const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle)
{
    auto scratch = prnn::rnn::getBackPropGradientsScratch(handle,
        activations.precision());

    prnn::rnn::backPropGradientsRecurrent(matrix::DynamicView(dWeights),
        matrix::ConstDynamicView(activations),
        matrix::ConstDynamicView(outputActivations),
        matrix::DynamicView(scratch),
        matrix::ConstDynamicView(reserve),
        handle);
}

}



