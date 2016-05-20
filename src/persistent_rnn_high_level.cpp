/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

// Persistent RNN Includes
#include <persistent_rnn_high_level.h>

#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/matrix_operations.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

namespace prnn
{

static matrix::Dimension extendDimensions(const matrix::Dimension& dimensions,
    const matrix::Precision& precision)
{
    auto newDimensions = dimensions;

    newDimensions[0] = prnn::rnn::getMaximumSizeRNNForThisGPU(precision);
    newDimensions[2] += 1;

    return newDimensions;
}

static matrix::Matrix getForwardPropScratch(const RecurrentOpsHandle& handle,
    const matrix::Dimension& dimension, const matrix::Precision& precision)
{
    auto scratchDimension = extendDimensions(dimension, precision);

    return matrix::zeros(scratchDimension, precision);
}

void forwardPropRecurrent(matrix::Matrix& activations,
    const matrix::Matrix& weights, const RecurrentOpsHandle& handle)
{
    auto scratch = getForwardPropScratch(handle, activations.size(), activations.precision());

    prnn::rnn::forwardPropRecurrent(matrix::DynamicView(activations),
        matrix::ConstDynamicView(weights),
        matrix::DynamicView(scratch), handle);
}

static matrix::Matrix getBackPropDeltasScratch(const RecurrentOpsHandle& handle,
    const matrix::Dimension& dimension, const matrix::Precision& precision)
{
    return getForwardPropScratch(handle, dimension, precision);
}

void backPropDeltasRecurrent(matrix::Matrix& deltas,
    const matrix::Matrix& weights,
    const matrix::Matrix& activationsData,
    const RecurrentOpsHandle& handle)
{
    auto activations = activationsData;

    auto scratch = getBackPropDeltasScratch(handle, deltas.size(), activations.precision());

    prnn::rnn::backPropDeltasRecurrent(matrix::DynamicView(deltas),
        matrix::ConstDynamicView(weights),
        matrix::DynamicView(activations),
        matrix::DynamicView(scratch),
        handle);
}

static matrix::Matrix getBackPropGradientsScratch(const RecurrentOpsHandle& handle,
    const matrix::Dimension& dimension, const matrix::Precision& precision)
{
    return matrix::Matrix();
}

void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& activations,
    const matrix::Matrix& deltas,
    const RecurrentOpsHandle& handle)
{
    auto scratch = getBackPropGradientsScratch(handle, activations.size(),
        activations.precision());

    prnn::rnn::backPropGradientsRecurrent(matrix::DynamicView(dWeights),
        matrix::ConstDynamicView(activations),
        matrix::ConstDynamicView(deltas),
        matrix::DynamicView(scratch), handle);
}

}



