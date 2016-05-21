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

void forwardPropRecurrent(matrix::Matrix& activations,
    const matrix::Matrix& weights, const RecurrentOpsHandle& handle)
{
    auto scratch = prnn::rnn::getForwardPropScratch(handle, activations.precision());

    prnn::rnn::forwardPropRecurrent(matrix::DynamicView(activations),
        matrix::ConstDynamicView(weights),
        matrix::DynamicView(scratch), handle);
}

void backPropDeltasRecurrent(matrix::Matrix& deltas,
    const matrix::Matrix& weights,
    const matrix::Matrix& activationsData,
    const RecurrentOpsHandle& handle)
{
    auto activations = activationsData;

    auto scratch = prnn::rnn::getBackPropDeltasScratch(handle, activations.precision());

    prnn::rnn::backPropDeltasRecurrent(matrix::DynamicView(deltas),
        matrix::ConstDynamicView(weights),
        matrix::DynamicView(activations),
        matrix::DynamicView(scratch),
        handle);
}

void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& activations,
    const matrix::Matrix& deltas,
    const RecurrentOpsHandle& handle)
{
    auto scratch = prnn::rnn::getBackPropGradientsScratch(handle,
        activations.precision());

    prnn::rnn::backPropGradientsRecurrent(matrix::DynamicView(dWeights),
        matrix::ConstDynamicView(activations),
        matrix::ConstDynamicView(deltas),
        matrix::DynamicView(scratch), handle);
}

}



