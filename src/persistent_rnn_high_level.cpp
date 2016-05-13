/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

#pragma once

// Persistent RNN Includes
#include <persistent_rnn_high_level.h>

namespace prnn {

void forward_prop_recurrent(const RecurrentOpsHandle& handle,
    const Operation& activationFunction, RecurrentLayerDirection direction,
    const Matrix& weights, Matrix& activations)
{
    Matrix scratch = get_forward_prop_scratch(handle, activations.size());

    forward_prop_recurrent(handle, activationFunction, direction, MatrixView(weights),
        MatrixView(activations), MatrixView(scratch));
}

void back_prop_deltas_recurrent(const RecurrentOpsHandle& handle,
    const Operation& activationFunction, RecurrentLayerDirection direction,
    const Matrix& weights,
    const Matrix& activations,
    Matrix& deltas)
{
    Matrix scratch = get_back_prop_deltas_scratch(handle, deltas.size());

    back_prop_deltas_recurrent(handle, activationFunction, direction, MatrixView(weights),
        MatrixView(activations), MatrixView(deltas), MatrixView(scratch));
}

void back_prop_gradients_recurrent(const RecurrentOpsHandle& handle,
    RecurrentLayerDirection direction,
    const Matrix& activations,
    const Matrix& deltas,
    Matrix& dWeights)
{
    Matrix scratch = get_back_prop_gradients_scratch(handle, activations.size());

    back_prop_gradients_recurrent(handle, direction, MatrixView(activations),
        MatrixView(deltas), MatrixView(dWeights), MatrixView(scratch));
}

}



