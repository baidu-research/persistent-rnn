/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

#pragma once

// Forward Declarations
namespace prnn { class Matrix;             }
namespace prnn { class RecurrentOpsHandle; }

namespace prnn {

/** \brief Forward propagate through a recurrent weight matrix.
 *  \param weights The recurrent weight matrix.
 *  \param activations The input/output activations from the previous layer
           (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void forwardPropRecurrent(const RecurrentOpsHandle& handle,
    const Matrix& weights, Matrix& activations);

/** \brief Back propagate through a recurrent weight matrix, generating deltas.
 *  \param weights The recurrent weight matrix.
 *  \param activations The input/output activations from the previous layer
           (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void backPropDeltasRecurrent(const RecurrentOpsHandle& handle,
    const Matrix& weights,
    const Matrix& activations,
    Matrix& deltas);

/** \brief Compute gradient for the recurrent weight matrix.
 *  \param activations Activations for the layer.
 *  \param deltas Deltas for the layer.
 *  \param dWeights The output gradients.
 */
void backPropGradientsRecurrent(const RecurrentOpsHandle& handle,
    const Matrix& activations,
    const Matrix& deltas,
    Matrix& dWeights);

}


