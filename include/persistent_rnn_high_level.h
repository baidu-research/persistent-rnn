/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

#pragma once

// Forward Declarations
namespace prnn { namespace matrix { class Matrix; } }

namespace prnn { class RecurrentOpsHandle; }

namespace prnn {

/** \brief Forward propagate through a recurrent weight matrix.
 *  \param weights The recurrent weight matrix.
    \param reserve Memory allocated for storing data needed for back propagation
                   (stored as [layer-size, mini-batch, timesteps, layer-count])
 *  \param activations The input/output activations from the previous layer
                       (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void forwardPropRecurrent(matrix::Matrix& activations,
                          matrix::Matrix& reserve,
                          const matrix::Matrix& weights,
                          const RecurrentOpsHandle& handle);

/** \brief Back propagate through a recurrent weight matrix, generating deltas.
 *  \param weights The recurrent weight matrix.
 *  \param activations The input/output activations from the previous layer
                       (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void backPropDeltasRecurrent(matrix::Matrix& deltas,
    const matrix::Matrix& weights,
    const matrix::Matrix& activations,
    const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle);

/** \brief Compute gradient for the recurrent weight matrix.
 *  \param activations Activations for the layer.
 *  \param deltas Deltas for the layer.
 *  \param dWeights The output gradients.
 */
void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& activations,
    const matrix::Matrix& deltas,
    const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle);

}


