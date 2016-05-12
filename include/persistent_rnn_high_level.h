/*! \file  persistent_rnn_high_level.h
    \date  May 10, 2016
    \brief C++ language interface to persistent RNN kernels.
*/

#pragma once

// Forward Declarations
namespace prnn { class Operation;          }
namespace prnn { class Matrix;             }
namespace prnn { class RecurrentOpsHandle; }

namespace prnn {

/** \brief Forward propagate through a recurrent weight matrix.
 *  \param activationFunction The activation function to apply to the end of each
           recurrent weight matrix application.
 *  \param weights The recurrent weight matrix.
 *  \param activations The input/output activations from the previous layer
           (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void forward_prop_recurrent(const RecurrentOpsHandle& handle,
    const Operation& activationFunction, RecurrentLayerDirection direction,
    const Matrix& weights, Matrix& activations);

/** \brief Back propagate through a recurrent weight matrix, generating deltas.
 *  \param activationFunction The activation (derivative) function to apply to the end of
           each recurrent weight matrix application.
 *  \param weights The recurrent weight matrix.
 *  \param activations The input/output activations from the previous layer
           (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void back_prop_deltas_recurrent(const RecurrentOpsHandle& handle,
    const Operation& activationFunction, RecurrentLayerDirection direction,
    const Matrix& weights,
    const Matrix& activations,
    Matrix& deltas);

/** \brief Compute gradient for the recurrent weight matrix.
 *  \param activations Activations for the layer.
 *  \param deltas Deltas for the layer.
 *  \param dWeights The output gradients.
 */
void back_prop_gradients_recurrent(const RecurrentOpsHandle& handle,
    RecurrentLayerDirection direction,
    const Matrix& activations,
    const Matrix& deltas,
    Matrix& dWeights);

}


