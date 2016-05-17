
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/matrix/matrix_view.h>

#include <prnn/detail/rnn/recurrent_ops_config.h>
#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops_kernels.h>


namespace prnn
{

namespace rnn
{

size_t getMaximumSizeRNNForThisGPU()
{

}

void forwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

void backPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& activations,
    const matrix::ConstDynamicView& deltas,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

}

}






