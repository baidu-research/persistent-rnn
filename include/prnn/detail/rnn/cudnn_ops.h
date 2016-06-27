
#pragma once

// Forward Declarations
namespace prnn                    { class RecurrentOpsHandle; }
namespace prnn { namespace matrix { class DynamicView;        } }
namespace prnn { namespace matrix { class ConstDynamicView;   } }

namespace prnn
{

namespace rnn
{

void cudnnForwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle);

void cudnnBackPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle);

void cudnnBackPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& deltas,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle);

}

}



