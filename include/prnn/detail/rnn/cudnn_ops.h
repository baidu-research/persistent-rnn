
#pragma once

// Forward Declarations
namespace prnn                    { class RecurrentOpsHandle; }
namespace prnn { namespace matrix { class DynamicView;        } }
namespace prnn { namespace matrix { class ConstDynamicView;   } }
namespace prnn { namespace matrix { class Precision;          } }

// Standard Library Includes
#include <cstddef>

namespace prnn
{

namespace rnn
{

void cudnnForwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle);

void cudnnBackPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& outputDeltas,
    const matrix::DynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle);

void cudnnBackPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle);

size_t cudnnGetReserveSize(const RecurrentOpsHandle& handle, const matrix::Precision& precision);
size_t cudnnGetScratchSize(const RecurrentOpsHandle& handle, const matrix::Precision& precision);
size_t cudnnGetWeightsSize(const RecurrentOpsHandle& handle, const matrix::Precision& precision);

size_t cudnnGetWeightsBegin(const RecurrentOpsHandle& handle, const matrix::Precision& precision,
    size_t index);
size_t cudnnGetWeightsEnd(const RecurrentOpsHandle& handle, const matrix::Precision& precision,
    size_t index);

}

}



