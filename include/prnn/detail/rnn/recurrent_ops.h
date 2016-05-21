

#pragma once


// Forward Declarations
namespace prnn { class RecurrentOpsHandle; }

namespace prnn { namespace matrix { class DynamicView;      } }
namespace prnn { namespace matrix { class ConstDynamicView; } }
namespace prnn { namespace matrix { class Precision;        } }
namespace prnn { namespace matrix { class Matrix;           } }


namespace prnn
{
namespace rnn
{

size_t getMaximumSizeRNNForThisGPU(const matrix::Precision&);

void forwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

void backPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::DynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& activations,
    const matrix::ConstDynamicView& deltas,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

matrix::Matrix getForwardPropScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Matrix getBackPropDeltasScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Matrix getBackPropGradientsScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);


}
}


