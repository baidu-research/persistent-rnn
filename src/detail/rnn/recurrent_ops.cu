
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/parallel/cuda.h>

#include <prnn/detail/util/metaprogramming.h>
#include <prnn/detail/util/logger.h>

#include <prnn/detail/rnn/recurrent_ops_config.h>
#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops_kernels.h>

namespace prnn
{

namespace rnn
{

namespace detail
{

class TileSizeSelector
{
public:
    TileSizeSelector(size_t major, size_t minor)
    : streamingMultiprocessorVersionMajor(major),
      streamingMultiprocessorVersionMinor(minor)
    {

    }

public:
    size_t getMaximumSize() const
    {
        if(streamingMultiprocessorVersionMajor == 5)
        {
            return 1088;
        }
        else if(streamingMultiprocessorVersionMajor == 6)
        {
            return 2720;
        }
        else
        {
            return 4;
        }
    }

public:
    size_t streamingMultiprocessorVersionMajor;
    size_t streamingMultiprocessorVersionMinor;

};

void getGPUMajorAndMinorVersion(int& major, int& minor)
{
    if(prnn::parallel::isCudaEnabled())
    {
        prnn::parallel::CudaRuntimeLibrary::cudaDeviceGetAttribute(&major,
            prnn::parallel::CudaRuntimeLibrary::cudaDevAttrComputeCapabilityMajor, 0);
        prnn::parallel::CudaRuntimeLibrary::cudaDeviceGetAttribute(&minor,
            prnn::parallel::CudaRuntimeLibrary::cudaDevAttrComputeCapabilityMajor, 0);
    }
}

} // namespace detail

size_t getMaximumSizeRNNForThisGPU()
{
    int major = 0;
    int minor = 0;

    detail::getGPUMajorAndMinorVersion(major, minor);

    return detail::TileSizeSelector(major, minor).getMaximumSize();
}

namespace detail
{

template <typename ArchitectureConfig>
static index_t* getSynchronizerScratch(typename ArchitectureConfig::RealType* scratch,
    const ArchitectureConfig& archParameters)
{
    size_t totalSize = archParameters.handle.layerSize * archParameters.handle.miniBatchSize *
        archParameters.handle.timesteps;

    return reinterpret_cast<index_t*>(scratch + totalSize);
}

template <typename ActivationFunction, typename ArchitectureConfig>
void dispatchForwardPropRecurrent(typename ArchitectureConfig::RealType* activations,
    const typename ArchitectureConfig::RealType* weights,
    typename ArchitectureConfig::RealType* scratch, const ArchitectureConfig& archParameters)
{
    typedef typename ArchitectureConfig::RealType RealType;

    size_t activationCount = archParameters.handle.layerSize;
    size_t miniBatchSize   = archParameters.handle.miniBatchSize;
    size_t timesteps       = archParameters.handle.timesteps;

    util::log("RecurrentOperations") << "Launch forward propagation with "
        << archParameters.block_count() << " blocks ("
        << archParameters.thread_count() << " threads), each handling "
        << archParameters.activations_per_block() << " activations out of "
        << activationCount << " total, mini batch size " << miniBatchSize << ", timesteps "
        << timesteps << ".\n";

    Synchronizer synchronizer(archParameters.block_count(), archParameters.handle.stream,
        getSynchronizerScratch(scratch, archParameters));

    typedef typename ArchitectureConfig::TileParameters TileConfig;

    typedef RecurrentConfig<RealType, ActivationFunction, TileConfig> Config;

    Config config(archParameters.handle);

    while(synchronizer.not_finished()) {
        PersistentEngineParameters<Config> parameters(config, weights, activations,
            scratch, archParameters.handle.skipConnectionScale, synchronizer);

        forward_prop_recurrent_kernel<<<archParameters.blocks(),
            archParameters.threads(), 0, archParameters.handle.stream>>>(parameters);

        synchronizer.check_for_failure();

        if (synchronizer.not_finished()) {
            util::log("RecurrentOperations") << " launch failed, restarting at phase "
                << synchronizer.get_current_phase() << ".\n";
            synchronizer.reset_failed_flag();
        }
    }

}

template<RecurrentLayerDirection direction, typename T, size_t sms, size_t smMajor>
class TileSelector
{
public:
    typedef TileConfig<24, 1088, 1088, 224, 224, 14, 14, direction> TileSize;

};

template<RecurrentLayerDirection direction, typename T>
class TileSelector<direction, T, 60, 6>
{
public:
    typedef TileConfig<60, 1820, 1820, 96, 96, 12, 12, direction> TileSize;
};

template<RecurrentLayerDirection direction>
class TileSelector<direction, float16, 60, 6>
{
public:
    typedef TileConfig<60, 2720, 2720, 352, 352, 22, 22, direction> TileSize;
};

template <typename ActivationFunction, typename T, RecurrentLayerDirection direction>
void forwardPropRecurrent(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<T>& precision)
{
    typedef typename T::type RealType;

    const RealType* weightsData    = weights.data<RealType>();
          RealType* activationData = activations.data<RealType>();
          RealType* scratchData    = scratch.data<RealType>();

    size_t timesteps = activations.size()[2];

    int major = 0;
    int minor = 0;

    getGPUMajorAndMinorVersion(major, minor);

    int smCount = 0;

    if(major == 6 && smCount >= 60)
    {
        typedef typename TileSelector<direction, RealType, 60, 6>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchForwardPropRecurrent<ActivationFunction, ArchParams>(activationData, weightsData,
            scratchData, architectureConfig);
    }
    else
    {
        typedef typename TileSelector<direction, RealType, 60, 6>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchForwardPropRecurrent<ActivationFunction, ArchParams>(activationData, weightsData,
            scratchData, architectureConfig);
    }
}

template <typename ActivationFunction, typename T>
void forwardPropRecurrentOverPrecisions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<T>& precision)
{
    typedef T PossiblePrecision;

    assert(PossiblePrecision() == activations.precision());

    if(handle.direction == prnn::RECURRENT_REVERSE)
    {
        forwardPropRecurrent<ActivationFunction, PossiblePrecision, prnn::RECURRENT_REVERSE>(
            activations, weights, scratch, handle, precision);
    }
    else
    {
        forwardPropRecurrent<ActivationFunction, PossiblePrecision, prnn::RECURRENT_FORWARD>(
            activations, weights, scratch, handle, precision);
    }
}

template<typename ActivationFunction, typename Precisions>
void forwardPropRecurrentOverPrecisions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const Precisions& precisions)
{
    typedef typename std::tuple_element<0, Precisions>::type PossiblePrecision;

    if(activations.precision() == PossiblePrecision())
    {
        forwardPropRecurrentOverPrecisions<ActivationFunction>(
            activations, weights, scratch, handle, std::tuple<PossiblePrecision>());
    }
    else
    {
        typedef typename util::RemoveFirstType<Precisions>::type RemainingPrecisions;

        forwardPropRecurrentOverPrecisions<ActivationFunction>(activations, weights,
            scratch, handle, RemainingPrecisions());
    }
}

template <typename ActivationFunction>
void forwardPropRecurrentOverPrecisions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{
    forwardPropRecurrentOverPrecisions<ActivationFunction>(activations, weights, scratch,
        handle, prnn::matrix::AllPrecisions());
}

template<typename ActivationFunction>
void forwardPropRecurrentOverActivationFunctions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<ActivationFunction>& activationFunction)
{
    assert(ActivationFunction() == handle.activationFunction.getForwardOperation());

    forwardPropRecurrentOverPrecisions<ActivationFunction>(activations, weights, scratch, handle);
}

template<typename Functions>
void forwardPropRecurrentOverActivationFunctions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const Functions& functions)
{
    typedef typename std::tuple_element<0, Functions>::type PossibleFunction;

    if(handle.activationFunction.getForwardOperation() == PossibleFunction())
    {
        forwardPropRecurrentOverActivationFunctions(activations, weights,
            scratch, handle, std::tuple<PossibleFunction>());
    }
    else
    {
        typedef typename prnn::util::RemoveFirstType<Functions>::type RemainingFunctions;

        forwardPropRecurrentOverActivationFunctions(activations, weights, scratch, handle,
            RemainingFunctions());
    }
}

void forwardPropRecurrentOverActivationFunctions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{
    forwardPropRecurrentOverActivationFunctions(activations, weights, scratch,
        handle, prnn::matrix::AllRecurrentForwardOps());
}

}

void forwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{
    assert(activations.precision() == weights.precision());
    assert(activations.precision() == scratch.precision());

    detail::forwardPropRecurrentOverActivationFunctions(activations, weights, scratch, handle);
}

void backPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{

}

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& activations,
    const matrix::ConstDynamicView& deltas,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle);

}

}






