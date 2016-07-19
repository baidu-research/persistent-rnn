
// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/blas_operations.h>
#include <prnn/detail/matrix/operation.h>
#include <prnn/detail/matrix/cudnn_library.h>

#include <prnn/detail/parallel/cuda.h>
#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/metaprogramming.h>
#include <prnn/detail/util/logger.h>

#include <prnn/detail/rnn/recurrent_ops_config.h>
#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops_kernels.h>
#include <prnn/detail/rnn/cudnn_ops.h>

namespace prnn
{

namespace rnn
{

namespace detail
{

template<RecurrentLayerDirection direction, typename T, size_t sms, size_t smMajor>
class TileSelector
{
public:
    typedef TileConfig<4, 384, 384, 192, 288, 6, 36, direction, T> TileSize;
    //typedef TileConfig<1, 8, 8, 4, 4, 2, 4, direction, T> TileSize;

};

#if CUDA_ARCH_MAJOR == 6

template<RecurrentLayerDirection direction, typename T>
class TileSelector<direction, T, 56, 6>
{
public:
    typedef TileConfig<56, 1792, 1792, 224, 256, 7, 32, direction, T> TileSize;
};

template<RecurrentLayerDirection direction>
class TileSelector<direction, float16, 56, 6>
{
public:
    typedef TileConfig<56, 2432, 2560, 352, 320, 11, 16, direction, T> TileSize;
};

#endif

#if CUDA_ARCH_MAJOR == 5

template<RecurrentLayerDirection direction, typename T>
class TileSelector<direction, T, 24, 5>
{
public:
    typedef TileConfig<24, 1152, 1152, 192, 288, 6, 36, direction, T> TileSize;
};

#endif

class TileSizeSelector
{
public:
    TileSizeSelector(size_t major, size_t minor, size_t smCount,
        const matrix::Precision& precision)
    : streamingMultiprocessorVersionMajor(major),
      streamingMultiprocessorVersionMinor(minor),
      streamingMultiprocessorCount(smCount),
      precision(precision)
    {

    }

public:
    size_t getMaximumSize() const
    {
        size_t maxSize = 0;

        if(streamingMultiprocessorVersionMajor == 6 && streamingMultiprocessorCount >= 60)
        {
            if(precision == matrix::HalfPrecision())
            {
                maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                    float16, 56, 6>::TileSize::MAXIMUM_LAYER_SIZE;
            }
            else
            {
                maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                    float, 56, 6>::TileSize::MAXIMUM_LAYER_SIZE;
            }
        }
        else if(streamingMultiprocessorVersionMajor == 5 && streamingMultiprocessorCount >= 24)
        {
            maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                float, 24, 5>::TileSize::MAXIMUM_LAYER_SIZE;
        }
        else
        {
            maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                float, 1, 0>::TileSize::MAXIMUM_LAYER_SIZE;
        }

        util::log("RecurrentOperations") << "major " << streamingMultiprocessorVersionMajor
            << ", minor " << streamingMultiprocessorVersionMinor << ", sms "
            << streamingMultiprocessorCount << ", max size is " << maxSize << "\n";

        return maxSize;
    }

    size_t getScratchSize() const
    {
        size_t maxSize = 0;

        if(streamingMultiprocessorVersionMajor == 6 && streamingMultiprocessorCount >= 60)
        {
            if(precision == matrix::HalfPrecision())
            {
                maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                    float16, 56, 6>::TileSize::EXPANDED_LAYER_SIZE;
            }
            else
            {
                maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                    float, 56, 6>::TileSize::EXPANDED_LAYER_SIZE;
            }
        }
        else if(streamingMultiprocessorVersionMajor == 5 && streamingMultiprocessorCount >= 24)
        {
            maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                float, 24, 5>::TileSize::EXPANDED_LAYER_SIZE;
        }
        else
        {
            maxSize = TileSelector<prnn::RECURRENT_FORWARD,
                float, 1, 0>::TileSize::EXPANDED_LAYER_SIZE;
        }

        util::log("RecurrentOperations") << "major " << streamingMultiprocessorVersionMajor
            << ", minor " << streamingMultiprocessorVersionMinor << ", sms "
            << streamingMultiprocessorCount << ", scratch size per timestep is "
            << maxSize << "\n";

        return maxSize;
    }

public:
    size_t streamingMultiprocessorVersionMajor;
    size_t streamingMultiprocessorVersionMinor;

    size_t streamingMultiprocessorCount;

public:
    matrix::Precision precision;

};

void getGPUMajorAndMinorVersion(int& major, int& minor, int& smCount)
{
    if(prnn::parallel::isCudaEnabled())
    {
        prnn::parallel::CudaRuntimeLibrary::cudaDeviceGetAttribute(&major,
            prnn::parallel::CudaRuntimeLibrary::cudaDevAttrComputeCapabilityMajor, 0);
        prnn::parallel::CudaRuntimeLibrary::cudaDeviceGetAttribute(&minor,
            prnn::parallel::CudaRuntimeLibrary::cudaDevAttrComputeCapabilityMajor, 0);
        prnn::parallel::CudaRuntimeLibrary::cudaDeviceGetAttribute(&smCount,
            prnn::parallel::CudaRuntimeLibrary::cudaDevAttrMultiProcessorCount, 0);
    }
}

} // namespace detail

size_t getMaximumSizeRNNForThisGPU(const matrix::Precision& precision)
{
    int major   = 0;
    int minor   = 0;
    int smCount = 0;

    detail::getGPUMajorAndMinorVersion(major, minor, smCount);

    return detail::TileSizeSelector(major, minor, smCount, precision).getMaximumSize();
}

size_t getScratchSizeRNNForThisGPU(const matrix::Precision& precision)
{
    int major   = 0;
    int minor   = 0;
    int smCount = 0;

    detail::getGPUMajorAndMinorVersion(major, minor, smCount);

    return detail::TileSizeSelector(major, minor, smCount, precision).getScratchSize();
}

namespace detail
{

template <typename ArchitectureConfig>
static index_t* getSynchronizerScratch(typename ArchitectureConfig::RealType* scratch,
    const ArchitectureConfig& archParameters)
{
    size_t totalSize = archParameters.scratch_activations_per_grid() *
        archParameters.handle.miniBatchSize *
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
        << archParameters.threads().x << " x " << archParameters.threads().y
        << " threads, in stream " << archParameters.handle.stream << "), each handling "
        << archParameters.activations_per_block() << " activations out of "
        << activationCount << " total, mini batch size " << miniBatchSize << ", timesteps "
        << timesteps << ".\n";

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(archParameters.handle.stream);

    Synchronizer synchronizer(archParameters.block_count(), stream,
        getSynchronizerScratch(scratch, archParameters));

    typedef typename ArchitectureConfig::TileParameters TileConfig;

    typedef RecurrentConfig<RealType, ActivationFunction, TileConfig> Config;

    Config config(archParameters.handle);

    while(synchronizer.not_finished()) {
        PersistentEngineParameters<Config> parameters(config, weights, activations,
            scratch, archParameters.handle.skipConnectionScale, synchronizer);

        forward_prop_recurrent_kernel<<<archParameters.blocks(),
            archParameters.threads(), 0, stream>>>(parameters);

        synchronizer.check_for_failure();

        if (synchronizer.not_finished()) {
            util::log("RecurrentOperations::Detail")
                << " forward prop launch failed, restarting at phase "
                << synchronizer.get_current_phase() << ".\n";
            synchronizer.reset_failed_flag();
        }
    }

}

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

    int major   = 0;
    int minor   = 0;
    int smCount = 0;

    getGPUMajorAndMinorVersion(major, minor, smCount);

    if(major == 6 && smCount >= 60)
    {
        typedef typename TileSelector<direction, RealType, 60, 6>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchForwardPropRecurrent<ActivationFunction, ArchParams>(activationData, weightsData,
            scratchData, architectureConfig);
    }
    else if(major == 5 && smCount >= 24)
    {
        typedef typename TileSelector<direction, RealType, 24, 5>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchForwardPropRecurrent<ActivationFunction, ArchParams>(activationData, weightsData,
            scratchData, architectureConfig);
    }
    else
    {
        typedef typename TileSelector<direction, RealType, 1, 0>::TileSize TileSize;
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
        handle, prnn::matrix::RecurrentPrecisions());
}

template<typename ActivationFunction>
void forwardPropRecurrentOverActivationFunctions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<ActivationFunction>& activationFunction)
{
    assert(ActivationFunction() == *handle.activationFunction.forwardOperation);

    forwardPropRecurrentOverPrecisions<ActivationFunction>(activations, weights, scratch, handle);
}

template<typename Functions>
void forwardPropRecurrentOverActivationFunctions(const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const Functions& functions)
{
    typedef typename std::tuple_element<0, Functions>::type PossibleFunction;

    if(*handle.activationFunction.forwardOperation == PossibleFunction())
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

void genericForwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& weights,
    const RecurrentOpsHandle& handle)
{
    bool reversed = (handle.direction == prnn::RECURRENT_REVERSE);

    size_t timesteps     = activations.size()[2];
    size_t miniBatchSize = activations.size()[1];
    size_t layerSize     = activations.size()[0];

    size_t currentTimestep = reversed ? timesteps - 1 : 0;

    // Start value
    auto currentInput = slice(activations, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentInput, currentInput, *handle.activationFunction.forwardOperation);

    // Propagate through time
    for(size_t timestep = 1; timestep < timesteps; ++timestep)
    {
        currentTimestep = reversed ? timesteps - timestep - 1 : timestep;

        auto nextInput = slice(activations, {0, 0, currentTimestep},
            {layerSize, miniBatchSize, currentTimestep + 1});

        auto reshapedNextInput    = reshape(nextInput,    {layerSize, miniBatchSize});
        auto reshapedCurrentInput = reshape(currentInput, {layerSize, miniBatchSize});

        gemm(
            reshapedNextInput,           1.0,
            weights,              false, 1.0,
            reshapedCurrentInput, false);

        apply(nextInput, currentInput, nextInput,
            matrix::MultiplyAccumulate(handle.skipConnectionScale));

        currentInput = nextInput;

        apply(currentInput, currentInput, *handle.activationFunction.forwardOperation);
    }
}

}

void forwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& weights,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    assert(activations.precision() == weights.precision());
    assert(activations.precision() == scratch.precision());
    assert(activations.precision() == reserve.precision());

    auto backend = getBackendThrowOnError(handle, activations.precision());

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        parallel::setNotSynchronized();

        cudnnForwardPropRecurrent(activations, inputActivations,
            weights, scratch, reserve, handle);
    }
    else if(backend == RECURRENT_PERSISTENT_BACKEND)
    {
        zeros(scratch);

        copy(activations, inputActivations);

        detail::forwardPropRecurrentOverActivationFunctions(activations,
            reshape(weights, {handle.layerSize, handle.layerSize}), scratch, handle);
    }
    else
    {
        copy(activations, inputActivations);

        detail::genericForwardPropRecurrent(activations,
            reshape(weights, {handle.layerSize, handle.layerSize}), handle);
    }
}

namespace detail
{

template <typename ActivationFunction, typename ArchitectureConfig>
void dispatchBackPropDeltasRecurrent(
    typename ArchitectureConfig::RealType* deltas,
    const typename ArchitectureConfig::RealType* weights,
    typename ArchitectureConfig::RealType* activations,
    typename ArchitectureConfig::RealType* scratch, const ArchitectureConfig& archParameters)
{
    typedef typename ArchitectureConfig::RealType RealType;

    size_t activationCount = archParameters.handle.layerSize;
    size_t miniBatchSize   = archParameters.handle.miniBatchSize;
    size_t timesteps       = archParameters.handle.timesteps;

    util::log("RecurrentOperations") << "Launch back propagation with "
        << archParameters.block_count() << " blocks ("
        << archParameters.thread_count() << " threads), each handling "
        << archParameters.activations_per_block() << " activations out of "
        << activationCount << " total, mini batch size " << miniBatchSize << ", timesteps "
        << timesteps << ".\n";

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(archParameters.handle.stream);

    Synchronizer synchronizer(archParameters.block_count(), stream,
        getSynchronizerScratch(scratch, archParameters));

    typedef typename ArchitectureConfig::TileParameters TileConfig;

    typedef RecurrentConfig<RealType, ActivationFunction, TileConfig> Config;

    Config config(archParameters.handle);

    while(synchronizer.not_finished()) {
        PersistentEngineParameters<Config> parameters(config, weights, activations,
            deltas, scratch, archParameters.handle.skipConnectionScale, synchronizer);

        back_prop_recurrent_deltas_kernel<<<archParameters.blocks(),
            archParameters.threads(), 0, stream>>>(parameters);

        synchronizer.check_for_failure();

        if (synchronizer.not_finished()) {
            util::log("RecurrentOperations") << " back prop launch failed, restarting at phase "
                << synchronizer.get_current_phase() << ".\n";
            synchronizer.reset_failed_flag();
        }
    }
}

template <typename ActivationFunction, typename T, RecurrentLayerDirection direction>
void backPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<T>& precision)
{
    typedef typename T::type RealType;

    const RealType* weightsData    = weights.data<RealType>();
          RealType* activationData = const_cast<RealType*>(activations.data<RealType>());
          RealType* deltaData      = deltas.data<RealType>();
          RealType* scratchData    = scratch.data<RealType>();

    int major   = 0;
    int minor   = 0;
    int smCount = 0;

    getGPUMajorAndMinorVersion(major, minor, smCount);

    if(major == 6 && smCount >= 60)
    {
        typedef typename TileSelector<direction, RealType, 60, 6>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchBackPropDeltasRecurrent<ActivationFunction, ArchParams>(deltaData,
            weightsData, activationData, scratchData, architectureConfig);
    }
    else if(major == 5 && smCount >= 24)
    {
        typedef typename TileSelector<direction, RealType, 24, 5>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchBackPropDeltasRecurrent<ActivationFunction, ArchParams>(deltaData,
            weightsData, activationData, scratchData, architectureConfig);
    }
    else
    {
        typedef typename TileSelector<direction, RealType, 1, 0>::TileSize TileSize;
        typedef RecurrentArchitectureParameters<RealType, TileSize> ArchParams;

        ArchParams architectureConfig(handle);

        dispatchBackPropDeltasRecurrent<ActivationFunction, ArchParams>(deltaData,
            weightsData, activationData, scratchData, architectureConfig);
    }
}

template <typename ActivationFunction, typename T>
void backPropDeltasRecurrentOverPrecisions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<T>& precision)
{
    typedef T PossiblePrecision;

    assert(PossiblePrecision() == activations.precision());

    if(handle.direction == prnn::RECURRENT_REVERSE)
    {
        backPropDeltasRecurrent<ActivationFunction, PossiblePrecision, prnn::RECURRENT_FORWARD>(
            deltas, weights, activations, scratch, handle, precision);
    }
    else
    {
        backPropDeltasRecurrent<ActivationFunction, PossiblePrecision, prnn::RECURRENT_REVERSE>(
            deltas, weights, activations, scratch, handle, precision);
    }
}

template<typename ActivationFunction, typename Precisions>
void backPropDeltasRecurrentOverPrecisions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const Precisions& precisions)
{
    typedef typename std::tuple_element<0, Precisions>::type PossiblePrecision;

    if(activations.precision() == PossiblePrecision())
    {
        backPropDeltasRecurrentOverPrecisions<ActivationFunction>(
            deltas, weights, activations, scratch, handle, std::tuple<PossiblePrecision>());
    }
    else
    {
        typedef typename util::RemoveFirstType<Precisions>::type RemainingPrecisions;

        backPropDeltasRecurrentOverPrecisions<ActivationFunction>(deltas, weights, activations,
            scratch, handle, RemainingPrecisions());
    }
}

template <typename ActivationFunction>
void backPropDeltasRecurrentOverPrecisions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{
    backPropDeltasRecurrentOverPrecisions<ActivationFunction>(deltas, weights, activations,
        scratch, handle, prnn::matrix::RecurrentPrecisions());
}

template<typename ActivationFunction>
void backPropDeltasRecurrentOverActivationFunctions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const std::tuple<ActivationFunction>& activationFunction)
{
    assert(ActivationFunction() == *handle.activationFunction.reverseOperation);

    backPropDeltasRecurrentOverPrecisions<ActivationFunction>(deltas,
        weights, activations, scratch, handle);
}

template<typename Functions>
void backPropDeltasRecurrentOverActivationFunctions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle,
    const Functions& functions)
{
    typedef typename std::tuple_element<0, Functions>::type PossibleFunction;

    if(*handle.activationFunction.reverseOperation == PossibleFunction())
    {
        backPropDeltasRecurrentOverActivationFunctions(deltas, weights, activations,
            scratch, handle, std::tuple<PossibleFunction>());
    }
    else
    {
        typedef typename prnn::util::RemoveFirstType<Functions>::type RemainingFunctions;

        backPropDeltasRecurrentOverActivationFunctions(deltas, weights, activations,
            scratch, handle, RemainingFunctions());
    }
}

void backPropDeltasRecurrentOverActivationFunctions(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const matrix::DynamicView& scratch, const RecurrentOpsHandle& handle)
{
    backPropDeltasRecurrentOverActivationFunctions(deltas, weights, activations, scratch,
        handle, prnn::matrix::AllRecurrentBackwardOps());
}

void genericBackPropDeltasRecurrent(const matrix::DynamicView& deltas,
    const matrix::ConstDynamicView& weights, const matrix::ConstDynamicView& activations,
    const RecurrentOpsHandle& handle)
{
    bool reversed = (handle.direction == prnn::RECURRENT_REVERSE);

    size_t maxTimesteps  = deltas.size()[2];
    size_t miniBatchSize = deltas.size()[1];
    size_t layerSize     = deltas.size()[0];

    auto currentTimestep = reversed ? 0 : maxTimesteps - 1;

    // Start value
    auto currentDeltas = slice(deltas,
        {0, 0, currentTimestep}, {layerSize, miniBatchSize, currentTimestep + 1});
    auto currentActivations = slice(activations,
        {0, 0, currentTimestep}, {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentDeltas, currentActivations, currentDeltas,
        *handle.activationFunction.reverseOperation);

    // go over all timesteps in reverse
    for(size_t t = 1; t < maxTimesteps; ++t)
    {
        size_t timestep = reversed ? t : maxTimesteps - t - 1;

        auto previousDeltas = slice(deltas, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        auto reshapedPreviousDeltas = reshape(previousDeltas, {layerSize, miniBatchSize});
        auto reshapedCurrentDeltas  = reshape(currentDeltas,  {layerSize, miniBatchSize});

        gemm(
            reshapedPreviousDeltas, 1.0,
            weights, true, 1.0,
            reshapedCurrentDeltas, false
        );

        apply(previousDeltas, currentDeltas, previousDeltas,
            matrix::MultiplyAccumulate(handle.skipConnectionScale));

        currentDeltas = previousDeltas;

        currentActivations = slice(activations, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        apply(currentDeltas, currentActivations, currentDeltas,
            *handle.activationFunction.reverseOperation);
    }
}

}

void backPropDeltasRecurrent(const matrix::DynamicView& inputDeltas,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& outputDeltas,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto backend = getBackendThrowOnError(handle, weights.precision());

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        parallel::setNotSynchronized();

        cudnnBackPropDeltasRecurrent(inputDeltas, weights, outputActivations, outputDeltas,
            scratch, reserve, handle);
    }
    else if(backend == RECURRENT_PERSISTENT_BACKEND)
    {
        zeros(scratch);

        copy(inputDeltas, outputDeltas);

        detail::backPropDeltasRecurrentOverActivationFunctions(inputDeltas,
            reshape(weights, {handle.layerSize, handle.layerSize}),
            outputActivations, scratch, handle);

        copy(reserve, inputDeltas);
    }
    else
    {
        copy(inputDeltas, outputDeltas);

        detail::genericBackPropDeltasRecurrent(inputDeltas,
            reshape(weights, {handle.layerSize, handle.layerSize}), outputActivations, handle);

        copy(reserve, inputDeltas);
    }
}

namespace detail
{

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& deltas,
    const RecurrentOpsHandle& handle)
{
    bool reversed = (handle.direction == prnn::RECURRENT_REVERSE);

    size_t timesteps     = handle.timesteps;
    size_t miniBatchSize = handle.miniBatchSize;
    size_t layerSize     = handle.layerSize;

    // Compute gradients
    size_t start = reversed ? 0 : 1;

    auto slicedDeltas = slice(deltas,
                             {0,         0,             start},
                             {layerSize, miniBatchSize, timesteps - (1 - start)});


    auto slicedActivations = slice(outputActivations,
                                  {0,         0,             1         - start},
                                  {layerSize, miniBatchSize, timesteps - start});

    gemm(dWeights, 0.0,
         reshape(slicedDeltas,
                {layerSize, miniBatchSize * (timesteps - 1)}), false, 1.0,
         reshape(slicedActivations,
                {layerSize, miniBatchSize * (timesteps - 1)}), true);

}

} // namespace detail

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle)
{
    auto backend = getBackendThrowOnError(handle, dWeights.precision());

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        zeros(dWeights);

        parallel::setNotSynchronized();

        cudnnBackPropGradientsRecurrent(dWeights, inputActivations,
            outputActivations, scratch, reserve, handle);
    }
    else
    {
        detail::backPropGradientsRecurrent(
            reshape(dWeights, getWeightDimensions(handle, dWeights.precision())),
            outputActivations,
            reshape(reserve, {handle.layerSize, handle.miniBatchSize, handle.timesteps}), handle);
    }
}

static matrix::Dimension extendDimensions(const matrix::Dimension& dimensions,
    const matrix::Precision& precision)
{
    auto newDimensions = dimensions;

    newDimensions[0] = prnn::rnn::getScratchSizeRNNForThisGPU(precision);
    newDimensions[2] += 1;

    return newDimensions;
}

static matrix::Dimension getForwardPropScratchDimensions(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    matrix::Dimension scratchDimension;

    auto backend = getBackendThrowOnError(handle, precision);

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        scratchDimension = {cudnnGetScratchSize(handle, precision) / precision.size()};
    }
    else
    {
        matrix::Dimension dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps);

        scratchDimension = extendDimensions(dimension, precision);
    }

    return scratchDimension;
}

matrix::Matrix getForwardPropScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return matrix::Matrix(getForwardPropScratchDimensions(handle, precision), precision);
}

size_t getForwardPropScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return getForwardPropScratchDimensions(handle, precision).product() * precision.size();
}

matrix::Dimension getReserveDimensions(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    matrix::Dimension size;

    auto backend = getBackendThrowOnError(handle, precision);

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        size.push_back(cudnnGetReserveSize(handle, precision) / precision.size());
    }
    else
    {
        size = {handle.layerSize, handle.miniBatchSize, handle.timesteps};
    }

    return size;
}

matrix::Dimension getWeightDimensions(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    matrix::Dimension size;

    auto backend = getBackendThrowOnError(handle, precision);

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        size.push_back(cudnnGetWeightsSize(handle, precision) / precision.size());
        size.push_back(1);
        size.push_back(1);
    }
    else
    {
        size = {handle.layerSize, handle.layerSize};
    }

    return size;
}

void getWeightsRange(matrix::Dimension& begin, matrix::Dimension& end,
    const RecurrentOpsHandle& handle, const matrix::Precision& precision,
    int index)
{
    begin.clear();
    end.clear();

    auto backend = getBackendThrowOnError(handle, precision);

    if(backend == RECURRENT_CUDNN_BACKEND)
    {
        begin.push_back(cudnnGetWeightsBegin(handle, precision, index));
        begin.push_back(0);
        begin.push_back(0);

        end.push_back(cudnnGetWeightsEnd(handle, precision, index));
        end.push_back(1);
        end.push_back(1);
    }
    else
    {
        begin = {0, 0};
        end   = {handle.layerSize, handle.layerSize};
    }
}

matrix::Matrix getBackPropDeltasScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return getForwardPropScratch(handle, precision);
}

size_t getBackPropDeltasScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return getForwardPropScratchSize(handle, precision);
}

matrix::Matrix getBackPropGradientsScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return getForwardPropScratch(handle, precision);
}

size_t getBackPropGradientsScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    return getForwardPropScratchSize(handle, precision);
}

}
}

