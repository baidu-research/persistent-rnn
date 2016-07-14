
// Persistent RNN Includes
#include <persistent_rnn_high_level.h>

#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/random_operations.h>
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/logger.h>
#include <prnn/detail/util/timer.h>
#include <prnn/detail/util/argument_parser.h>

// Standard Library Includes
#include <random>
#include <iostream>

static double getFlopCount(prnn::RecurrentOpsHandle& handle)
{
    return 2.0 * handle.layerSize * handle.layerSize * handle.miniBatchSize * handle.timesteps;
}

void benchmarkRnnForward(size_t iterations, size_t layerSize, size_t miniBatchSize,
    size_t timesteps, size_t layers, prnn::RecurrentLayerBackend backend,
    const prnn::matrix::Precision& precision) {

    auto weights       = prnn::matrix::rand({layerSize, layerSize               }, precision);
    auto activations   = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);
    auto inActivations = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);

    prnn::RecurrentOpsHandle handle(layerSize, miniBatchSize, timesteps, layers,
        prnn::RecurrentRectifiedLinear(),
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        backend);

    auto scratch = prnn::rnn::getForwardPropScratch(handle, precision);
    auto reserve = prnn::createReserveRecurrent(handle, precision);

    // warm up
    prnn::rnn::forwardPropRecurrent(
        prnn::matrix::DynamicView(activations),
        prnn::matrix::ConstDynamicView(inActivations),
        prnn::matrix::ConstDynamicView(weights),
        prnn::matrix::DynamicView(scratch),
        prnn::matrix::DynamicView(reserve),
        handle);
    prnn::parallel::synchronize();

    prnn::util::Timer timer;

    timer.start();

    for (size_t i = 0; i < iterations; ++i) {
        prnn::rnn::forwardPropRecurrent(
            prnn::matrix::DynamicView(activations),
            prnn::matrix::ConstDynamicView(inActivations),
            prnn::matrix::ConstDynamicView(weights),
            prnn::matrix::DynamicView(scratch),
            prnn::matrix::DynamicView(reserve),
            handle);
    }

    prnn::parallel::synchronize();

    timer.stop();

    double totalFlops = iterations * getFlopCount(handle);

    double teraflops = totalFlops / (timer.seconds() * 1.0e12);
    double microsecondsPerKernel = timer.seconds() * 1.0e6 / iterations;

    std::cout << "RNN Forward Propagation: " << teraflops << " TFLOPS/s\n";
    std::cout << "RNN Average Kernel Time: " << microsecondsPerKernel << " us\n";
}

void benchmarkRnnReverse(size_t iterations, size_t layerSize, size_t miniBatchSize,
    size_t timesteps, size_t layers, prnn::RecurrentLayerBackend backend,
    const prnn::matrix::Precision& precision)
{
    auto weights     = prnn::matrix::rand({layerSize, layerSize               }, precision);
    auto activations = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);
    auto deltas      = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);
    auto outDeltas   = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);

    prnn::RecurrentOpsHandle handle(layerSize, miniBatchSize, timesteps, layers,
        prnn::RecurrentRectifiedLinear(),
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        backend);

    auto scratch = prnn::rnn::getBackPropDeltasScratch(handle, precision);
    auto reserve = prnn::createReserveRecurrent(handle, precision);

    prnn::matrix::randn(reserve);

    // warm up
    prnn::rnn::backPropDeltasRecurrent(prnn::matrix::DynamicView(deltas),
        prnn::matrix::ConstDynamicView(weights),
        prnn::matrix::ConstDynamicView(activations),
        prnn::matrix::ConstDynamicView(outDeltas),
        prnn::matrix::DynamicView(scratch),
        prnn::matrix::DynamicView(reserve),
        handle);
    prnn::parallel::synchronize();

    prnn::util::Timer timer;

    timer.start();

    for (size_t i = 0; i < iterations; ++i) {
        prnn::rnn::backPropDeltasRecurrent(prnn::matrix::DynamicView(deltas),
            prnn::matrix::ConstDynamicView(weights),
            prnn::matrix::ConstDynamicView(activations),
            prnn::matrix::ConstDynamicView(outDeltas),
            prnn::matrix::DynamicView(scratch),
            prnn::matrix::DynamicView(reserve), handle);
    }

    prnn::parallel::synchronize();

    timer.stop();

    double totalFlops = iterations * getFlopCount(handle);

    double teraflops = totalFlops / (timer.seconds() * 1.0e12);
    double microsecondsPerKernel = timer.seconds() * 1.0e6 / iterations;

    std::cout << "RNN Back Propagation Deltas: " << teraflops << " TFLOPS/s\n";
    std::cout << "RNN Average Kernel Time: " << microsecondsPerKernel << " us\n";
}

void benchmarkRnnGradients(size_t iterations, size_t layerSize, size_t miniBatchSize,
    size_t timesteps, size_t layers, prnn::RecurrentLayerBackend backend,
    const prnn::matrix::Precision& precision)
{
    auto weights     = prnn::matrix::rand({layerSize, layerSize               }, precision);
    auto activations = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);
    auto deltas      = prnn::matrix::rand({layerSize, miniBatchSize, timesteps}, precision);

    prnn::RecurrentOpsHandle handle(layerSize, miniBatchSize, timesteps, layers,
        prnn::RecurrentRectifiedLinear(),
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        backend);

    auto scratch = prnn::rnn::getBackPropGradientsScratch(handle, precision);
    auto reserve = prnn::createReserveRecurrent(handle, precision);

    prnn::matrix::randn(reserve);

    // warm up
    prnn::rnn::backPropGradientsRecurrent(
        prnn::matrix::DynamicView(weights),
        prnn::matrix::ConstDynamicView(activations),
        prnn::matrix::ConstDynamicView(deltas),
        prnn::matrix::DynamicView(scratch),
        prnn::matrix::DynamicView(reserve),
        handle);
    prnn::parallel::synchronize();

    prnn::util::Timer timer;

    timer.start();

    for (size_t i = 0; i < iterations; ++i) {
        prnn::rnn::backPropGradientsRecurrent(
            prnn::matrix::DynamicView(weights),
            prnn::matrix::ConstDynamicView(activations),
            prnn::matrix::ConstDynamicView(deltas),
            prnn::matrix::DynamicView(scratch),
            prnn::matrix::DynamicView(reserve),
            handle);
    }

    prnn::parallel::synchronize();

    timer.stop();

    double totalFlops = iterations * getFlopCount(handle);

    double teraflops = totalFlops / (timer.seconds() * 1.0e12);
    double microsecondsPerKernel = timer.seconds() * 1.0e6 / iterations;

    std::cout << "RNN Back Propagation Deltas: " << teraflops << " TFLOPS/s\n";
    std::cout << "RNN Average Kernel Time: " << microsecondsPerKernel << " us\n";

}

void runBenchmark(size_t iterations, size_t layerSize, size_t miniBatchSize,
    size_t timesteps, size_t layers, prnn::RecurrentLayerBackend backend,
    prnn::matrix::Precision& precision)
{
    benchmarkRnnForward(  iterations, layerSize, miniBatchSize, timesteps, layers, backend, precision);
    benchmarkRnnReverse(  iterations, layerSize, miniBatchSize, timesteps, layers, backend, precision);
    benchmarkRnnGradients(iterations, layerSize, miniBatchSize, timesteps, layers, backend, precision);
}

prnn::RecurrentLayerBackend getBackend(const std::string& backend)
{
    if(backend == "persistent")
    {
        return prnn::RECURRENT_PERSISTENT_BACKEND;
    }
    else if(backend == "cudnn")
    {
        return prnn::RECURRENT_CUDNN_BACKEND;
    }
    else if(backend == "best")
    {
        return prnn::RECURRENT_BEST_BACKEND;
    }

    throw std::runtime_error("Invalid backend " + backend);
}

int main(int argc, char** argv) {

    prnn::util::ArgumentParser parser(argc, argv);

    prnn::matrix::Precision precision = prnn::matrix::SinglePrecision();

    size_t iterations     = 20;
    size_t layerSize      = prnn::rnn::getMaximumSizeRNNForThisGPU(precision);
    size_t miniBatcheSize = 2;
    size_t timesteps      = 64;
    size_t layers         = 1;

    std::string backend;

    parser.parse("-i", "--iterations",      iterations,     iterations,     "Iterations to run each recurrent operation.");
    parser.parse("-l", "--layer-size",      layerSize,      layerSize,      "The size of the recurrent layer.");
    parser.parse("-b", "--mini-batch-size", miniBatcheSize, miniBatcheSize, "The number of utterances per mini-batch.");
    parser.parse("-t", "--timesteps",       timesteps,      timesteps,      "The length of each utterance.");
    parser.parse("-l", "--layers",          layers,         layers,         "The number of recurrent layers to stack.");
    parser.parse("-b", "--backend",         backend,        backend,        "The backend to use (persistent, cudnn, best).");

    parser.parse();

    prnn::util::enable_log("RecurrentOperations::Detail");

    runBenchmark(iterations, layerSize, miniBatcheSize, timesteps, layers,
        getBackend(backend), precision);
}


