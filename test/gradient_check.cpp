

// Persistent RNN Includes
#include <persistent_rnn_high_level.h>
#include <persistent_rnn.h>

#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/random_operations.h>
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/util/logger.h>
#include <prnn/detail/util/argument_parser.h>
#include <prnn/detail/util/string.h>

// Standard Library Includes
#include <random>
#include <iostream>

static prnn::RecurrentLayerBackend getBackend(const std::string& backend)
{
    if(backend == "persistent")
    {
        return prnn::RECURRENT_PERSISTENT_BACKEND;
    }
    else if(backend == "cudnn")
    {
        return prnn::RECURRENT_CUDNN_BACKEND;
    }

    return prnn::RECURRENT_BEST_BACKEND;
}

class Options
{
public:
    Options() : failed(false), allowNotSupported(false) { }

public:
    size_t layerSize;
    size_t miniBatchSize;
    size_t timesteps;
    size_t layers;

public:
    size_t gradientCheckSamples;

public:
    double epsilon;

public:
    prnn::RecurrentLayerDirection direction;
    prnn::RecurrentLayerType      layerType;
    prnn::RecurrentLayerInputMode inputType;

public:
    std::string backend;

public:
    std::string specificSample;

public:
    double skipConnectionScale;

public:
    bool verbose;

public:
    bool failed;
    bool allowNotSupported;
    bool haltOnFailure;

public:
    bool runSweep;

public:
    std::string toString() const
    {
        return prnn::RecurrentOpsHandle(layerSize,
            miniBatchSize,
            timesteps,
            layers,
            prnn::RecurrentRectifiedLinear(),
            direction,
            layerType,
            inputType,
            getBackend(backend),
            0.0).toString();
    }

};

static prnnBackend_t getBackend(const prnn::RecurrentOpsHandle& handle)
{
    if(handle.backend == prnn::RECURRENT_PERSISTENT_BACKEND)
    {
        return PRNN_PERSISTENT_BACKEND;
    }
    else if(handle.backend == prnn::RECURRENT_CUDNN_BACKEND)
    {
        return PRNN_CUDNN_BACKEND;
    }
    else
    {
        return PRNN_BEST_BACKEND;
    }
}

static prnnRNNInputMode_t getInputType(const prnn::RecurrentLayerInputMode mode)
{
    if(mode == prnn::RECURRENT_LINEAR_INPUT)
    {
        return PRNN_LINEAR_INPUT;
    }
    else
    {
        return PRNN_SKIP_INPUT;
    }
}

static prnnRNNMode_t getLayerType(const prnn::RecurrentLayerType mode)
{
    if(mode == prnn::RECURRENT_SIMPLE_TYPE)
    {
        return PRNN_RNN_RELU;
    }
    else if(mode == prnn::RECURRENT_GRU_TYPE)
    {
        return PRNN_GRU;
    }
    else
    {
        return PRNN_LSTM;
    }
}

static prnnDirectionMode_t getDirection(const prnn::RecurrentLayerDirection direction)
{
    if(direction == prnn::RECURRENT_FORWARD)
    {
        return PRNN_UNIDIRECTIONAL;
    }
    else if(direction == prnn::RECURRENT_BIDIRECTIONAL)
    {
        return PRNN_BIDIRECTIONAL;
    }
    else
    {
        return PRNN_REVERSE;
    }

}

size_t parseSamplePosition(const std::string& position, size_t totalElements)
{
    size_t offset = 0;

    std::stringstream stream;

    stream << prnn::util::removeWhitespace(position);

    stream >> offset;

    offset = offset % totalElements;

    return offset;
}

void TestSimpleRecurrentOps(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::matrix::srand(377);

    size_t layerSize = options.layerSize;
    size_t timesteps  = options.timesteps;
    size_t miniBatchSize = options.miniBatchSize;

    prnn::RecurrentOpsHandle handle(layerSize,
        miniBatchSize,
        timesteps,
        options.layers,
        prnn::RecurrentRectifiedLinear(),
        options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend));

    auto weights     = createWeightsRecurrent(handle, precision);
    auto activations = ones({layerSize, miniBatchSize, timesteps}, precision);
    auto reserve     = createReserveRecurrent(handle, precision);

    forwardPropRecurrent(activations, reserve, weights, handle);

    auto deltas = ones(activations.size(), precision);

    backPropDeltasRecurrent(deltas, weights, activations, reserve, handle);

    auto dWeights = ones(weights.size(), precision);

    backPropGradientsRecurrent(dWeights, activations, deltas, reserve, handle);

    // just make sure nothing crashes
}

void sliceWindow(prnn::matrix::Matrix& inputs, size_t windowSize)
{
    size_t activationCount = inputs.size()[0];
    size_t miniBatchSize   = inputs.size()[1];
    size_t timesteps       = inputs.size()[2];

    size_t activationSize = std::min(windowSize, activationCount);

    inputs = slice(inputs,
        {0,               0,               0},
        {activationSize,  miniBatchSize,   timesteps});
}

prnn::matrix::Matrix extractWindow(prnn::matrix::Matrix inputs, size_t windowSize)
{
    size_t activationCount = inputs.size()[0];
    size_t miniBatchSize   = inputs.size()[1];
    size_t timesteps       = inputs.size()[2];

    size_t activationSize = std::min(windowSize, activationCount);

    auto slicedInputs = slice(inputs,
        {0,               0,               0},
        {activationSize,  miniBatchSize,   timesteps});

    auto zeros = prnn::matrix::zeros(inputs.size(), inputs.precision());

    auto slicedZeros = slice(zeros,
        {0,              0,             0},
        {activationSize, miniBatchSize, timesteps});

    copy(slicedZeros, slicedInputs);

    return zeros;
}

double computeCost(prnn::matrix::Matrix activations, prnn::matrix::Matrix reference,
    size_t windowSize)
{
    sliceWindow(activations, windowSize);
    sliceWindow(reference,   windowSize);

    auto difference = apply(prnn::matrix::Matrix(activations),
        reference, prnn::matrix::Subtract());
    auto squaredDifference = apply(difference, prnn::matrix::Square());

    double squaredSum = reduce(squaredDifference, {}, prnn::matrix::Add())[0];

    return 0.5 * squaredSum / activations.size()[1];
}

prnn::matrix::Matrix computeDeltas(prnn::matrix::Matrix completeActivations,
    prnn::matrix::Matrix completeReference, size_t windowSize)
{
    auto activations = extractWindow(completeActivations, windowSize);
    auto reference   = extractWindow(completeReference,   windowSize);

    size_t miniBatchSize = activations.size()[1];

    return apply(apply(prnn::matrix::Matrix(activations), reference, prnn::matrix::Subtract()),
        prnn::matrix::Multiply(1.0 / miniBatchSize));
}

static void assertLessThanOrEqual(double left, double right)
{
    if(left > right)
    {
        std::stringstream stream;

        stream << "Assertion Failed (" << left << " <= " << right << ")\n";

        throw std::logic_error(stream.str());
    }
}

static void assertGreaterThanOrEqual(double left, double right)
{
    if(left < right)
    {
        std::stringstream stream;

        stream << "Assertion Failed (" << left << " >= " << right << ")\n";

        throw std::logic_error(stream.str());
    }
}

static void assertEqual(double left, double right)
{
    if(left != right)
    {
        std::stringstream stream;

        stream << "Assertion Failed (" << left << " >= " << right << ")\n";

        throw std::logic_error(stream.str());
    }
}

static void assertApproximatelyEqual(const prnn::matrix::Matrix& left,
    const prnn::matrix::Matrix& right)
{
    if(left.size() != right.size())
    {
        throw std::logic_error("Assertion failed matrix size (" + left.size().toString() +
            ") != (" + right.size().toString() + ")");
    }

    if(left.precision() != right.precision())
    {
        throw std::logic_error("Assertion failed matrix precision (" + left.precision().toString() +
            ") != (" + right.precision().toString() + ")");
    }

    auto l = left.begin();
    auto r = right.begin();

    for(; l != left.end(); ++l, ++r)
    {
        double difference = std::abs(*l - *r);

        if(difference > 1.0e-6)
        {
            std::stringstream stream;

            stream << "Assertion Failed matrix element (" << *l << " !~= " << *r << ")\n";

            throw std::logic_error(stream.str());
        }
    }
}

void TestSimpleRecurrentOpsGradientCheck(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    std::default_random_engine randomEngine;

    randomEngine.seed(377);

    prnn::matrix::srand(377);

    size_t layerSize     = options.layerSize;
    size_t timesteps     = options.timesteps;
    size_t miniBatchSize = options.miniBatchSize;
    size_t samples       = options.gradientCheckSamples;

    prnn::RecurrentOpsHandle handle(layerSize, miniBatchSize, timesteps, options.layers,
        prnn::RecurrentRectifiedLinear(), options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend),
        options.skipConnectionScale);

    auto weights = createWeightsRecurrent(handle, precision);
    rand(weights);
    apply(weights, weights, prnn::matrix::Multiply(2.0/(timesteps*layerSize)));

    samples = std::min(weights.elements(), samples);

    auto inputActivations = rand({layerSize, miniBatchSize, timesteps}, precision);
    auto reserve = createReserveRecurrent(handle, precision);

    auto referenceActivations = rand({layerSize, miniBatchSize, timesteps}, precision);

    auto outputActivations = copy(inputActivations);

    prnn::util::log("TestRecurrent") << "Input Weights     " << weights.debugString();
    prnn::util::log("TestRecurrent") << "Input Activations " << outputActivations.debugString();

    forwardPropRecurrent(outputActivations, reserve, weights, handle);

    prnn::util::log("TestRecurrent") << "Output Activations " << outputActivations.debugString();

    prnn::util::log("TestRecurrent") << "Reference Activations "
        << referenceActivations.debugString();

    double cost = computeCost  (outputActivations, referenceActivations, layerSize);
    auto deltas = computeDeltas(outputActivations, referenceActivations, layerSize);

    prnn::util::log("TestRecurrent") << "Input Deltas " << deltas.debugString();

    backPropDeltasRecurrent(deltas, weights, outputActivations, reserve, handle);

    auto dWeights = zeros(weights.size(), precision);

    backPropGradientsRecurrent(dWeights, inputActivations, outputActivations, reserve, handle);

    prnn::util::log("TestRecurrent") << "Output Deltas " << deltas.debugString();
    prnn::util::log("TestRecurrent") << "dWeights      " << dWeights.debugString();

    size_t gradientCount = weights.elements();

    std::vector<size_t> samplePositions = {parseSamplePosition(options.specificSample,
        gradientCount)};

    while (samplePositions.size() < samples) {
        samplePositions.push_back(randomEngine() % gradientCount);
    }

    // double sided grad check over the sample positions
    double epsilon    = options.epsilon;
    double difference = 0.0;
    double total      = 0.0;

    for (auto& samplePosition : samplePositions) {

        double originalValue = weights(samplePosition);
        double gradient      = dWeights(samplePosition);

        weights(samplePosition) = originalValue - epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << samplePosition
            << ")     from " << originalValue << " to "
            << (originalValue - epsilon) << "\n";

        auto copiedOutputActivations = copy(inputActivations);
        forwardPropRecurrent(copiedOutputActivations, reserve, weights, handle);

        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            copiedOutputActivations.debugString();

        double leftCost = computeCost(copiedOutputActivations, referenceActivations,
            layerSize);

        weights(samplePosition) = originalValue + epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << samplePosition
            << ")     from " << originalValue << " to "
            << (originalValue + epsilon) << "\n";

        copiedOutputActivations = copy(inputActivations);

        forwardPropRecurrent(copiedOutputActivations, reserve, weights, handle);
        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            copiedOutputActivations.debugString();

        double rightCost = computeCost(copiedOutputActivations, referenceActivations,
            layerSize);

        weights(samplePosition) = originalValue;

        double numericalGradient = (rightCost - leftCost) / (2.0 * epsilon);
        double localDifference = numericalGradient - gradient;

        difference += std::pow(localDifference, 2.0);

        double absoluteDifference = (localDifference == 0 || gradient == 0) ?
            0.0 : std::abs(localDifference) / std::abs(gradient);

        total += std::pow(gradient, 2.0);

        double scaledDifference = (total == 0.0) ? difference : (difference / total);

        if (absoluteDifference > 1e-3 || !std::isfinite(localDifference))
        {
            prnn::util::log("TestRecurrent") << "For weight (" << samplePosition
                << ") computed gradient " << gradient << " does not match estimated gradient "
                << numericalGradient << ", cost " << cost << " left cost "
                << leftCost << ", right cost " << rightCost <<  ", " << localDifference
                << " difference, " << scaledDifference<< " scaled difference\n";
        }
        else
        {
            prnn::util::log("TestRecurrent") << "For weight (" << samplePosition
                << ") computed gradient " << gradient << " matches estimated gradient "
                << numericalGradient << "\n";
        }
    }

    difference = (difference == 0.0 && total == 0.0) ? 0.0 : (difference / total);

    assertLessThanOrEqual(    difference, 2e-1 );
    assertGreaterThanOrEqual( difference, 1e-16);
}

void TestRecurrentOpsGradientCheck(const Options& options)
{
    Options newOptions = options;

    newOptions.direction = prnn::RECURRENT_FORWARD;

    TestSimpleRecurrentOpsGradientCheck(newOptions);
}

void TestReverseRecurrentOpsGradientCheck(const Options& options)
{
    Options newOptions = options;

    newOptions.direction = prnn::RECURRENT_FORWARD;

    TestSimpleRecurrentOpsGradientCheck(newOptions);
}

static void assertSuccess(prnnStatus_t status)
{
    if(status == PRNN_STATUS_NOT_SUPPORTED)
    {
        throw prnn::NotSupported();
    }

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::logic_error(std::string("PRNN C interface call returned error: '") +
            prnnGetErrorString(status) + "'");
    }
}

void TestCTensor(const Options& options)
{
    prnnTensorDescriptor_t descriptor;

    assertSuccess(prnnCreateTensorDescriptor(&descriptor));

    const int inputDimensions[3] = {static_cast<int>(options.layerSize),
                                    static_cast<int>(options.miniBatchSize),
                                    static_cast<int>(options.timesteps)};

    const int inputStrides[3]    = {static_cast<int>(1),
                                    static_cast<int>(options.layerSize),
                                    static_cast<int>(options.layerSize * options.miniBatchSize)};

    assertSuccess(prnnSetTensorNdDescriptor(descriptor,
                                            PRNN_DATA_FLOAT,
                                            3,
                                            inputDimensions,
                                            inputStrides));

    int numberOfDimensions = 0;
    int dimensions[PRNN_DIM_MAX];
    int strides[PRNN_DIM_MAX];
    prnnDataType_t dataType = PRNN_INVALID_DATA;

    assertSuccess(prnnGetTensorNdDescriptor(descriptor,
                                            PRNN_DIM_MAX,
                                            &dataType,
                                            &numberOfDimensions,
                                            dimensions,
                                            strides));

    assertEqual(numberOfDimensions, 3);

    assertEqual(dimensions[0], options.layerSize);
    assertEqual(dimensions[1], options.miniBatchSize);
    assertEqual(dimensions[2], options.timesteps);

    assertEqual(strides[0], 1);
    assertEqual(strides[1], options.layerSize);
    assertEqual(strides[2], options.layerSize * options.miniBatchSize);

    assertSuccess(prnnDestroyTensorDescriptor(descriptor));
}

void TestCRNN(const Options& options)
{
    prnnHandle_t handle;
    prnnRNNDescriptor_t descriptor;
    std::vector<prnnTensorDescriptor_t> inputDescriptors;

    prnn::RecurrentOpsHandle highLevelHandle(options.layerSize,
        options.miniBatchSize,
        options.timesteps,
        options.layers,
        prnn::RecurrentRectifiedLinear(),
        options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend),
        0.0);

    assertSuccess(prnnCreate(&handle));

    const int inputDimensions[3] = {static_cast<int>(options.miniBatchSize),
                                    static_cast<int>(options.layerSize),
                                    static_cast<int>(1)};

    const int inputStrides[3]    = {static_cast<int>(options.layerSize),
                                    static_cast<int>(1),
                                    static_cast<int>(1)};

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        inputDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&inputDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(inputDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       options.layerSize,
                                       options.layers,
                                       nullptr,
                                       getInputType(options.inputType),
                                       getDirection(options.direction),
                                       getLayerType(options.layerType),
                                       PRNN_DATA_FLOAT,
                                       getBackend(highLevelHandle)));

    size_t reserveSize = 0;

    assertSuccess(prnnGetRNNTrainingReserveSize(handle,
                                                descriptor,
                                                options.timesteps,
                                                inputDescriptors.data(),
                                                &reserveSize));

    assertGreaterThanOrEqual(reserveSize, sizeof(float));

    size_t parameterSize = 0;

    assertSuccess(prnnGetRNNParamsSize(handle,
                                       descriptor,
                                       inputDescriptors.data(),
                                       &parameterSize));

    assertGreaterThanOrEqual(parameterSize,
        sizeof(float) * options.layerSize * options.layerSize);

    size_t workspaceSize = 0;

    assertSuccess(prnnGetRNNWorkspaceSize(handle,
                                          descriptor,
                                          options.timesteps,
                                          inputDescriptors.data(),
                                          &workspaceSize));

    assertGreaterThanOrEqual(workspaceSize, sizeof(float));

    for(auto& inputDescriptor : inputDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(inputDescriptor));
    }

    assertSuccess(prnnDestroyRNNDescriptor(descriptor));
    assertSuccess(prnnDestroy(handle));
}

static prnn::matrix::Matrix createWeightsWithCInterface(
    const prnn::RecurrentOpsHandle& highLevelHandle,
    const prnn::matrix::Precision& precision)
{
    prnnHandle_t handle;

    assertSuccess(prnnCreate(&handle));

    prnnRNNDescriptor_t descriptor;

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       highLevelHandle.layerSize,
                                       highLevelHandle.layers,
                                       nullptr,
                                       getInputType(highLevelHandle.inputMode),
                                       getDirection(highLevelHandle.direction),
                                       getLayerType(highLevelHandle.layerType),
                                       PRNN_DATA_FLOAT,
                                       getBackend(highLevelHandle)));

    std::vector<prnnTensorDescriptor_t> inputActivationsDescriptors;

    const int inputDimensions[3] = {static_cast<int>(highLevelHandle.miniBatchSize),
                                    static_cast<int>(highLevelHandle.layerSize),
                                    static_cast<int>(1)};

    const int inputStrides[3]    = {static_cast<int>(highLevelHandle.layerSize),
                                    static_cast<int>(1),
                                    static_cast<int>(1)};

    for(size_t i = 0; i < highLevelHandle.timesteps; ++i)
    {
        inputActivationsDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&inputActivationsDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(inputActivationsDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    size_t size = 0;

    assertSuccess(prnnGetRNNParamsSize(handle,
                                       descriptor,
                                       inputActivationsDescriptors.data(),
                                       &size));

    for(auto& inputDescriptor : inputActivationsDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(inputDescriptor));
    }

    assertSuccess(prnnDestroyRNNDescriptor(descriptor));
    assertSuccess(prnnDestroy(handle));

    return prnn::matrix::randn({size / precision.size(), 1, 1}, precision);
}

void TestCForwardOps(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::RecurrentOpsHandle highLevelHandle(options.layerSize,
        options.miniBatchSize,
        options.timesteps,
        options.layers,
        prnn::RecurrentRectifiedLinear(),
        options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend),
        0.0);

    auto weights = createWeightsWithCInterface(highLevelHandle, precision);

    apply(weights, weights, prnn::matrix::Multiply(1.0e-2));

    auto inputActivations = rand({options.layerSize, options.miniBatchSize, options.timesteps, 1},
        precision);
    auto highLevelReserve = prnn::createReserveRecurrent(highLevelHandle, precision);

    auto referenceActivations = copy(inputActivations);

    forwardPropRecurrent(referenceActivations, highLevelReserve, weights, highLevelHandle);

    prnnHandle_t handle;

    assertSuccess(prnnCreate(&handle));

    prnnRNNDescriptor_t descriptor;

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       options.layerSize,
                                       options.layers,
                                       nullptr,
                                       getInputType(options.inputType),
                                       getDirection(options.direction),
                                       getLayerType(options.layerType),
                                       PRNN_DATA_FLOAT,
                                       getBackend(highLevelHandle)));

    std::vector<prnnTensorDescriptor_t> inputActivationsDescriptors;

    const int inputDimensions[3] = {static_cast<int>(options.miniBatchSize),
                                    static_cast<int>(options.layerSize),
                                    static_cast<int>(1)};

    const int inputStrides[3]    = {static_cast<int>(options.layerSize),
                                    static_cast<int>(1),
                                    static_cast<int>(1)};

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        inputActivationsDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&inputActivationsDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(inputActivationsDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    size_t reserveSize = 0;

    assertSuccess(prnnGetRNNTrainingReserveSize(handle,
                                                descriptor,
                                                options.timesteps,
                                                inputActivationsDescriptors.data(),
                                                &reserveSize));

    size_t workspaceSize = 0;

    assertSuccess(prnnGetRNNWorkspaceSize(handle,
                                          descriptor,
                                          options.timesteps,
                                          inputActivationsDescriptors.data(),
                                          &workspaceSize));

    size_t weightsSize = 0;

    assertSuccess(prnnGetRNNParamsSize(handle,
                                       descriptor,
                                       inputActivationsDescriptors.data(),
                                       &weightsSize));

    prnn::matrix::Matrix workspace({workspaceSize / precision.size()}, precision);
    prnn::matrix::Matrix reserve({reserveSize / precision.size()}, precision);

    prnnTensorDescriptor_t weightsDescriptor;

    assertSuccess(prnnCreateTensorDescriptor(&weightsDescriptor));

    const int weightsDimensions[3] = {static_cast<int>(weightsSize / precision.size()), 1, 1};

    const int weightsStrides[3]    = {1, weightsDimensions[0], weightsDimensions[0]};

    assertSuccess(prnnSetTensorNdDescriptor(weightsDescriptor,
                                            PRNN_DATA_FLOAT,
                                            3,
                                            weightsDimensions,
                                            weightsStrides));

    assertSuccess(prnnRNNForward(handle,
                                 descriptor,
                                 highLevelHandle.timesteps,
                                 inputActivationsDescriptors.data(),
                                 inputActivations.data(),
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 weightsDescriptor,
                                 weights.data(),
                                 inputActivationsDescriptors.data(),
                                 inputActivations.data(),
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 workspace.data(),
                                 workspaceSize,
                                 reserve.data(),
                                 reserveSize));

    assertApproximatelyEqual(inputActivations, referenceActivations);

    assertSuccess(prnnDestroyTensorDescriptor(weightsDescriptor));

    for(auto& inputDescriptor : inputActivationsDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(inputDescriptor));
    }

    assertSuccess(prnnDestroyRNNDescriptor(descriptor));
    assertSuccess(prnnDestroy(handle));
}

void TestCDeltaOps(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::RecurrentOpsHandle highLevelHandle(options.layerSize,
        options.miniBatchSize,
        options.timesteps,
        options.layers,
        prnn::RecurrentRectifiedLinear(),
        options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend),
        0.0);


    auto weights = createWeightsWithCInterface(highLevelHandle, precision);

    apply(weights, weights, prnn::matrix::Multiply(1.0e-2));

    auto inputActivations = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto deltas = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto highLevelReserve = prnn::createReserveRecurrent(highLevelHandle, precision);

    zeros(highLevelReserve);
    auto referenceDeltas = copy(deltas);

    backPropDeltasRecurrent(referenceDeltas, weights, inputActivations, highLevelReserve,
        highLevelHandle);

    prnnHandle_t handle;

    assertSuccess(prnnCreate(&handle));

    prnnRNNDescriptor_t descriptor;

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       options.layerSize,
                                       options.layers,
                                       nullptr,
                                       getInputType(options.inputType),
                                       getDirection(options.direction),
                                       getLayerType(options.layerType),
                                       PRNN_DATA_FLOAT,
                                       getBackend(highLevelHandle)));

    std::vector<prnnTensorDescriptor_t> inputActivationsDescriptors;

    const int inputDimensions[3] = {static_cast<int>(options.miniBatchSize),
                                    static_cast<int>(options.layerSize),
                                    static_cast<int>(1)};

    const int inputStrides[3]    = {static_cast<int>(options.layerSize),
                                    static_cast<int>(1),
                                    static_cast<int>(1)};

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        inputActivationsDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&inputActivationsDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(inputActivationsDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    std::vector<prnnTensorDescriptor_t> deltasDescriptors;

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        deltasDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&deltasDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(deltasDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    size_t reserveSize = 0;

    assertSuccess(prnnGetRNNTrainingReserveSize(handle,
                                                descriptor,
                                                options.timesteps,
                                                inputActivationsDescriptors.data(),
                                                &reserveSize));

    size_t workspaceSize = 0;

    assertSuccess(prnnGetRNNWorkspaceSize(handle,
                                          descriptor,
                                          options.timesteps,
                                          inputActivationsDescriptors.data(),
                                          &workspaceSize));

    size_t weightsSize = 0;

    assertSuccess(prnnGetRNNParamsSize(handle,
                                       descriptor,
                                       inputActivationsDescriptors.data(),
                                       &weightsSize));

    prnn::matrix::Matrix workspace({workspaceSize / precision.size()}, precision);
    prnn::matrix::Matrix reserve({reserveSize / precision.size()}, precision);

    zeros(reserve);

    prnnTensorDescriptor_t weightsDescriptor;

    assertSuccess(prnnCreateTensorDescriptor(&weightsDescriptor));

    const int weightsDimensions[3] = {static_cast<int>(weightsSize / precision.size()), 1, 1};

    const int weightsStrides[3]    = {1, weightsDimensions[0], weightsDimensions[0]};

    assertSuccess(prnnSetTensorNdDescriptor(weightsDescriptor,
                                            PRNN_DATA_FLOAT,
                                            3,
                                            weightsDimensions,
                                            weightsStrides));

    auto outputDeltas = copy(deltas);

    assertSuccess(prnnRNNBackwardData(handle,
                                      descriptor,
                                      options.timesteps,
                                      inputActivationsDescriptors.data(),
                                      inputActivations.data(),
                                      deltasDescriptors.data(),
                                      outputDeltas.data(),
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      weightsDescriptor,
                                      weights.data(),
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      deltasDescriptors.data(),
                                      deltas.data(),
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      workspace.data(),
                                      workspaceSize,
                                      reserve.data(),
                                      reserveSize));

    assertApproximatelyEqual(referenceDeltas, deltas);

    for(auto& inputDescriptor : inputActivationsDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(inputDescriptor));
    }

    for(auto& deltasDescriptor : deltasDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(deltasDescriptor));
    }

    assertSuccess(prnnDestroyTensorDescriptor(weightsDescriptor));

    assertSuccess(prnnDestroyRNNDescriptor(descriptor));
    assertSuccess(prnnDestroy(handle));
}

void TestCGradientOps(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::RecurrentOpsHandle highLevelHandle(options.layerSize,
        options.miniBatchSize,
        options.timesteps,
        options.layers,
        prnn::RecurrentRectifiedLinear(),
        options.direction,
        options.layerType,
        options.inputType,
        getBackend(options.backend),
        0.0);

    auto dWeights = createWeightsWithCInterface(highLevelHandle, precision);

    auto inputActivations = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto outputActivations = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto highLevelReserve = prnn::createReserveRecurrent(highLevelHandle, precision);

    rand(highLevelReserve);

    auto referenceDWeights = copy(dWeights);

    backPropGradientsRecurrent(referenceDWeights, inputActivations, outputActivations,
        highLevelReserve, highLevelHandle);

    prnnHandle_t handle;

    assertSuccess(prnnCreate(&handle));

    prnnRNNDescriptor_t descriptor;

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       options.layerSize,
                                       options.layers,
                                       nullptr,
                                       getInputType(options.inputType),
                                       getDirection(options.direction),
                                       getLayerType(options.layerType),
                                       PRNN_DATA_FLOAT,
                                       getBackend(highLevelHandle)));

    std::vector<prnnTensorDescriptor_t> inputActivationsDescriptors;

    const int inputDimensions[3] = {static_cast<int>(options.miniBatchSize),
                                    static_cast<int>(options.layerSize),
                                    static_cast<int>(1)};

    const int inputStrides[3]    = {static_cast<int>(options.layerSize),
                                    static_cast<int>(1),
                                    static_cast<int>(1)};

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        inputActivationsDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&inputActivationsDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(inputActivationsDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    std::vector<prnnTensorDescriptor_t> outputActivationsDescriptors;

    for(size_t i = 0; i < options.timesteps; ++i)
    {
        outputActivationsDescriptors.push_back(nullptr);

        assertSuccess(prnnCreateTensorDescriptor(&outputActivationsDescriptors.back()));
        assertSuccess(prnnSetTensorNdDescriptor(outputActivationsDescriptors.back(),
                                                PRNN_DATA_FLOAT,
                                                3,
                                                inputDimensions,
                                                inputStrides));
    }

    size_t reserveSize = 0;

    assertSuccess(prnnGetRNNTrainingReserveSize(handle,
                                                descriptor,
                                                options.timesteps,
                                                inputActivationsDescriptors.data(),
                                                &reserveSize));

    size_t workspaceSize = 0;

    assertSuccess(prnnGetRNNWorkspaceSize(handle,
                                          descriptor,
                                          options.timesteps,
                                          inputActivationsDescriptors.data(),
                                          &workspaceSize));

    size_t weightsSize = 0;

    assertSuccess(prnnGetRNNParamsSize(handle,
                                       descriptor,
                                       inputActivationsDescriptors.data(),
                                       &weightsSize));


    prnn::matrix::Matrix workspace({workspaceSize / precision.size()}, precision);
    prnn::matrix::Matrix reserve({reserveSize / precision.size()}, precision);

    assertEqual(reserve.size().product(), highLevelReserve.size().product());

    copy(reserve, highLevelReserve);

    prnnTensorDescriptor_t dWeightsDescriptor;

    assertSuccess(prnnCreateTensorDescriptor(&dWeightsDescriptor));

    const int weightsDimensions[3] = {static_cast<int>(weightsSize / precision.size()), 1, 1};

    const int weightsStrides[3]    = {1, weightsDimensions[0], weightsDimensions[0]};

    assertSuccess(prnnSetTensorNdDescriptor(dWeightsDescriptor,
                                            PRNN_DATA_FLOAT,
                                            3,
                                            weightsDimensions,
                                            weightsStrides));

    assertSuccess(prnnRNNBackwardWeights(handle,
                                         descriptor,
                                         options.timesteps,
                                         inputActivationsDescriptors.data(),
                                         inputActivations.data(),
                                         nullptr,
                                         nullptr,
                                         outputActivationsDescriptors.data(),
                                         outputActivations.data(),
                                         workspace.data(),
                                         workspaceSize,
                                         dWeightsDescriptor,
                                         dWeights.data(),
                                         reserve.data(),
                                         reserveSize));

    assertApproximatelyEqual(referenceDWeights, dWeights);

    for(auto& inputDescriptor : inputActivationsDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(inputDescriptor));
    }

    for(auto& outputActivationDescriptor : outputActivationsDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(outputActivationDescriptor));
    }

    assertSuccess(prnnDestroyTensorDescriptor(dWeightsDescriptor));

    assertSuccess(prnnDestroyRNNDescriptor(descriptor));
    assertSuccess(prnnDestroy(handle));
}

void RunTest(const std::string& testName, void (*function)(const Options& options),
    Options& options)
{
    if(options.failed && options.haltOnFailure)
    {
        return;
    }

    try
    {
        function(options);
        std::cout << "Test '" << testName << "' Passed\n";
    }
    catch(prnn::NotSupported& e)
    {
        if(options.allowNotSupported)
        {
            std::cout << "Test '" << testName << "' is known to be not supported.\n";
        }
        else
        {
            std::cout << "Test '" << testName << "' is supposed to be supported.\n";
            options.failed = true;
        }
    }
    catch(std::exception& e)
    {
        std::cout << "Test '" << testName << "' Failed with error '" << e.what() << "'\n";
        options.failed = true;
    }
}

bool isPersistentBackendSelected(const Options& options)
{
    return prnn::RECURRENT_PERSISTENT_BACKEND ==
        prnn::getBackend(prnn::RecurrentOpsHandle(options.layerSize,
            options.miniBatchSize,
            options.timesteps,
            options.layers,
            prnn::RecurrentRectifiedLinear(),
            options.direction,
            options.layerType,
            options.inputType,
            getBackend(options.backend),
            0.0), prnn::matrix::SinglePrecision());
}

bool isCudnnSupported(const Options& options)
{
    return prnn::isCudnnBackendSupported(prnn::RecurrentOpsHandle(options.layerSize,
            options.miniBatchSize,
            options.timesteps,
            options.layers,
            prnn::RecurrentRectifiedLinear(),
            options.direction,
            options.layerType,
            options.inputType,
            getBackend(options.backend),
            0.0), prnn::matrix::SinglePrecision());
}

std::vector<Options> getSweepRange(const Options& initialOptions)
{
    if(!initialOptions.runSweep)
    {
        return {initialOptions};
    }

    std::vector<Options> range;

    auto layerSizes = {512, 2, 192};
    auto miniBatchSizes = {7, 1};
    auto timesteps = {2, 13};
    auto inputTypes = {prnn::RECURRENT_SKIP_INPUT, prnn::RECURRENT_LINEAR_INPUT};
    auto directions = {prnn::RECURRENT_FORWARD};
    auto scales = {0.0, 0.5};
    auto layerTypes = {prnn::RECURRENT_LSTM_TYPE, prnn::RECURRENT_GRU_TYPE,
        prnn::RECURRENT_SIMPLE_TYPE};

    Options options = initialOptions;

    for(auto& layerType : layerTypes)
    {
        options.layerType = layerType;

        for(auto& direction : directions)
        {
            options.direction = direction;

            for(auto& inputType : inputTypes)
            {
                options.inputType = inputType;

                for(auto& layerSize : layerSizes)
                {
                    options.layerSize = layerSize;

                    for(auto& timestep : timesteps)
                    {
                        options.timesteps = timestep;

                        for(auto& miniBatchSize : miniBatchSizes)
                        {
                            options.miniBatchSize = miniBatchSize;

                            for(auto& scale : scales)
                            {
                                options.skipConnectionScale = scale;

                                options.allowNotSupported = false;

                                if(!isCudnnSupported(options) &&
                                    layerType != prnn::RECURRENT_SIMPLE_TYPE)
                                {
                                    options.allowNotSupported = true;
                                }

                                if(inputType == prnn::RECURRENT_SKIP_INPUT &&
                                    !isPersistentBackendSelected(options))
                                {
                                    break;
                                }

                                if(inputType == prnn::RECURRENT_LINEAR_INPUT &&
                                    scale != 0.0)
                                {
                                    break;
                                }

                                if(inputType == prnn::RECURRENT_LINEAR_INPUT)
                                {
                                    options.allowNotSupported = true;
                                }


                                range.push_back(options);
                                range.back().backend = "best";

                                options.allowNotSupported = true;

                                range.push_back(options);
                                range.back().backend = "cudnn";

                                range.push_back(options);
                                range.back().backend = "persistent";

                                if(layerType != prnn::RECURRENT_SIMPLE_TYPE)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return range;
}

void runSweep(const Options& initialOptions)
{
    auto allOptions = getSweepRange(initialOptions);

    for(auto& options : allOptions)
    {
        std::cout << "Running tests for " << options.toString() << "\n";

        RunTest("C Interface Tensor Test",       TestCTensor,                   options);
        RunTest("C Interface RNN Test",          TestCRNN,                      options);
        RunTest("C Interface Forward Ops Test",  TestCForwardOps,               options);
        RunTest("C Interface Delta Ops Test",    TestCDeltaOps,                 options);
        RunTest("C Interface Gradient Ops Test", TestCGradientOps,              options);
        RunTest("Simple Recurrent Ops Test",     TestSimpleRecurrentOps,        options);
        RunTest("Recurrent Ops Gradient Check",  TestRecurrentOpsGradientCheck, options);

        if(options.failed && options.haltOnFailure)
        {
            return;
        }
    }

    std::cout << "All Tests Passed\n";
}

int main(int argc, char** argv)
{
    prnn::util::ArgumentParser parser(argc, argv);

    auto precision = prnn::matrix::SinglePrecision();

    Options options;

    options.layerSize     = prnn::rnn::getMaximumSizeRNNForThisGPU(precision);
    options.miniBatchSize = 3;
    options.timesteps     = 10;
    options.layers        = 1;
    options.verbose       = false;
    options.epsilon       = 1.0e-3;
    options.inputType     = prnn::RECURRENT_SKIP_INPUT;
    options.layerType     = prnn::RECURRENT_SIMPLE_TYPE;
    options.direction     = prnn::RECURRENT_FORWARD;

    options.gradientCheckSamples = 8;

    options.haltOnFailure = true;

    options.specificSample = "0";
    options.skipConnectionScale = 0.5;

    options.backend = "best";

    options.runSweep = true;

    parser.parse("-t", "--timeteps",   options.timesteps,     options.timesteps,     "The number of timesteps to run the RNN for.");
    parser.parse("-b", "--mini-batch", options.miniBatchSize, options.miniBatchSize, "The mini-batch size to run through the layer.");
    parser.parse("-l", "--layer-size", options.layerSize,     options.layerSize,     "The size of the RNN layer to operate on.");
    parser.parse("-n", "--layers",     options.layers,        options.layers,        "The number of RNN layers to stack.");
    parser.parse("-e", "--epsilon",    options.epsilon,       options.epsilon,       "Epsilon used for the gradient check.");

    parser.parse("", "--backend", options.backend, options.backend, "Select the backend to test.");

    parser.parse("", "--specific-sample", options.specificSample, options.specificSample, "Specific weight to sample.");
    parser.parse("", "--skip-connection-scale", options.skipConnectionScale, options.skipConnectionScale, "Scaling factor for skip connections.");

    parser.parse("-s", "--grad-check-samples", options.gradientCheckSamples,
        options.gradientCheckSamples, "The number of weights to perform gradient check on.");

    parser.parse("", "--halt-on-failure", options.haltOnFailure,
        options.haltOnFailure, "Abort on the first test failure.");

    parser.parse("", "--no-sweep", options.runSweep,
        options.runSweep, "Don't run a sweep over all tests.");

    parser.parse("-v", "--verbose", options.verbose, options.verbose, "Enable all PRNN logs.");

    parser.parse();

    if(options.verbose)
    {
        prnn::util::enable_all_logs();
    }

    runSweep(options);

    return 0;
}


