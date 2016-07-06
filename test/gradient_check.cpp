

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

class Options
{
public:
    Options() : failed(false) { }

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

public:
    std::string specificSample;

public:
    bool usePersistentBackProp;
    bool usePersistentForwardProp;

public:
    double skipConnectionScale;

public:
    bool verbose;

public:
    bool failed;
    bool haltOnFailure;
    bool useCudnn;

};

static prnnBackend_t getBackend(const prnn::RecurrentOpsHandle& handle)
{
    if(handle.useCudnn)
    {
        return PRNN_CUDNN_BACKEND;
    }
    else if(handle.allowPersistentKernels)
    {
        return PRNN_PERSISTENT_BACKEND;
    }
    else
    {
        return PRNN_GEMM_BACKEND;
    }
}

size_t parseSamplePosition(const std::string& position, size_t rows, size_t columns)
{
    size_t row = 0;
    size_t column = 0;

    auto coordinates = prnn::util::split(position, ",");

    if(coordinates.size() > 0)
    {
        std::stringstream stream;

        stream << prnn::util::removeWhitespace(coordinates[0]);

        stream >> row;

        row = row % rows;
    }

    if(coordinates.size() > 1)
    {
        std::stringstream stream;

        stream << prnn::util::removeWhitespace(coordinates[1]);

        stream >> column;

        columns = column % columns;
    }

    return row + column * rows;
}


void TestSimpleRecurrentOps(const Options& options)
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::matrix::srand(377);

    size_t layer_size = options.layerSize;
    size_t timesteps  = options.timesteps;
    size_t mini_batch = options.miniBatchSize;

    prnn::RecurrentOpsHandle handle(layer_size, mini_batch, timesteps, options.layers,
        prnn::RecurrentRectifiedLinear());

    auto weights     = ones({layer_size, layer_size}, precision);
    auto activations = ones({layer_size, mini_batch, timesteps}, precision);
    auto reserve     = createReserveRecurrent(handle, precision);

    forwardPropRecurrent(activations, reserve, weights, handle);

    auto deltas = ones(activations.size(), precision);

    backPropDeltasRecurrent(deltas, weights, activations, reserve, handle);

    auto dWeights = ones(weights.size(), precision);

    backPropGradientsRecurrent(dWeights, activations, deltas, reserve, handle);

    // just make sure nothing crashes
}

void slice_window(prnn::matrix::Matrix& inputs, size_t window_size)
{
    size_t activation_count = inputs.size()[0];
    size_t mini_batch_size  = inputs.size()[1];
    size_t timesteps        = inputs.size()[2];

    size_t activation_size = std::min(window_size, activation_count);

    inputs = slice(inputs,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});
}

prnn::matrix::Matrix extract_window(prnn::matrix::Matrix inputs, size_t window_size)
{
    size_t activation_count = inputs.size()[0];
    size_t mini_batch_size  = inputs.size()[1];
    size_t timesteps        = inputs.size()[2];

    size_t activation_size = std::min(window_size, activation_count);

    auto sliced_inputs = slice(inputs,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});

    auto zeros = prnn::matrix::zeros(inputs.size(), inputs.precision());

    auto sliced_zeros = slice(zeros,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});

    copy(sliced_zeros, sliced_inputs);

    return zeros;
}

double compute_cost(prnn::matrix::Matrix activations, prnn::matrix::Matrix reference,
    size_t window_size)
{
    slice_window(activations, window_size);
    slice_window(reference,   window_size);

    auto difference = apply(prnn::matrix::Matrix(activations),
        reference, prnn::matrix::Subtract());
    auto squaredDifference = apply(difference, prnn::matrix::Square());

    double squaredSum = reduce(squaredDifference, {}, prnn::matrix::Add())[0];

    return 0.5 * squaredSum / activations.size()[1];
}

prnn::matrix::Matrix compute_deltas(prnn::matrix::Matrix complete_activations,
    prnn::matrix::Matrix complete_reference, size_t window_size)
{
    auto activations = extract_window(complete_activations, window_size);
    auto reference   = extract_window(complete_reference,   window_size);

    size_t mini_batch = activations.size()[1];

    return apply(apply(prnn::matrix::Matrix(activations), reference, prnn::matrix::Subtract()),
        prnn::matrix::Multiply(1.0 / mini_batch));
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

    std::default_random_engine random_engine;

    random_engine.seed(377);

    prnn::matrix::srand(377);

    size_t layer_size = options.layerSize;
    size_t timesteps  = options.timesteps;
    size_t mini_batch = options.miniBatchSize;
    size_t samples    = options.gradientCheckSamples;

    size_t window_rows    = layer_size;
    size_t window_columns = window_rows;
    size_t window_outputs = window_rows;

    samples = std::min(window_rows * window_columns, samples);

    prnn::RecurrentOpsHandle handle(layer_size, mini_batch, timesteps, options.layers,
        prnn::RecurrentRectifiedLinear(), options.direction,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        options.usePersistentForwardProp,
        options.useCudnn,
        options.skipConnectionScale);

    auto weights = createWeightsRecurrent(handle, precision);
    zeros(weights);

    auto layerWeights = reshape(sliceLayerWeights(weights, handle, 2), {layer_size, layer_size});

    auto weights_slice = slice(layerWeights, {0, 0}, {window_rows, window_columns});

    copy(weights_slice, rand({window_rows, window_columns}, precision));

    apply(weights_slice, weights_slice, prnn::matrix::Multiply(1.0e-2));

    auto input_activations = zeros({layer_size, mini_batch, timesteps}, precision);
    auto reserve = createReserveRecurrent(handle, precision);

    auto reference_activations = zeros({layer_size, mini_batch, timesteps}, precision);

    copy(
        slice(input_activations,
                     {0, 0, 0},
                     {window_outputs, mini_batch, timesteps}),
        rand({window_outputs, mini_batch, timesteps}, precision));

    copy(
        slice(reference_activations,
                    {0, 0, 0},
                    {window_outputs, mini_batch, timesteps}),
        rand({window_outputs, mini_batch, timesteps}, precision));

    auto output_activations = copy(input_activations);

    prnn::util::log("TestRecurrent") << "Input Weights     " << weights_slice.debugString();
    prnn::util::log("TestRecurrent") << "Input Activations " <<
        reshape(copy(slice(output_activations,
                             {0,0,0},
                             {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();

    forwardPropRecurrent(output_activations, reserve, weights, handle);

    prnn::util::log("TestRecurrent") << "Output Activations " <<
        reshape(copy(slice(output_activations,
                           {0,0,0},
                           {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();

    prnn::util::log("TestRecurrent") << "Reference Activations " <<
        reshape(copy(slice(reference_activations,
                           {0,0,0},
                           {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();

    double cost = compute_cost  (output_activations, reference_activations, window_outputs);
    auto deltas = compute_deltas(output_activations, reference_activations, window_outputs);

    prnn::util::log("TestRecurrent") << "Input Deltas " <<
        reshape(copy(slice(deltas,
                           {0,0,0},
                           {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();

    handle.allowPersistentKernels = options.usePersistentBackProp;

    backPropDeltasRecurrent(deltas, weights, output_activations, reserve, handle);

    auto dWeights = zeros(weights.size(), precision);
    auto layerDWeights = reshape(sliceLayerWeights(dWeights, handle, 2), {layer_size, layer_size});

    backPropGradientsRecurrent(dWeights, input_activations, output_activations, reserve, handle);

    handle.allowPersistentKernels = options.usePersistentForwardProp;

    prnn::util::log("TestRecurrent") << "Output Deltas " <<
        reshape(copy(slice(deltas,
                           {0,0,0},
                           {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();
    prnn::util::log("TestRecurrent") << "dWeights      " << layerDWeights.debugString();

    size_t gradient_count = window_rows * window_columns;

    std::vector<size_t> sample_positions = {parseSamplePosition(options.specificSample,
        window_rows, window_columns)};

    while (sample_positions.size() < samples) {
        sample_positions.push_back(random_engine() % gradient_count);
    }

    // grad check over the sample positions
    double epsilon    = options.epsilon;
    double difference = 0.0;
    double total      = 0.0;

    for (auto& sample_position : sample_positions) {

        size_t sample_row    = sample_position % window_rows;
        size_t sample_column = sample_position / window_rows;

        double original_value = layerWeights(sample_row, sample_column);
        double gradient       = layerDWeights(sample_row, sample_column);

        layerWeights(sample_row, sample_column) = original_value - epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value - epsilon) << "\n";

        auto copied_output_activations = copy(input_activations);
        forwardPropRecurrent(copied_output_activations, reserve, weights, handle);

        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            reshape(copy(slice(copied_output_activations,
                                     {0,0,0},
                                     {window_outputs, mini_batch, timesteps})),
                           {window_outputs, mini_batch * timesteps}).debugString();

        double left_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        layerWeights(sample_row, sample_column) = original_value + epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value + epsilon) << "\n";

        copied_output_activations = copy(input_activations);

        forwardPropRecurrent(copied_output_activations, reserve, weights, handle);
        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            reshape(copy(slice(copied_output_activations,
                               {0,0,0},
                               {window_outputs, mini_batch, timesteps})),
                   {window_outputs, mini_batch * timesteps}).debugString();

        double right_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        layerWeights(sample_row, sample_column) = original_value;

        double numerical_gradient = (right_cost - left_cost) / (2.0 * epsilon);
        double local_difference = numerical_gradient - gradient;

        difference += std::pow(local_difference, 2.0);

        double absolute_difference = (local_difference == 0 || gradient == 0) ?
            0.0 : std::abs(local_difference) / std::abs(gradient);

        total += std::pow(gradient, 2.0);

        double scaled_difference = (total == 0.0) ? difference : (difference / total);

        if (absolute_difference > 1e-3 || !std::isfinite(local_difference))
        {
            prnn::util::log("TestRecurrent") << "For weight (" << sample_row << ", "
                << sample_column
                << ") computed gradient " << gradient << " does not match estimated gradient "
                << numerical_gradient << ", cost " << cost << " left cost "
                << left_cost << ", right cost " << right_cost <<  ", " << local_difference
                << " difference, " << scaled_difference<< " scaled difference\n";
        }
        else
        {
            prnn::util::log("TestRecurrent") << "For weight (" << sample_row << ", "
                << sample_column
                << ") computed gradient " << gradient << " matches estimated gradient "
                << numerical_gradient << "\n";
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
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        options.usePersistentForwardProp,
        options.useCudnn, 0.0);

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
                                       PRNN_SKIP_INPUT,
                                       PRNN_UNIDIRECTIONAL,
                                       PRNN_RNN_RELU,
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
                                       PRNN_SKIP_INPUT,
                                       PRNN_UNIDIRECTIONAL,
                                       PRNN_RNN_RELU,
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
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        options.usePersistentForwardProp,
        options.useCudnn, 0.0);

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
                                       PRNN_SKIP_INPUT,
                                       PRNN_UNIDIRECTIONAL,
                                       PRNN_RNN_RELU,
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
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        options.usePersistentForwardProp,
        options.useCudnn, 0.0);

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
                                       PRNN_SKIP_INPUT,
                                       PRNN_UNIDIRECTIONAL,
                                       PRNN_RNN_RELU,
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
        prnn::RECURRENT_FORWARD,
        prnn::RECURRENT_SIMPLE_TYPE,
        prnn::RECURRENT_SKIP_INPUT,
        options.usePersistentForwardProp,
        options.useCudnn, 0.0);

    auto dWeights = createWeightsWithCInterface(highLevelHandle, precision);

    auto inputActivations = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto deltas = rand({options.layerSize, options.miniBatchSize, options.timesteps},
        precision);
    auto highLevelReserve = prnn::createReserveRecurrent(highLevelHandle, precision);

    auto referenceDWeights = copy(dWeights);

    backPropGradientsRecurrent(referenceDWeights, inputActivations, deltas, highLevelReserve,
        highLevelHandle);

    prnnHandle_t handle;

    assertSuccess(prnnCreate(&handle));

    prnnRNNDescriptor_t descriptor;

    assertSuccess(prnnCreateRNNDescriptor(&descriptor));

    assertSuccess(prnnSetRNNDescriptor(descriptor,
                                       options.layerSize,
                                       options.layers,
                                       nullptr,
                                       PRNN_SKIP_INPUT,
                                       PRNN_UNIDIRECTIONAL,
                                       PRNN_RNN_RELU,
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
                                         deltasDescriptors.data(),
                                         deltas.data(),
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

    for(auto& deltasDescriptor : deltasDescriptors)
    {
        assertSuccess(prnnDestroyTensorDescriptor(deltasDescriptor));
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
    catch(std::exception& e)
    {
        std::cout << "Test '" << testName << "' Failed with error '" << e.what() << "'\n";
        options.failed = true;
    }
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

    options.gradientCheckSamples = 32;

    options.usePersistentBackProp = true;
    options.usePersistentForwardProp = true;

    options.haltOnFailure = true;

    options.specificSample = "0,0";
    options.skipConnectionScale = 0.5;

    options.useCudnn = false;

    parser.parse("-t", "--timeteps",   options.timesteps,     options.timesteps,     "The number of timesteps to run the RNN for.");
    parser.parse("-b", "--mini-batch", options.miniBatchSize, options.miniBatchSize, "The mini-batch size to run through the layer.");
    parser.parse("-l", "--layer-size", options.layerSize,     options.layerSize,     "The size of the RNN layer to operate on.");
    parser.parse("-n", "--layers",     options.layers,        options.layers,        "The number of RNN layers to stack.");
    parser.parse("-e", "--epsilon",    options.epsilon,       options.epsilon,       "Epsilon used for the gradient check.");

    parser.parse("", "--persistent-back",    options.usePersistentBackProp,    options.usePersistentBackProp,    "Use persistent kernels for back prop.");
    parser.parse("", "--persistent-forward", options.usePersistentForwardProp, options.usePersistentForwardProp, "Use persistent kernels for forward prop.");
    parser.parse("", "--use-cudnn", options.useCudnn, options.useCudnn, "Use CUDNN as the backend for recurrent ops instead of this library.");

    parser.parse("", "--specific-sample", options.specificSample, options.specificSample, "Specific weight to sample.");
    parser.parse("", "--skip-connection-scale", options.skipConnectionScale, options.skipConnectionScale, "Scaling factor for skip connections.");

    parser.parse("-s", "--grad-check-samples", options.gradientCheckSamples,
        options.gradientCheckSamples, "The number of weights to perform gradient check on.");

    parser.parse("", "--halt-on-failure", options.haltOnFailure,
        options.haltOnFailure, "Abort on the first test failure.");

    parser.parse("-v", "--verbose", options.verbose, options.verbose, "Enable all PRNN logs.");

    parser.parse();

    if(options.verbose)
    {
        prnn::util::enable_all_logs();
    }

    RunTest("C Interface Tensor Test",              TestCTensor,                          options);
    RunTest("C Interface RNN Test",                 TestCRNN,                             options);
    RunTest("C Interface Forward Ops Test",         TestCForwardOps,                      options);
    RunTest("C Interface Delta Ops Test",           TestCDeltaOps,                        options);
    RunTest("C Interface Gradient Ops Test",        TestCGradientOps,                     options);
    RunTest("Recurrent Forward Ops Gradient Check", TestRecurrentOpsGradientCheck,        options);
    RunTest("Simple Recurrent Ops Test",            TestSimpleRecurrentOps,               options);
    //RunTest("Recurrent Reverse Ops Gradient Check", TestReverseRecurrentOpsGradientCheck, options);

    return 0;
}




