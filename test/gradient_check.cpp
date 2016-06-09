

// Persistent RNN Includes
#include <persistent_rnn_high_level.h>

#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/random_operations.h>
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>
#include <prnn/detail/rnn/recurrent_ops.h>

#include <prnn/detail/util/logger.h>

// Standard Library Includes
#include <random>
#include <iostream>

void TestSimpleRecurrentOps()
{
    auto precision = prnn::matrix::SinglePrecision();

    prnn::matrix::srand(377);

    size_t layer_size = prnn::rnn::getMaximumSizeRNNForThisGPU(precision);
    size_t timesteps  = 10;
    size_t mini_batch = 2;

    prnn::RecurrentOpsHandle handle(layer_size, mini_batch, timesteps,
        prnn::RecurrentRectifiedLinear(), prnn::RECURRENT_FORWARD);

    auto weights     = ones({layer_size, layer_size}, precision);
    auto activations = ones({layer_size, mini_batch, timesteps}, precision);

    forwardPropRecurrent(activations, weights, handle);

    auto deltas = ones(activations.size(), precision);

    backPropDeltasRecurrent(deltas, weights, activations, handle);

    auto dWeights = ones(weights.size(), precision);

    backPropGradientsRecurrent(dWeights, activations, deltas, handle);

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

void assertLessThanOrEqual(double left, double right)
{
    if(left <= right)
    {
        std::stringstream stream;

        stream << "Assertion Failed (" << left << " <= " << right << ")\n";

        throw std::logic_error(stream.str());
    }
}

void assertGreaterThanOrEqual(double left, double right)
{
    if(left >= right)
    {
        std::stringstream stream;

        stream << "Assertion Failed (" << left << " <= " << right << ")\n";

        throw std::logic_error(stream.str());
    }
}

void TestSimpleRecurrentOpsGradientCheck(prnn::RecurrentLayerDirection direction)
{
    auto precision = prnn::matrix::SinglePrecision();

    std::default_random_engine random_engine;

    random_engine.seed(377);

    prnn::matrix::srand(377);

    size_t layer_size = prnn::rnn::getMaximumSizeRNNForThisGPU(precision);
    size_t timesteps  = 3;
    size_t mini_batch = 3;
    size_t samples    = 4;

    size_t window_rows    = layer_size;
    size_t window_columns = window_rows;
    size_t window_outputs = window_rows;

    samples = std::min(window_rows * window_columns, samples);

    prnn::RecurrentOpsHandle handle(layer_size, mini_batch, timesteps,
        prnn::RecurrentRectifiedLinear(), direction);

    auto weights = zeros({layer_size, layer_size}, precision);
    auto weights_slice = slice(weights, {0, 0}, {window_rows, window_columns});

    copy(weights_slice, rand({window_rows, window_columns}, precision));

    apply(weights, weights, prnn::matrix::Multiply(1.0e-2));

    auto input_activations = zeros({layer_size, mini_batch, timesteps}, precision);

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

    prnn::util::log("TestRecurrent") << "Input Weights     " << weights.debugString();
    prnn::util::log("TestRecurrent") << "Input Activations " <<
        reshape(copy(slice(output_activations,
                             {0,0,0},
                             {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();

    forwardPropRecurrent(output_activations, weights, handle);

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

    backPropDeltasRecurrent(deltas, weights, output_activations, handle);

    auto dWeights = ones(weights.size(), precision);

    backPropGradientsRecurrent(dWeights, output_activations, deltas, handle);

    prnn::util::log("TestRecurrent") << "Output Deltas " <<
        reshape(copy(slice(deltas,
                           {0,0,0},
                           {window_outputs, mini_batch, timesteps})),
            {window_outputs, mini_batch * timesteps}).debugString();
    prnn::util::log("TestRecurrent") << "dWeights      " << dWeights.debugString();

    size_t gradient_count = window_rows * window_columns;

    std::vector<size_t> sample_positions = {0};

    for (size_t sample = 0; sample != samples; ++sample) {
        sample_positions.push_back(random_engine() % gradient_count);
    }

    // grad check over the sample positions
    double epsilon    = 1.0e-4;
    double difference = 0.0;
    double total      = 0.0;

    for (auto& sample_position : sample_positions) {

        size_t sample_row    = sample_position % window_rows;
        size_t sample_column = sample_position / window_rows;

        double original_value = weights (sample_row, sample_column);
        double gradient       = dWeights(sample_row, sample_column);

        weights(sample_row, sample_column) = original_value - epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value - epsilon) << "\n";

        auto copied_output_activations = copy(input_activations);
        forwardPropRecurrent(copied_output_activations, weights, handle);

        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            reshape(copy(slice(copied_output_activations,
                                     {0,0,0},
                                     {window_outputs, mini_batch, timesteps})),
                           {window_outputs, mini_batch * timesteps}).debugString();

        double left_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        weights(sample_row, sample_column) = original_value + epsilon;

        prnn::util::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value + epsilon) << "\n";

        copied_output_activations = copy(input_activations);

        forwardPropRecurrent(copied_output_activations, weights, handle);
        prnn::util::log("TestRecurrent") << "Updated Output Activations " <<
            reshape(copy(slice(copied_output_activations,
                               {0,0,0},
                               {window_outputs, mini_batch, timesteps})),
                   {window_outputs, mini_batch * timesteps}).debugString();

        double right_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        weights(sample_row, sample_column) = original_value;

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

    assertGreaterThanOrEqual(difference, 3e-2 );
    assertLessThanOrEqual(   difference, 1e-16);
}

void TestRecurrentOpsGradientCheckHelper(prnn::RecurrentLayerDirection d)
{
    TestSimpleRecurrentOpsGradientCheck(d);
}

void TestRecurrentOpsGradientCheck()
{
    TestRecurrentOpsGradientCheckHelper(prnn::RECURRENT_FORWARD);
}

void TestReverseRecurrentOpsGradientCheck()
{
    TestRecurrentOpsGradientCheckHelper(prnn::RECURRENT_REVERSE);
}

void RunTest(const std::string& testName, void (*function)(void) )
{
    try
    {
        function();
        std::cout << "Test '" << testName << "' Passed\n";
    }
    catch(std::exception& e)
    {
        std::cout << "Test '" << testName << "' Failed with error '" << e.what() << "'\n";
    }
}

int main(int argc, char** argv)
{
    prnn::util::enable_all_logs();

 //   RunTest("Simple Recurrent Ops Test",            TestSimpleRecurrentOps              );
    RunTest("Recurrent Forward Ops Gradient Check", TestRecurrentOpsGradientCheck       );
    //RunTest("Recurrent Reverse Ops Gradient Check", TestReverseRecurrentOpsGradientCheck);

    return 0;
}




