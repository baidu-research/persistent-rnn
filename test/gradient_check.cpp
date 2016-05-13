
#include <prnn/persistent_rnn_high_level.h>

void TestSimpleRecurrentOps()
{
    prnn::srand(377);

    int layer_size = 512;
    int timesteps  = 100;
    int mini_batch = 2;

    prnn::RecurrentOpsHandle handle(layer_size, mini_batch, prnn::relu(), prnn::RECURENT_FORWARD);

    auto weights     = prnn::ones({layer_size, layer_size});
    auto activations = prnn::ones({layer_size, mini_batch, timesteps});

    prnn::forward_prop_recurrent(handle, weights, activations);

    auto deltas = prnn::ones(activations.size());

    prnn::mbsp_back_prop_deltas_recurrent(handle, weights, activations, deltas);

    auto dWeights = prnn::ones(weights.size());

    prnn::back_prop_gradients_recurrent(handle, activations, deltas, dWeights);

    // just make sure nothing crashes
}

void slice_window(prnn::Matrix& inputs, int window_size)
{
    int activation_count = inputs.size()[0];
    int mini_batch_size  = inputs.size()[1];
    int timesteps        = inputs.size()[2];

    int activation_size = std::min(window_size, activation_count);

    inputs = slice(inputs,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});
}

prnn::Matrix extract_window(prnn::Matrix inputs, int window_size)
{
    int activation_count = inputs.size()[0];
    int mini_batch_size  = inputs.size()[1];
    int timesteps        = inputs.size()[2];

    int activation_size = std::min(window_size, activation_count);

    auto sliced_inputs = slice(inputs,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});

    auto zeros = prnn::zeros(inputs.size());

    auto sliced_zeros = slice(zeros,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});

    copy(sliced_inputs, sliced_zeros);

    return zeros;
}

double compute_cost(prnn::Matrix activations, prnn::Matrix reference, int window_size)
{
    slice_window(activations, window_size);
    slice_window(reference,   window_size);

    auto difference = binary_op(prnn::minus(), activations, reference);
    auto squaredDifference = unary_op(prnn::square(), difference);

    double squaredSum = static_cast<double>(reduce(prnn::plus(), {}, squaredDifference)[0]);

    return 0.5 * squaredSum / activations.size()[1];
}

prnn::Matrix compute_deltas(prnn::Matrix complete_activations,
    prnn::Matrix complete_reference, int window_size)
{
    auto activations = extract_window(complete_activations, window_size);
    auto reference   = extract_window(complete_reference,   window_size);

    size_t mini_batch = activations.size()[1];

    return unary_op(prnn::scalar_multiplies(1.0 / mini_batch),
        binary_op(prnn::minus(), activations, reference));
}

void TestSimpleRecurrentOpsGradientCheck(prnn::RecurrentLayerDirection direction)
{
    std::default_random_engine random_engine;

    random_engine.seed(377);

    prnn::set_place(p);
    prnn::srand(377);
    prnn::srandn(377);

    int layer_size = getMaximumSizeRNNForThisGPU();
    int timesteps  = 10;
    int mini_batch = 2;
    int samples    = 20;

    int window_rows    = layer_size;
    int window_columns = layer_size;
    int window_outputs = window_rows;

    samples = std::min(window_rows * window_columns, samples);

    prnn::RecurrentOpsConfig config(layer_size, mini_batch);
    prnn::RecurrentOpsHandle handle(config);

    prnn::Matrix weights = prnn::zeros({layer_size, layer_size});
    auto weights_slice = slice(weights, {0, 0}, {window_rows, window_columns});

    prnn::copy(prnn::randn({window_rows, window_columns}), weights_slice);

    prnn::unary_op(prnn::scalar_multiplies(1.0e-2), weights, weights);

    auto input_activations = prnn::zeros({layer_size, mini_batch, timesteps});

    auto reference_activations = prnn::zeros({layer_size, mini_batch, timesteps});

    prnn::copy(
        prnn::rand({window_outputs, mini_batch, timesteps}),
        prnn::slice(input_activations,
                     {0, 0, 0},
                     {window_outputs, mini_batch, timesteps}));

    prnn::copy(
        prnn::rand({window_outputs, mini_batch, timesteps}),
        prnn::slice(reference_activations,
                    {0, 0, 0},
                    {window_outputs, mini_batch, timesteps}));

    auto output_activations = copy(input_activations);

    logger::log("TestRecurrent") << "Input Weights     " << prnn::preview_array(weights);
    logger::log("TestRecurrent") << "Input Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                                {0,0,0},
                                                {window_outputs, mini_batch,
                                                                timesteps})),
                       {window_outputs, mini_batch * timesteps}));

    prnn::forward_prop_recurrent(handle, prnn::relu(), direction,
        weights, output_activations);

    logger::log("TestRecurrent") << "Output Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                                {0,0,0},
                                                {window_outputs, mini_batch,
                                                                timesteps})),
                       {window_outputs, mini_batch * timesteps)));
    logger::log("TestRecurrent") << "Reference Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(reference_activations,
                                                {0,0,0},
                                                {window_outputs, mini_batch,
                                                                timesteps})),
                       {window_outputs, mini_batch * timesteps}));

    double cost = compute_cost(output_activations, reference_activations, window_outputs);
    prnn::Matrix<Real, 3> deltas = compute_deltas(output_activations, reference_activations,
        window_outputs);

    logger::log("TestRecurrent") << "Input Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                                {0,0,0},
                                                {window_outputs, mini_batch,
                                                                timesteps})),
                       {window_outputs, mini_batch * timesteps}));

    prnn::back_prop_deltas_recurrent(handle, prnn::mult_drelu(),
        direction, weights, output_activations, deltas);

    prnn::Matrix dWeights = prnn::ones(weights.size());

    prnn::back_prop_gradients_recurrent(handle, direction,
        output_activations, deltas, dWeights);

    logger::log("TestRecurrent") << "Output Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                             {0,0,0},
                                             {window_outputs, mini_batch,
                                                            timesteps})),
                       {window_outputs, mini_batch * timesteps}));
    logger::log("TestRecurrent") << "dWeights      " << prnn::preview_array(dWeights);

    size_t gradient_count = window_rows * window_columns;

    std::vector<size_t> sample_positions = {0};

    for (int sample = 0; sample != samples; ++sample) {
        sample_positions.push_back(random_engine() % gradient_count);
    }

    // grad check over the sample positions
    double epsilon    = 1.0e-4;
    double difference = 0.0;
    double total      = 0.0;

    for (auto& sample_position : sample_positions) {

        size_t sample_row    = sample_position % window_rows;
        size_t sample_column = sample_position / window_rows;

        Real original_value = weights [{sample_row, sample_column}];
        Real gradient       = dWeights[{sample_row, sample_column}];

        weights[sample_row, sample_column] = original_value - epsilon;

        logger::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value - epsilon) << "\n";

        auto copied_output_activations = copy(input_activations);
        prnn::forward_prop_recurrent(handle, prnn::relu(), direction,
            weights, copied_output_activations);
        logger::log("TestRecurrent") << "Updated Output Activations " << prnn::preview_array(
            prnn::reshape(prnn::copy(prnn::slice(copied_output_activations,
                                                    {0,0,0},
                                                    {window_outputs, mini_batch,
                                                                    timesteps})),
                           {window_outputs, mini_batch * timesteps}));

        double left_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        set(weights, {sample_row, sample_column},
            static_cast<Real>(original_value + epsilon));

        logger::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value + epsilon) << "\n";

        copied_output_activations = copy(input_activations);
        prnn::forward_prop_recurrent(handle, prnn::relu(), direction,
            weights, copied_output_activations);
        logger::log("TestRecurrent") << "Updated Output Activations " << prnn::preview_array(
            prnn::reshape(prnn::copy(prnn::slice(copied_output_activations,
                                                    {0,0,0},
                                                    {window_outputs, mini_batch,
                                                                    timesteps})),
                           {window_outputs, mini_batch * timesteps}));

        double right_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        set(weights, {sample_row, sample_column}, original_value);

        double numerical_gradient = (right_cost - left_cost) / (2.0 * epsilon);
        double local_difference = numerical_gradient - gradient;

        difference += std::pow(local_difference, 2.0);

        double absolute_difference = (local_difference == 0 || gradient == 0) ?
            0.0 : std::abs(local_difference) / std::abs(gradient);

        total += std::pow(gradient, 2.0);

        double scaled_difference = (total == 0.0) ? difference : (difference / total);

        if (absolute_difference > 1e-3 || !std::isfinite(local_difference))
        {
            logger::log("TestRecurrent") << "For weight (" << sample_row << ", " << sample_column
                << ") computed gradient " << gradient << " does not match estimated gradient "
                << numerical_gradient << ", cost " << cost << " left cost "
                << left_cost << ", right cost " << right_cost <<  ", " << local_difference
                << " difference, " << scaled_difference<< " scaled difference\n";
        }
        else
        {
            logger::log("TestRecurrent") << "For weight (" << sample_row << ", " << sample_column
                << ") computed gradient " << gradient << " matches estimated gradient "
                << numerical_gradient << "\n";
        }
    }

    difference = (difference == 0.0 && total == 0.0) ? 0.0 : (difference/total);

    ASSERT_LEQUAL(difference, 3e-2);
    ASSERT_GEQUAL(difference, 1e-16);
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

void RunTest(const std::string& testName, void(*)(void) function)
{
    try
    {
        function();
        std::cout << "Test " << testName << " Passed\n";
    }
    catch(std::exception& e)
    {
        std::cout << "Test " << testName << " Failed\n";
    }
}

int main(int argc, char** argv)
{
    RunTest("Recurrent Forward Ops Gradient Check", TestRecurrentOpsGradientCheck());
    RunTest("Recurrent Reverse Ops Gradient Check", TestReverseRecurrentOpsGradientCheck());

    return 0;
}




