
#include <prnn/persistent_rnn_high_level.h>

void TestSimpleRecurrentOps() {

    prnn::srand(377);

    int layer_size = 512;
    int timesteps  = 100;
    int mini_batch = 2;

    prnn::RecurrentOpsConfig config(layer_size, mini_batch);
    prnn::RecurrentOpsHandle handle(config);

    auto weights     = prnn::ones({layer_size, layer_size});
    auto activations = prnn::ones({layer_size, mini_batch, timesteps});

    prnn::forward_prop_recurrent(handle, prnn::relu(), prnn::RECURRENT_FORWARD,
                                 weights, activations);

    auto deltas = prnn::ones(activations.size());

    prnn::mbsp_back_prop_deltas_recurrent(handle, prnn::mult_drelu(),
                                          prnn::RECURRENT_FORWARD, weights,
                                          activations, deltas);

    auto dWeights = prnn::ones(weights.size());

    prnn::back_prop_gradients_recurrent(handle, prnn::RECURRENT_FORWARD,
                                        activations, deltas, dWeights);

    // just make sure nothing crashes
}

void slice_window(prnn::Array& inputs, int window_size) {
    int activation_count = inputs.size()[0];
    int mini_batch_size  = inputs.size()[1];
    int timesteps        = inputs.size()[2];

    int activation_size = std::min(window_size, activation_count);

    inputs = slice(inputs,
        {0,               0,               0},
        {activation_size, mini_batch_size, timesteps});
}

prnn::Array extract_window(prnn::Array inputs, int window_size) {
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

double compute_cost(prnn::Array activations, prnn::Array reference, int window_size) {
    slice_window(activations, window_size);
    slice_window(reference,   window_size);

    auto difference = binary_op(prnn::minus(), activations, reference);
    auto squaredDifference = unary_op(prnn::square(), difference);

    double squaredSum = static_cast<double>(reduce(prnn::plus(), {}, squaredDifference)[0]);

    return 0.5 * squaredSum / activations.size()[1];
}

prnn::Array compute_deltas(prnn::Array complete_activations,
    prnn::Array complete_reference, int window_size) {

    auto activations = extract_window(complete_activations, window_size);
    auto reference   = extract_window(complete_reference,   window_size);

    size_t mini_batch = activations.size()[1];

    return unary_op(prnn::scalar_multiplies(1.0 / mini_batch),
        binary_op(prnn::minus(), activations, reference));
}

void TestSimpleRecurrentOpsGradientCheck(prnn::RecurrentLayerDirection direction) {

    std::default_random_engine random_engine;

    random_engine.seed(377);

    prnn::set_place(p);
    prnn::srand(377);
    prnn::srandn(377);

    int layer_size = 224;
    int timesteps  = 2;
    int mini_batch = 2;
    int samples    = 4;

    int window_rows    = 224;
    int window_columns = 224;
    int window_outputs = window_rows;

    samples = std::min(window_rows * window_columns, samples);

    // note that we are testing the persistent kernels here
    prnn::RecurrentOpsConfig config(layer_size, mini_batch);
    prnn::RecurrentOpsHandle handle(config);

    prnn::Array weights = prnn::zeros({layer_size, layer_size});
    auto weights_slice = slice(weights, {0, 0}, {window_rows, window_columns});

    prnn::copy(prnn::randn(prnn::make_dim(window_rows, window_columns)), weights_slice);

    prnn::unary_op(prnn::scalar_multiplies(1.0e-2), weights, weights);

    auto input_activations = prnn::zeros<Real>(
        prnn::make_dim(layer_size, mini_batch, timesteps));

    auto reference_activations = prnn::zeros<Real>(
        prnn::make_dim(layer_size, mini_batch, timesteps));

    prnn::copy(
        prnn::copy(prnn::rand<Real>(prnn::make_dim(window_outputs, mini_batch, timesteps),
                          prnn::CpuPlace()), prnn::get_place()),
        prnn::slice(input_activations,
                     prnn::make_dim(0, 0, 0),
                     prnn::make_dim(window_outputs, mini_batch, timesteps)));

    prnn::copy(
        prnn::copy(prnn::rand<Real>(prnn::make_dim(window_outputs, mini_batch, timesteps),
                          prnn::CpuPlace()), prnn::get_place()),
        prnn::slice(reference_activations,
                     prnn::make_dim(0, 0, 0),
                     prnn::make_dim(window_outputs, mini_batch, timesteps)));

    auto output_activations = copy(input_activations);

    logger::log("TestRecurrent") << "Input Weights     " << prnn::preview_array(weights);
    logger::log("TestRecurrent") << "Input Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                                prnn::make_dim(0,0,0),
                                                prnn::make_dim(window_outputs, mini_batch,
                                                                timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    prnn::mbsp_forward_prop_recurrent(handle, prnn::relu(), direction,
        weights, output_activations);

    logger::log("TestRecurrent") << "Output Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                                prnn::make_dim(0,0,0),
                                                prnn::make_dim(window_outputs, mini_batch,
                                                                timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));
    logger::log("TestRecurrent") << "Reference Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(reference_activations,
                                                prnn::make_dim(0,0,0),
                                                prnn::make_dim(window_outputs, mini_batch,
                                                                timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    double cost = compute_cost(output_activations, reference_activations, window_outputs);
    prnn::Array<Real, 3> deltas = compute_deltas(output_activations, reference_activations,
        window_outputs);

    logger::log("TestRecurrent") << "Input Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                                prnn::make_dim(0,0,0),
                                                prnn::make_dim(window_outputs, mini_batch,
                                                                timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    prnn::mbsp_back_prop_deltas_recurrent(handle, prnn::mult_drelu(),
        direction, weights, output_activations, deltas);

    prnn::Array<Real, 2> dWeights = prnn::ones<Real>(weights.size());

    prnn::mbsp_back_prop_gradients_recurrent(handle, direction,
        output_activations, deltas, dWeights);

    logger::log("TestRecurrent") << "Output Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                                prnn::make_dim(0,0,0),
                                                prnn::make_dim(window_outputs, mini_batch,
                                                                timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));
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

        Real original_value = weights [prnn::make_dim(sample_row, sample_column)];
        Real gradient       = dWeights[prnn::make_dim(sample_row, sample_column)];

        set(weights, prnn::make_dim(sample_row, sample_column),
            static_cast<Real>(original_value - epsilon));

        logger::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value - epsilon) << "\n";

        auto copied_output_activations = copy(input_activations);
        prnn::mbsp_forward_prop_recurrent(handle, prnn::relu(), direction,
            weights, copied_output_activations);
        logger::log("TestRecurrent") << "Updated Output Activations " << prnn::preview_array(
            prnn::reshape(prnn::copy(prnn::slice(copied_output_activations,
                                                    prnn::make_dim(0,0,0),
                                                    prnn::make_dim(window_outputs, mini_batch,
                                                                    timesteps))),
                           prnn::make_dim(window_outputs, mini_batch * timesteps)));

        double left_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        set(weights, prnn::make_dim(sample_row, sample_column),
            static_cast<Real>(original_value + epsilon));

        logger::log("TestRecurrent") << "Updated Input Weight (" << sample_row << ", "
            << sample_column << ")     from " << original_value << " to "
            << (original_value + epsilon) << "\n";

        copied_output_activations = copy(input_activations);
        prnn::mbsp_forward_prop_recurrent(handle, prnn::relu(), direction,
            weights, copied_output_activations);
        logger::log("TestRecurrent") << "Updated Output Activations " << prnn::preview_array(
            prnn::reshape(prnn::copy(prnn::slice(copied_output_activations,
                                                    prnn::make_dim(0,0,0),
                                                    prnn::make_dim(window_outputs, mini_batch,
                                                                    timesteps))),
                           prnn::make_dim(window_outputs, mini_batch * timesteps)));

        double right_cost = compute_cost(copied_output_activations, reference_activations,
            window_outputs);

        set(weights, prnn::make_dim(sample_row, sample_column), original_value);

        double numerical_gradient = (right_cost - left_cost) / (2.0 * epsilon);
        double local_difference = numerical_gradient - gradient;

        difference += std::pow(local_difference, 2.0);

        double absolute_difference = (local_difference == 0 || gradient == 0) ?
            0.0 : std::abs(local_difference) / std::abs(gradient);

        total += std::pow(gradient, 2.0);

        double scaled_difference = (total == 0.0) ? difference : (difference / total);

        if (absolute_difference > 1e-3 || !std::isfinite(local_difference)) {
            logger::log("TestRecurrent") << "For weight (" << sample_row << ", " << sample_column
                << ") computed gradient " << gradient << " does not match estimated gradient "
                << numerical_gradient << ", cost " << cost << " left cost "
                << left_cost << ", right cost " << right_cost <<  ", " << local_difference
                << " difference, " << scaled_difference<< " scaled difference\n";
        }
        else {
            logger::log("TestRecurrent") << "For weight (" << sample_row << ", " << sample_column
                << ") computed gradient " << gradient << " matches estimated gradient "
                << numerical_gradient << "\n";
        }
    }

    difference = (difference == 0.0 && total == 0.0) ? 0.0 : (difference/total);

    ASSERT_LEQUAL(difference, 3e-2);
    ASSERT_GEQUAL(difference, 1e-16);
}

void TestSimpleRecurrentOpsReferenceCheck(prnn::Place p,
    prnn::RecurrentLayerDirection direction) {

    typedef float Real;

    std::default_random_engine random_engine;

    random_engine.seed(377);

    prnn::set_place(prnn::CpuPlace());
    prnn::srand(377);
    prnn::srandn(377);

    prnn::set_place(p);
    prnn::srand(377);
    prnn::srandn(377);

    int layer_size = 1152;
    int timesteps  = 10;
    int mini_batch = 16;
    int samples    = 100;

    int window_rows    = 1152;
    int window_columns = 1152;
    int window_outputs = window_rows;

    samples = std::min(window_rows * window_columns, samples);

    // note that we are testing the persistent kernels here
    prnn::RecurrentOpsConfig config(layer_size, mini_batch, true);
    prnn::RecurrentOpsHandle handle(config);

    // note that we are using the generic kernels here
    prnn::RecurrentOpsConfig nonpersistent_config(layer_size, mini_batch, false);
    prnn::RecurrentOpsHandle nonpersistent_handle(nonpersistent_config);

    prnn::Array<Real, 2> weights = prnn::zeros<Real>(prnn::make_dim(layer_size, layer_size));
    auto weights_slice = slice(weights, prnn::make_dim(0, 0),
        prnn::make_dim(window_rows, window_columns));

    prnn::copy(prnn::randn<Real>(prnn::make_dim(window_rows, window_columns),
        prnn::CpuPlace()), weights_slice);

    prnn::unary_op(prnn::scalar_multiplies(1.0e-2), weights, weights);

    prnn::Array<Real, 3> input_activations = prnn::zeros<Real>(
        prnn::make_dim(layer_size, mini_batch, timesteps));

    prnn::Array<Real, 3> reference_activations = prnn::zeros<Real>(
        prnn::make_dim(layer_size, mini_batch, timesteps));

    prnn::copy(
        prnn::copy(prnn::rand<Real>(prnn::make_dim(window_outputs, mini_batch, timesteps),
                          prnn::CpuPlace()), prnn::get_place()),
        prnn::slice(input_activations,
                     prnn::make_dim(0, 0, 0),
                     prnn::make_dim(window_outputs, mini_batch, timesteps)));

    prnn::copy(
        prnn::copy(prnn::rand<Real>(prnn::make_dim(window_outputs, mini_batch, timesteps),
                          prnn::CpuPlace()), prnn::get_place()),
        prnn::slice(reference_activations,
                     prnn::make_dim(0, 0, 0),
                     prnn::make_dim(window_outputs, mini_batch, timesteps)));

    auto output_activations = copy(input_activations);
    auto nonpersistent_output_activations = copy(input_activations);

    logger::log("TestRecurrent") << "Input Weights     " << prnn::preview_array(weights);
    logger::log("TestRecurrent") << "Input Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    prnn::mbsp_forward_prop_recurrent(handle, prnn::relu(), direction,
        weights, output_activations);

    prnn::mbsp_forward_prop_recurrent(nonpersistent_handle, prnn::relu(), direction,
        weights, nonpersistent_output_activations);

    logger::log("TestRecurrent") << "Output Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activations,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));
    logger::log("TestRecurrent") << "Reference Activations " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(reference_activations,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    auto deltas = compute_deltas(output_activations, reference_activations,
        window_outputs);
    auto nonpersistent_deltas = copy(deltas);

    logger::log("TestRecurrent") << "Input Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    prnn::mbsp_back_prop_deltas_recurrent(handle, prnn::mult_drelu(),
        direction, weights, output_activations, deltas);

    prnn::mbsp_back_prop_deltas_recurrent(nonpersistent_handle, prnn::mult_drelu(),
        direction, weights, output_activations, nonpersistent_deltas);

    auto dWeights = prnn::ones<Real>(weights.size());
    auto nonpersistent_dWeights = prnn::ones<Real>(weights.size());

    prnn::mbsp_back_prop_gradients_recurrent(handle, direction,
        output_activations, deltas, dWeights);

    prnn::mbsp_back_prop_gradients_recurrent(nonpersistent_handle, direction,
        output_activations, deltas, nonpersistent_dWeights);

    logger::log("TestRecurrent") << "Output Deltas " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(deltas,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));
    logger::log("TestRecurrent") << "dWeights      " << prnn::preview_array(dWeights);

    auto output_activation_difference = prnn::unary_op(prnn::absolute(),
        prnn::binary_op(prnn::minus(), output_activations,
            nonpersistent_output_activations));

    logger::log("TestRecurrent") << "Output Activation Differences " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(output_activation_difference,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    ASSERT_LEQUAL(static_cast<double>(prnn::reduce(prnn::plus(), {},
        output_activation_difference)[0]) /
        (0.0 + window_outputs * mini_batch * timesteps), 1.0e-6);

    auto delta_difference = prnn::unary_op(prnn::absolute(),
        prnn::binary_op(prnn::minus(), deltas, nonpersistent_deltas));

    logger::log("TestRecurrent") << "Deltas Differences " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(delta_difference,
                                             prnn::make_dim(0,0,0),
                                             prnn::make_dim(window_outputs, mini_batch, timesteps))),
                       prnn::make_dim(window_outputs, mini_batch * timesteps)));

    ASSERT_LEQUAL(static_cast<double>(prnn::reduce(prnn::plus(), {}, delta_difference)[0]) /
        (0.0 + window_outputs * mini_batch * timesteps), 1.0e-6);

    auto dWeights_difference = prnn::unary_op(prnn::absolute(),
        prnn::binary_op(prnn::minus(), dWeights, nonpersistent_dWeights));

    logger::log("TestRecurrent") << "dWeights Differences " << prnn::preview_array(
        prnn::reshape(prnn::copy(prnn::slice(dWeights_difference,
                                                prnn::make_dim(0,0),
                                                prnn::make_dim(window_outputs, window_outputs))),
                       prnn::make_dim(window_outputs, window_outputs)));

    ASSERT_LEQUAL(static_cast<double>(prnn::reduce(prnn::plus(), {}, dWeights_difference)[0]) /
        (window_outputs * window_outputs), 1.0e-6);
}

void TestRecurrentOpsGradientCheckHelper(prnn::Place p, prnn::RecurrentLayerDirection d) {
    TestSimpleRecurrentOpsGradientCheck(p, d);
}

void TestRecurrentOpsGradientCheckCpu() {
    TestRecurrentOpsGradientCheckHelper(prnn::RECURRENT_FORWARD);
}

void TestRecurrentOpsGradientCheckGpu() {
    TestRecurrentOpsGradientCheckHelper(prnn::RECURRENT_FORWARD);
}

void TestReverseRecurrentOpsGradientCheckGpu() {
    TestRecurrentOpsGradientCheckHelper(prnn::RECURRENT_REVERSE);
}

void TestRecurrentOpsReferenceCheckGpu() {
    TestSimpleRecurrentOpsReferenceCheck(prnn::RECURRENT_FORWARD);
}

int main(int argc, char** argv) {



    return 0;
}




