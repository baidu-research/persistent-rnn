#pragma once

// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops_config.h>
#include <prnn/detail/rnn/synchronizer.h>

#include <prnn/detail/util/atomics.h>

#define DEBUG_RECURRENT_OPS 0

#if DEBUG_RECURRENT_OPS

#define dprintf(...) do { if( blockIdx.x == 0 && blockIdx.y == 0) \
    { std::printf(__VA_ARGS__); } } while(0)

#define t0printf(...) do { if(threadIdx.x == 0 && threadIdx.y == 0 && \
    blockIdx.x == 0 && blockIdx.y == 0) { std::printf(__VA_ARGS__); } } while(0)

#define UNROLL

#else

#define dprintf(...) do { } while(0)

#define t0printf(...) do { } while(0)

#define UNROLL _Pragma("unroll")

#endif

namespace prnn {
namespace rnn {

template<typename Config>
class PersistentEngineParameters
{
public:
    typedef typename Config::RealType RealType;

public:
    PersistentEngineParameters(const Config& config,
        const RealType* weights, RealType* activations,
        RealType* activations_scratch,
        double skip_connection_scale, const Synchronizer& synchronizer_)
    : PersistentEngineParameters(config, weights, nullptr, activations,
        activations_scratch, skip_connection_scale, synchronizer_)
    {

    }

    PersistentEngineParameters(const Config& config,
        const RealType* weights,
        RealType* back_prop_activations,
        RealType* activations_or_deltas,
        RealType* activations_scratch,
        double skip_connection_scale, const Synchronizer& synchronizer_)
    : weights(weights),
      back_prop_activations(back_prop_activations),
      activations(activations_or_deltas),
      activation_scratch(activations_scratch),
      skip_connection_scale(skip_connection_scale),
      layer_size(config.handle.layerSize),
      mini_batch_size(config.handle.miniBatchSize),
      timesteps(config.handle.timesteps),
      is_fully_covered(config.handle.layerSize % Config::BLOCK_TILE_COLUMNS == 0),
      synchronizer(synchronizer_),
      config(config)
    {
        iteration_step = layer_size;
        input_to_output_offset = layer_size * (mini_batch_size);
        scratch_input_to_output_offset = Config::GRID_TILE_ROWS * (mini_batch_size);
        iterations = mini_batch_size * (timesteps - 1);

        first_iteration = synchronizer_.get_current_phase();

        scratch_step_size = Config::GRID_TILE_ROWS;

        reduction_threads_per_value = (layer_size + Config::THREAD_TILE_COLUMNS - 1) /
            Config::THREAD_TILE_COLUMNS;
    }

    public:
        __device__ RealType* get_deltas() const {
            return activations;
        }

        __device__ RealType* get_deltas_scratch() const {
            return activation_scratch;
        }

public:
    const RealType* weights;
    RealType*       back_prop_activations;
    RealType*       activations;
    RealType*       activation_scratch;
    RealType        skip_connection_scale;
    index_t         layer_size;
    index_t         mini_batch_size;
    index_t         timesteps;

public:
    index_t input_to_output_offset;
    index_t scratch_input_to_output_offset;
    index_t iteration_step;
    index_t iterations;
    index_t first_iteration;

public:
    index_t scratch_step_size;

public:
    index_t reduction_threads_per_value;

public:
    bool is_fully_covered;

public:
    GpuSynchronizer synchronizer;

public:
    Config config;

};

template<typename Config>
class PersistentEngine
{
private:
    typedef typename Config::RealType RealType;
    typedef int32_t fp16x2;

    typedef typename Config::FixedPointType FixedPoint;
    typedef typename Config::IntType        IntType;

    typedef typename Config::ThreadTileWeights      ThreadTileWeights;
    typedef typename Config::ThreadTileInputs       ThreadTileInputs;
    typedef typename Config::ThreadTileAccumulators ThreadTileAccumulators;
    typedef typename Config::DataLoadingBuffer      DataLoadingBuffer;
    typedef typename Config::SharedDataStorage      SharedDataStorage;

    typedef typename Config::GlobalAccessType GlobalAccessType;
    typedef typename Config::SharedAccessType SharedAccessType;
    typedef typename Config::SharedStoreType  SharedStoreType;
    typedef typename Config::WeightAccessType WeightAccessType;

    typedef typename Config::OutputSharedAccessType OutputSharedAccessType;

    enum {
        THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK
    };

    enum {
        BLOCKS_PER_SM = Config::BLOCKS_PER_SM
    };

    class RegisterState
    {
    public:
        __device__ RegisterState(const PersistentEngineParameters<Config>& parameters)
        :
          shared_base(0),
          barrier_success(true),
          skip_connection_scale(parameters.skip_connection_scale),
          layer_size(parameters.layer_size),
          input_to_output_offset(parameters.input_to_output_offset),
          scratch_input_to_output_offset(parameters.scratch_input_to_output_offset),
          iteration_step(parameters.iteration_step),
          iterations(parameters.iterations),
          scratch_step_size(parameters.scratch_step_size),
          reduction_threads_per_value(parameters.reduction_threads_per_value)
        {

            if (Config::DIRECTION == prnn::RECURRENT_REVERSE) {
                index_t iteration = parameters.timesteps * parameters.mini_batch_size -
                    parameters.first_iteration - 1;

                back_prop_activation_base_pointer = parameters.back_prop_activations +
                    iteration * parameters.layer_size;
                input_base_pointer = parameters.activations +
                    iteration * parameters.layer_size;

                activation_scratch = parameters.activation_scratch +
                    Config::GRID_TILE_COLUMNS * iteration;
            }
            else {
                back_prop_activation_base_pointer = parameters.back_prop_activations +
                    parameters.first_iteration * parameters.layer_size;
                input_base_pointer = parameters.activations +
                    parameters.first_iteration * parameters.layer_size;

                activation_scratch = parameters.activation_scratch +
                    Config::GRID_TILE_COLUMNS * parameters.first_iteration;
            }
        }

    public:
        RealType* activation_scratch;

    public:
        index_t shared_base;

    public:
        bool barrier_success;

    public:
        RealType* back_prop_activation_base_pointer;
        RealType* input_base_pointer;

    public:
        RealType skip_connection_scale;
        index_t layer_size;

    public:
        index_t input_to_output_offset;
        index_t scratch_input_to_output_offset;
        index_t iteration_step;
        index_t iterations;

    public:
        index_t scratch_step_size;

    public:
        index_t reduction_threads_per_value;

    };

public:
    __device__ PersistentEngine(const PersistentEngineParameters<Config>& parameters)
    : parameters(parameters), synchronizer(parameters.synchronizer) {

    }

public:
    __device__ inline void run_forward()
    {
        t0printf("Thread (%d, %d, %d, %d) - Starting persistent forward engine on "
            "iteration %d (scratch %p) (acts/deltas %p).\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, parameters.first_iteration,
            parameters.activation_scratch, parameters.activations);

        if (!is_restarted()) {
            populate_scratch();
        }
        else if (parameters.first_iteration >=
            parameters.iterations + parameters.mini_batch_size) {
            return;
        }

        DataLoadingBuffer data_buffer;
        ThreadTileAccumulators accumulators;

        RegisterState register_state(parameters);

        __shared__ SharedDataStorage shared_state;

        warm_start(register_state, shared_state, data_buffer, accumulators);

        ThreadTileWeights weights;

        load_weights(weights);

        index_t iteration = parameters.first_iteration + 1;
        for(; iteration < register_state.iterations; ++iteration)
        {
            t0printf("Thread (%d, %d, %d, %d) - Starting iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iteration);

            perform_iteration(register_state, shared_state, weights, data_buffer, accumulators);

            if (!register_state.barrier_success) {
                break;
            }
        }

        if (register_state.barrier_success) {
            clean_up(register_state, shared_state, weights, data_buffer, accumulators, iteration);
        }

        if (!register_state.barrier_success) {
            t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            synchronizer.set_concurrent_execution_failed();
            synchronizer.set_phase(iteration);
        }
    }

public:
    __device__ inline void run_back_prop_deltas()
    {
        t0printf("Thread (%d, %d, %d, %d) - Starting persistent back prop deltas engine on "
            "iteration %d (scratch %p) (deltas %p) (back prop acts %p).\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, parameters.first_iteration,
            parameters.activation_scratch, parameters.activations,
            parameters.back_prop_activations);

        if (!is_restarted()) {
            populate_scratch_back_prop();
        }
        else if (parameters.first_iteration >=
            parameters.iterations + parameters.mini_batch_size) {
            return;
        }

        DataLoadingBuffer data_buffer;
        DataLoadingBuffer activation_buffer;
        ThreadTileAccumulators accumulators;

        RegisterState register_state(parameters);

        __shared__ SharedDataStorage shared_state;

        warm_start_back_prop(register_state, shared_state,
            data_buffer, activation_buffer, accumulators);

        ThreadTileWeights weights;

        load_transposed_weights(weights);

        index_t iteration = parameters.first_iteration + 1;
        for(; iteration < register_state.iterations; ++iteration)
        {
            t0printf("Thread (%d, %d, %d, %d) - Starting iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iteration);

            perform_back_prop_iteration(register_state, shared_state,
                weights, data_buffer, activation_buffer,
                accumulators);

            if (!register_state.barrier_success) {
                break;
            }
        }

        if (register_state.barrier_success) {
            clean_up_back_prop(register_state, shared_state,
                weights, data_buffer, activation_buffer,
                accumulators, iteration);
        }

        if (!register_state.barrier_success) {
            t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            synchronizer.set_concurrent_execution_failed();
            synchronizer.set_phase(iteration);
        }
    }

private:
    __device__ void perform_iteration(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        ThreadTileAccumulators& accumulators, bool load_output = true,
        bool should_store_accumulators = true)
    {
        ThreadTileInputs thread_inputs;

        load_input(register_state, data_buffer, load_output);
        load_thread_tile_inputs(register_state, shared_state, thread_inputs);

        perform_thread_tile_math(accumulators, weights, thread_inputs);

        if (should_store_accumulators) {
            store_accumulators(register_state, accumulators);
        }

        advance_shared_pointers(register_state);

        detect_barrier_success(register_state, data_buffer);

        handle_barrier_failure(register_state, data_buffer);
        format_input(register_state, shared_state, data_buffer);

        synchronize_block();

        advance_pointers(register_state);

        initialize_accumulators(register_state, shared_state, accumulators);
    }

    __device__ void perform_back_prop_iteration(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        DataLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators, bool load_output = true,
        bool should_store_accumulators = true)
    {
        ThreadTileInputs thread_inputs;

        load_input(register_state, data_buffer, load_output);
        load_back_prop_activations(register_state, activation_buffer);

        load_thread_tile_inputs(register_state, shared_state, thread_inputs);

        perform_thread_tile_math(accumulators, weights, thread_inputs);

        if (should_store_accumulators) {
            store_accumulators_back_prop(register_state, accumulators);
        }

        advance_shared_pointers(register_state);

        detect_barrier_success(register_state, data_buffer);

        handle_barrier_failure(register_state, data_buffer);
        format_input_back_prop(register_state, shared_state, data_buffer, activation_buffer);

        synchronize_block();

        advance_pointers_back_prop(register_state);

        initialize_accumulators(register_state, shared_state, accumulators);
    }

private:
    __device__ void warm_start(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer, ThreadTileAccumulators& accumulators) {

        t0printf("Thread (%d, %d, %d, %d) - Warm starting first iteration.\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

        load_input(register_state, data_buffer,
            parameters.first_iteration < parameters.iterations);
        format_input(register_state, shared_state, data_buffer);

        synchronize_block();

        initialize_accumulators(register_state, shared_state, accumulators);

        advance_pointers(register_state);
    }

    __device__ void warm_start_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer, DataLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators) {

        t0printf("Thread (%d, %d, %d, %d) - Warm starting first back prop iteration.\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

        load_input(register_state, data_buffer,
            parameters.first_iteration < parameters.iterations);
        load_back_prop_activations(register_state, activation_buffer);

        format_input_back_prop(register_state, shared_state, data_buffer, activation_buffer);

        synchronize_block();

        initialize_accumulators(register_state, shared_state, accumulators);

        advance_pointers_back_prop(register_state);
    }

private:
    __device__ void clean_up(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        ThreadTileAccumulators& accumulators, index_t iteration) {

        for(; iteration < parameters.iterations + parameters.mini_batch_size;
            ++iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Clean up iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iteration);

            bool should_store_accumulators = iteration < parameters.iterations + 1;

            perform_iteration(register_state, shared_state,
                weights, data_buffer, accumulators, false,
                should_store_accumulators);
        }
    }

    __device__ void clean_up_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        DataLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators, index_t iteration) {

        for(; iteration < parameters.iterations + parameters.mini_batch_size;
            ++iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Clean up back prop iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, iteration);

            bool should_store_accumulators = iteration < parameters.iterations + 1;

            perform_back_prop_iteration(register_state, shared_state,
                weights, data_buffer, activation_buffer,
                accumulators, false, should_store_accumulators);
        }
    }


private:
    __device__ bool is_restarted() {
        return parameters.first_iteration != 0;
    }

private:
    __device__ index_t thread_id_in_grid() const {
        return threadIdx.x +
            threadIdx.y * (blockDim.x) +
            blockIdx.x * (blockDim.x * blockDim.y) +
            blockIdx.y * (blockDim.x * blockDim.y * gridDim.x);
    }

    __device__ index_t grid_size() const {
        return blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    }

    __device__ void populate_scratch() {
        index_t id   = thread_id_in_grid();
        index_t size = grid_size();

        for (index_t offset = id; offset < parameters.layer_size * parameters.mini_batch_size;
            offset += size) {

            auto value = parameters.activations[offset];

            dprintf("Thread (%d, %d, %d, %d) - Warm starting scratch "
                "offset[%d] (%p) = activations[%d] (%p) (%f) \n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                get_scratch_offset(offset),
                parameters.activation_scratch + get_scratch_offset(offset),
                offset, parameters.activations + offset, (float)value);

            reinterpret_cast<IntType&>(
                parameters.activation_scratch[get_scratch_offset(offset)]) =
                to_fixed_point_with_counter(value, Config::THREADS_PER_GLOBAL_REDUCTION);
        }
    }

private:
    __device__ void populate_scratch_back_prop() {
        index_t id   = thread_id_in_grid();
        index_t size = grid_size();

        auto* deltas_base = parameters.get_deltas() +
            (parameters.timesteps - 1) * parameters.mini_batch_size * parameters.layer_size;
        auto* deltas_scratch_base = parameters.get_deltas_scratch() +
            (parameters.timesteps - 1) * parameters.mini_batch_size  * Config::GRID_TILE_ROWS;

        for (index_t offset = id; offset < parameters.layer_size * parameters.mini_batch_size;
            offset += size) {

            auto value = deltas_base[offset];

            dprintf("Thread (%d, %d, %d, %d) - Warm starting scratch "
                "offset[%d] (%p) = deltas[%d] (%p) (%f) \n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                get_scratch_offset(offset),
                deltas_scratch_base + get_scratch_offset(offset),
                offset, deltas_base + offset, value);

            reinterpret_cast<IntType&>(
                deltas_scratch_base[get_scratch_offset(offset)]) =
                to_fixed_point_with_counter(value, Config::THREADS_PER_GLOBAL_REDUCTION);
        }
    }

private:
    __device__ void load_weights(ThreadTileWeights& weights) {

       // if (parameters.is_fully_covered) {
       //     unrolled_load_weights(weights);
       // }
       // else {
            safe_load_weights(weights, false);
       // }
    }

    __device__ void unrolled_load_weights(ThreadTileWeights& weights) {

        index_t thread_tile_base_row = threadIdx.x * Config::THREAD_TILE_ROWS +
            blockIdx.x * Config::BLOCK_TILE_ROWS;
        index_t thread_tile_base_column = threadIdx.y * Config::VALUES_PER_SHARED_LOAD +
            blockIdx.y * Config::BLOCK_TILE_COLUMNS;

        index_t thread_tile_base = thread_tile_base_row +
            thread_tile_base_column * Config::GRID_TILE_ROWS;

        index_t thread_column_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;

        index_t row_step = Config::VALUES_PER_WEIGHT_LOAD;
        index_t column_offset_step = parameters.layer_size - row_step;
        index_t column_step = thread_column_step * parameters.layer_size - column_offset_step;

        index_t thread_tile_index = thread_tile_base;

        UNROLL
        for (index_t column_segment = 0; column_segment < Config::THREAD_TILE_COLUMN_SEGMENTS;
            ++column_segment) {

            UNROLL
            for (index_t column_offset = 0; column_offset < Config::VALUES_PER_SHARED_LOAD;
                ++column_offset) {

                index_t column = column_segment * Config::VALUES_PER_SHARED_LOAD + column_offset;

                UNROLL
                for (index_t row = 0; row < Config::THREAD_TILE_ROWS; row += row_step) {

                    dprintf("Thread (%d, %d, %d, %d) - Loading thread weights[%d][%d] = "
                        "data_weights[%d]\n",
                        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                        (int)row, (int)column, (int)thread_tile_index);

                    WeightAccessType global_value = reinterpret_cast<const WeightAccessType&>(
                        parameters.weights[thread_tile_index]);

                    UNROLL
                    for (index_t r = 0; r < row_step; ++r) {

                        weights.data[row + r][column] = reinterpret_cast<RealType*>(
                            &global_value)[r];
                    }

                    thread_tile_index += row_step;
                }

                thread_tile_index += column_offset_step;
            }

            thread_tile_index += column_step;
        }

        UNROLL
        for (index_t column_offset = 0;
            column_offset < Config::THREAD_TILE_COLUMN_SEGMENT_REMAINDER;
            ++column_offset) {

            index_t column = column_offset
                + Config::THREAD_TILE_COLUMN_SEGMENTS * Config::VALUES_PER_SHARED_LOAD;

            UNROLL
            for (index_t row = 0; row < Config::THREAD_TILE_ROWS; row += row_step) {

                WeightAccessType global_value = reinterpret_cast<const WeightAccessType&>(
                    parameters.weights[thread_tile_index]);

                UNROLL
                for (index_t r = 0; r < row_step; ++r) {
                    weights.data[row + r][column] = reinterpret_cast<RealType*>(
                        &global_value)[r];
                }

                thread_tile_index += row_step;
            }

            thread_tile_index += column_offset_step;
        }
    }

    __device__ void safe_load_weights(ThreadTileWeights& weights, bool transpose) {

        index_t thread_tile_base_row = threadIdx.x * Config::THREAD_TILE_ROWS +
            blockIdx.x * Config::BLOCK_TILE_ROWS;
        index_t thread_tile_base_column = threadIdx.y * Config::VALUES_PER_SHARED_LOAD +
            blockIdx.y * Config::BLOCK_TILE_COLUMNS;

        index_t thread_column_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;

        UNROLL
        for (index_t column_segment = 0, data_column = 0;
            column_segment < Config::THREAD_TILE_COLUMN_SEGMENTS;
            ++column_segment, data_column += thread_column_step) {

            UNROLL
            for (index_t column_offset = 0; column_offset < Config::VALUES_PER_SHARED_LOAD;
                ++column_offset) {

                index_t column = column_segment * Config::VALUES_PER_SHARED_LOAD + column_offset;

                UNROLL
                for (index_t row = 0; row < Config::THREAD_TILE_ROWS; ++row) {
                    index_t current_row    = row         + thread_tile_base_row;
                    index_t current_column = data_column + thread_tile_base_column + column_offset;

                    weights.data[row][column] = 0.0;

                    bool is_in_range = current_row < parameters.layer_size &&
                        current_column < parameters.layer_size;

                    index_t thread_tile_index = 0;

                    if (transpose) {
                        thread_tile_index = current_column + current_row * parameters.layer_size;
                    }
                    else {
                        thread_tile_index = current_row + current_column * parameters.layer_size;
                    }

                    if (is_in_range) {

                        weights.data[row][column] = parameters.weights[thread_tile_index];
                        t0printf("Thread (%d, %d, %d, %d) - Loading thread weights[%d][%d] = "
                            "data_weights[%d][%d] %f\n",
                            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                            (int)row, (int)column, (int) current_row, (int)current_column,
                            (float)weights.data[row][column]);
                    }
                }

            }
        }

        index_t data_column = Config::THREAD_TILE_COLUMN_SEGMENTS * thread_column_step;

        UNROLL
        for (index_t column_offset = 0;
            column_offset < Config::THREAD_TILE_COLUMN_SEGMENT_REMAINDER;
            ++column_offset) {

            index_t column = column_offset
                + Config::THREAD_TILE_COLUMN_SEGMENTS * Config::VALUES_PER_SHARED_LOAD;

            UNROLL
            for (index_t row = 0; row < Config::THREAD_TILE_ROWS; ++row) {
                index_t current_row    = row         + thread_tile_base_row;
                index_t current_column = data_column + thread_tile_base_column + column_offset;

                weights.data[row][column] = 0.0;

                bool is_in_range = current_row < parameters.layer_size &&
                    current_column < parameters.layer_size;

                index_t thread_tile_index = current_row +
                    current_column * parameters.layer_size;

                if (is_in_range) {

                    weights.data[row][column] = parameters.weights[thread_tile_index];
                    t0printf("Thread (%d, %d, %d, %d) - Loading thread weights[%d][%d] = "
                        "data_weights[%d][%d] %f\n",
                        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                        (int)row, (int)column, (int) current_row, (int)current_column,
                        (float)weights.data[row][column]);
                }
            }

        }
    }

    __device__ void load_transposed_weights(ThreadTileWeights& weights) {
        safe_load_weights(weights, true);
    }

private:
    __device__ void format_input(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer) {

        convert_input_to_native_format(data_buffer);

        apply_nonlinearity(data_buffer);

        store_nonlinear_input_global(register_state, data_buffer);

        store_nonlinear_input_shared(register_state, shared_state, data_buffer);
    }

    __device__ void format_input_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer, DataLoadingBuffer& activation_buffer) {

        convert_input_to_native_format(data_buffer);

        apply_back_prop_nonlinearity(data_buffer, activation_buffer);

        store_nonlinear_input_global(register_state, data_buffer);

        store_nonlinear_input_shared(register_state, shared_state, data_buffer);
    }

    __device__ void convert_input_to_native_format(DataLoadingBuffer& data_buffer) {
        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            data_buffer.data[i] = is_input_thread() ?
                to_floating_point(reinterpret_cast<const IntType&>(data_buffer.data[i])) :
                data_buffer.data[i];
        }
    }

    // noinline so that the uncommon case doesn't polute the main loop
    __device__ __noinline__ void external_load_input(RegisterState register_state,
        DataLoadingBuffer& data_buffer) {

        load_input(register_state, data_buffer);
        detect_barrier_success(register_state, data_buffer);
    }

    __device__ void load_input(RegisterState& register_state, DataLoadingBuffer& data_buffer,
        bool load_output = true) {

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_GLOBAL_LOAD) {

            predicated_load_vector(register_state, data_buffer.data[i], i, load_output);
        }
    }

    __device__ void load_back_prop_activations(RegisterState& register_state,
        DataLoadingBuffer& activation_buffer) {

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_GLOBAL_LOAD) {

            predicated_load_back_prop_activation_vector(register_state,
                activation_buffer.data[i], i);
        }
    }

    __device__ void detect_barrier_success(RegisterState& register_state,
        DataLoadingBuffer& data_buffer) {

        bool performing_check = register_state.barrier_success;

        register_state.barrier_success = true;

        #pragma unrol
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            register_state.barrier_success &=
                (!performing_check) || check_barrier(register_state, data_buffer.data[i]);
        }
    }

    __device__ void apply_back_prop_nonlinearity(DataLoadingBuffer& data_buffer,
        DataLoadingBuffer& activation_buffer) {

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            if (is_input_thread()) {
                data_buffer.data[i] = parameters.config.apply_activation_derivative(
                    activation_buffer.data[i], data_buffer.data[i]);
            }
        }
    }

    __device__ void apply_nonlinearity(DataLoadingBuffer& data_buffer) {
        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            if (is_input_thread()) {
                data_buffer.data[i] = parameters.config.apply_activation_function(
                    data_buffer.data[i]);
            }
        }
    }

    __device__ void store_nonlinear_input_global(RegisterState& register_state,
        DataLoadingBuffer& data_buffer) {
        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_GLOBAL_LOAD) {

            predicated_store_vector(register_state, data_buffer.data[i], i);
        }
    }

    __device__ void store_nonlinear_input_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer) {

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_SHARED_STORE) {
            predicated_store_vector_shared(register_state, shared_state, data_buffer.data[i], i);
        }
    }

    __device__ void handle_barrier_failure(RegisterState& register_state,
        DataLoadingBuffer& data_buffer) {

        if (!register_state.barrier_success) {
            DataLoadingBuffer temp_buffer;
            RegisterState temp_state = register_state;

            spin_on_barrier_failure(temp_state, temp_buffer);
            data_buffer = temp_buffer;
            register_state = temp_state;
        }

    }

    __device__ __noinline__ void spin_on_barrier_failure(RegisterState& register_state,
        DataLoadingBuffer& data_buffer) {

        for (index_t i = 0; i < 5; ++i) {
            external_load_input(register_state, data_buffer);

            if (register_state.barrier_success) {
                break;
            }
        }
    }

    __device__ void synchronize_block() {
        __syncthreads();
    }

    __device__ void initialize_accumulators(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileAccumulators& accumulators) {

        UNROLL
        for (index_t row = 0; row < Config::THREAD_TILE_ROWS;
            row += Config::VALUES_PER_OUTPUT_SHARED_LOAD) {

            set_accumulators_to_zero(accumulators, row);

            load_output_data_from_shared(register_state, shared_state, accumulators, row);
            load_scaled_input_data_from_shared(register_state, shared_state, accumulators, row);
        }
    }

private:
    __device__ bool is_leader_thread() const {
        return threadIdx.y == 0;
    }

    __device__ index_t get_block_id_x() const {
        return blockIdx.x;
    }

    __device__ bool is_input_thread() const {
        return get_linear_thread_id() < Config::INPUT_LOAD_GROUP_SIZE;
    }

    __device__ void predicated_load_vector(RegisterState& register_state,
        RealType& value, index_t value_offset, bool load_output) {

        index_t  block_offset = get_block_id_x() * Config::BLOCK_TILE_COLUMNS;
        index_t thread_offset = get_thread_id_in_load_group() * Config::GLOBAL_VALUES_PER_THREAD;

        RealType* load_base = is_input_thread() ?
            register_state.activation_scratch :
            (Config::DIRECTION == prnn::RECURRENT_REVERSE ?
                register_state.input_base_pointer - register_state.input_to_output_offset :
                register_state.input_base_pointer + register_state.input_to_output_offset);

        index_t io_offset = value_offset + thread_offset + block_offset;
        index_t offset    = io_offset;

        GlobalAccessType loaded_data;

        UNROLL
        for (int i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            loaded_data.data[i] = 0;
        }

        bool condition = (io_offset < register_state.layer_size) &&
            (is_input_thread() || load_output);

        register_state.barrier_success = condition && is_input_thread();

        for (int i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            if (condition) {
                dprintf("Thread (%d, %d, %d, %d) - Loading %s activation[%d] "
                    "(%d block, %d thread, %d io) (%p) = %f (%s)\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    is_input_thread() ? "input" : "output",
                    offset, block_offset, thread_offset, io_offset + i,
                    load_base + offset + i,
                    is_input_thread() ?
                    to_floating_point(reinterpret_cast<const IntType&>(loaded_data.data[i])) :
                    (float)loaded_data.data[i],
                    condition ? "enabled" : "disabled");
            }
        }

        predicated_atomic_global_load_relaxed(loaded_data, *reinterpret_cast<GlobalAccessType*>(
            load_base + offset), condition);

        reinterpret_cast<GlobalAccessType&>(value) = loaded_data;
    }

    __device__ void predicated_load_back_prop_activation_vector(RegisterState& register_state,
        RealType& value, index_t value_offset) {

        index_t  block_offset = get_block_id_x() * Config::BLOCK_TILE_COLUMNS;
        index_t thread_offset = get_thread_id_in_load_group() * Config::GLOBAL_VALUES_PER_THREAD;

        RealType* load_base = register_state.back_prop_activation_base_pointer;

        index_t io_offset = value_offset + thread_offset + block_offset;
        index_t offset    = io_offset;

        GlobalAccessType loaded_data;

        UNROLL
        for (int i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            loaded_data.data[i] = 0;
        }

        bool condition = (io_offset < register_state.layer_size) && is_input_thread();

        predicated_atomic_global_load_relaxed(loaded_data, *reinterpret_cast<GlobalAccessType*>(
            load_base + offset), condition);

        for (int i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            if (condition) {
                dprintf("Thread (%d, %d, %d, %d) - Loading back prop activation[%d] "
                    "(%d block, %d thread, %d io) (%p) = %f (%s)\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset, block_offset, thread_offset, io_offset + i,
                    load_base + offset + i,
                    (float)loaded_data.data[i],
                    condition ? "enabled" : "disabled");
            }
        }

        reinterpret_cast<GlobalAccessType&>(value) = loaded_data;
    }

    __device__ void predicated_store_vector(RegisterState& register_state,
        RealType& data, index_t value_offset) {

        RealType* output_base = get_output_pointer(register_state);

        index_t block_offset  = get_block_id_x() * Config::BLOCK_TILE_COLUMNS;
        index_t thread_offset = get_linear_thread_id() * Config::GLOBAL_VALUES_PER_THREAD;

        index_t offset = thread_offset + block_offset + value_offset;

        bool condition = (offset < register_state.layer_size) && is_input_thread();

        predicated_atomic_global_store_relaxed(
            *reinterpret_cast<GlobalAccessType*>(output_base + offset),
            reinterpret_cast<GlobalAccessType&>(data), condition);

        for (int i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i) {
            if (condition) {
                dprintf("Thread (%d, %d, %d, %d) - Saving final activation[%d] "
                    "(%d block, %d thread, %d value) (%p) = %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset + i, block_offset, thread_offset, value_offset + i,
                    output_base + offset + i,
                    (float)reinterpret_cast<GlobalAccessType&>(data).data[i]);
            }
        }
    }

    __device__ void predicated_store_vector_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        const RealType& data, index_t value_offset) {

        index_t thread_offset = get_linear_thread_id() * Config::GLOBAL_VALUES_PER_THREAD;

        index_t offset = thread_offset + value_offset + register_state.shared_base;

        for (int i = 0; i < Config::VALUES_PER_SHARED_STORE; ++i) {
            index_t block_offset = get_block_id_x() * Config::BLOCK_TILE_COLUMNS;
            index_t offset_in_group = get_thread_id_in_load_group() *
                Config::GLOBAL_VALUES_PER_THREAD + value_offset + block_offset + i;

            if (offset_in_group < register_state.layer_size) {
                dprintf("Thread (%d, %d, %d, %d) - Storing loaded value to shared [%d] "
                    "(group offset %d) = %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset + i, offset_in_group,
                    (float)reinterpret_cast<const SharedStoreType&>(data).data[i]);
            }
        }

        reinterpret_cast<SharedStoreType&>(shared_state.data[offset]) =
            reinterpret_cast<const SharedStoreType&>(data);
    }

    __device__ index_t get_thread_id_in_load_group() const {
        index_t my_thread_id = get_linear_thread_id();

        bool is_in_input_load_group = my_thread_id < Config::INPUT_LOAD_GROUP_SIZE;

        return is_in_input_load_group ? my_thread_id :
            my_thread_id - Config::INPUT_LOAD_GROUP_SIZE;
    }

    __device__ index_t get_linear_thread_id() const {
        return threadIdx.x + threadIdx.y * Config::THREADS_PER_COLUMN;
    }

    __device__ bool check_barrier(RegisterState& register_state, const RealType& value) const {
        index_t counter = extract_counter(value);

        t0printf("Thread (%d, %d, %d, %d) - Checking barrier counter %d against requirement %d\n",
            blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
            counter, register_state.reduction_threads_per_value);

        return counter >= register_state.reduction_threads_per_value;
    }

    __device__ void set_accumulators_to_zero(ThreadTileAccumulators& accumulators, index_t start) {
        for (index_t row = 0; row < Config::VALUES_PER_OUTPUT_SHARED_LOAD; ++row) {
            accumulators.data[start + row] = 0;
        }
    }

    __device__ void load_output_data_from_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileAccumulators& accumulators, index_t row) {

        index_t thread_offset = threadIdx.x * Config::THREAD_TILE_ROWS;

        index_t shared_output_offset = Config::BLOCK_TILE_COLUMNS;

        index_t shared_offset = register_state.shared_base + shared_output_offset +
            thread_offset + row;

        bool condition = is_leader_thread();

        predicated_atomic_shared_load_relaxed(
            reinterpret_cast<OutputSharedAccessType&>(accumulators.data[row]),
            reinterpret_cast<OutputSharedAccessType&>(shared_state.data[shared_offset]),
            condition);

        UNROLL
        for (index_t r = 0; r < Config::VALUES_PER_OUTPUT_SHARED_LOAD; ++r) {

            t0printf("Thread (%d, %d, %d, %d) - Loading tile outputs "
                "from shared memory at %d (offset %d) = %f.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, r + row, shared_offset + r,
                (float)(accumulators.data[row + r]));
        }
    }

    __device__ void load_scaled_input_data_from_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileAccumulators& accumulators, index_t row) {

        index_t thread_offset = threadIdx.y * Config::THREAD_TILE_ROWS;
        index_t shared_offset = register_state.shared_base + thread_offset + row;

        OutputSharedAccessType temp;

        UNROLL
        for(index_t i = 0; i < Config::VALUES_PER_OUTPUT_SHARED_LOAD; ++i) {
            reinterpret_cast<RealType*>(&temp)[i] = 0.0;
        }

        bool condition = is_leader_thread();

        predicated_atomic_shared_load_relaxed(temp, reinterpret_cast<OutputSharedAccessType&>(
            shared_state.data[shared_offset]), condition);

        UNROLL
        for(index_t i = 0; i < Config::VALUES_PER_OUTPUT_SHARED_LOAD; ++i) {

            auto value = reinterpret_cast<RealType*>(&temp)[i];

            auto result = value *
                register_state.skip_connection_scale + accumulators.data[row + i];

            t0printf("Thread (%d, %d, %d, %d) - Loading scaled input data from shared "
                "memory at (offset %d) %f = value (%f) * scale (%f) + output (%f).\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, shared_offset + i,
                (float)result, (float)value, (float)register_state.skip_connection_scale,
                (float)accumulators.data[row + i]);

            accumulators.data[row + i] = result;
        }
    }

private:
    __device__ void load_thread_tile_inputs(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileInputs& thread_inputs) {

        index_t thread_offset = register_state.shared_base +
            Config::VALUES_PER_SHARED_LOAD * threadIdx.y;

        index_t thread_offset_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;

        UNROLL
        for (index_t column = 0; column < Config::THREAD_TILE_COLUMNS;
            column += Config::VALUES_PER_SHARED_LOAD, thread_offset += thread_offset_step) {

            auto value = reinterpret_cast<SharedAccessType&>(
                shared_state.data[thread_offset]);

            for(index_t i = 0; i < Config::VALUES_PER_SHARED_LOAD; ++i) {
                t0printf("Thread (%d, %d, %d, %d) - Loading tile inputs from shared memory "
                    "at %d = %f.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    thread_offset + i, value.data[i]);
            }

            reinterpret_cast<SharedAccessType&>(thread_inputs.data[column]) = value;
        }
    }

    __device__ void perform_thread_tile_math(ThreadTileAccumulators& accumulators,
        ThreadTileWeights& weights, ThreadTileInputs& thread_inputs) {
        UNROLL
        for (index_t row = 0; row < Config::THREAD_TILE_ROWS; row += Config::SIMD) {
            UNROLL
            for (index_t column = 0; column < Config::THREAD_TILE_COLUMNS; ++column) {

                simd_ffma<RealType>(accumulators, weights, thread_inputs, row, column);
            }
        }
    }

    template<typename T>
    __device__
    typename std::enable_if<std::is_same<T, float16>::value>::type
    simd_ffma(ThreadTileAccumulators& accumulators,
        ThreadTileWeights& weights, ThreadTileInputs& thread_inputs,
        index_t row, index_t column) {

        fp16x2 simd_column_data;

        UNROLL
        for (index_t s = 0; s < Config::SIMD; ++s) {
            reinterpret_cast<float16*>(&simd_column_data)[s] = thread_inputs[column];
        }

        to_fp16x2(accumulators.data[row]) = packed_fp16x2_ffma<RealType>(
            to_fp16x2(weights.data[row][column]),
            simd_column_data,
            to_fp16x2(accumulators.data[row])
            );
    }

    template<typename T>
    __device__
    typename std::enable_if<!std::is_same<T, float16>::value>::type
    simd_ffma(ThreadTileAccumulators& accumulators,
        ThreadTileWeights& weights, ThreadTileInputs& thread_inputs,
        index_t row, index_t column) {

        UNROLL
        for (index_t r = 0; r < Config::SIMD; ++r) {

            index_t row_base = threadIdx.x * Config::THREAD_TILE_ROWS +
                blockIdx.x * Config::BLOCK_TILE_ROWS;
            index_t column_base = threadIdx.y * Config::VALUES_PER_SHARED_LOAD +
                blockIdx.y * Config::BLOCK_TILE_COLUMNS;

            index_t thread_offset_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;
            index_t column_segment = column / Config::VALUES_PER_SHARED_LOAD;
            index_t column_offset  = column_segment * thread_offset_step;


            bool condition = (row_base + row + r < parameters.layer_size) &&
                (column_base + column + column_offset < parameters.layer_size);

            if (condition) {
                t0printf("Thread (%d, %d, %d, %d) - accumulator[%d] %f = weight[%d, %d] %f * "
                    "activation[%d] %f + accumulator[%d] %f.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row + r,
                    (float)(accumulators.data[row + r] + weights.data[row + r][column] *
                        thread_inputs.data[column]),
                    row + r, column, weights.data[row + r][column],
                    column, (float)(thread_inputs.data[column]),
                    row + r, (float)accumulators.data[row + r]);
            }

            accumulators.data[row + r] +=
                weights.data[row + r][column] *
                thread_inputs.data[column];
        }
    }

    __device__ void store_accumulators(RegisterState& register_state,
        ThreadTileAccumulators& accumulators) {

        auto* output_pointer = get_thread_output_pointer(register_state, 0);

        atomic_increment_accumulators(register_state, accumulators, output_pointer);
    }

    __device__ void store_accumulators_back_prop(RegisterState& register_state,
        ThreadTileAccumulators& accumulators) {

        auto* output_pointer = get_back_prop_thread_output_pointer(register_state, 0);

        atomic_increment_accumulators(register_state, accumulators, output_pointer);
    }

    __device__ void atomic_increment_accumulators(RegisterState& register_state,
        ThreadTileAccumulators& accumulators, RealType* output_pointer) {

        UNROLL
        for (index_t row = 0; row < Config::THREAD_TILE_ROWS; row += 1) {
            atomic_increment_reduce_relaxed(*reinterpret_cast<IntType*>(output_pointer),
                to_fixed_point_with_counter(accumulators.data[row], 1));

            t0printf("Thread (%d, %d, %d, %d) - atomic increment %d (%p) "
                "%f (%x) = %f (%x).\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, output_pointer,
                (float)to_floating_point(result +
                    to_fixed_point_with_counter(accumulators.data[row], 1)),
                (int)(result + to_fixed_point_with_counter(accumulators.data[row], 1)),
                (float)accumulators.data[row],
                (int)(to_fixed_point_with_counter(accumulators.data[row], 1)));

            output_pointer += 1;
        }
    }

private:
    __device__ IntType add_counter(IntType t, IntType counter) const {
        IntType shift = Config::FIXED_POINT_COUNTER_BITS;

        return (t << shift) | counter;
    }

    __device__ IntType remove_counter(IntType t) const {
        IntType shift = Config::FIXED_POINT_COUNTER_BITS;

        return t >> shift;
    }

    __device__ IntType extract_counter(const RealType& t) const {
        IntType mask = (((IntType)1) << (Config::FIXED_POINT_COUNTER_BITS)) - 1;

        return reinterpret_cast<const IntType&>(t) & mask;
    }

    __device__ IntType to_fixed_point_with_counter(RealType value, index_t counter) const {
        FixedPoint fixed = value;

        return add_counter(fixed.raw_data(), counter);
    }

    __device__ RealType to_floating_point(IntType data) const
    {
        FixedPoint fixed(remove_counter(data));

        RealType result = fixed.template extract<RealType>();

        return result;
    }

    __device__ fp16x2& to_fp16x2(RealType& data) const {
        return reinterpret_cast<fp16x2&>(data);
    }

    template<typename T>
    __device__
    typename std::enable_if<std::is_same<T, float16>::value, fp16x2>::type
    packed_fp16x2_ffma(fp16x2 a, fp16x2 b, fp16x2 c) const {
        fp16x2 result;

        asm("fma.rn.f16x2" : "=f"(result) : "f"(a), "f"(b), "f"(c) );

        return result;
    }

    __device__ RealType* get_thread_output_pointer(RegisterState& register_state, index_t row) {
        RealType* output_buffer = register_state.activation_scratch +
            register_state.scratch_input_to_output_offset -
            Config::GRID_TILE_ROWS;

        index_t blockId = blockIdx.x;

        index_t blockOffset = blockId * Config::BLOCK_TILE_ROWS;

        index_t threadId = threadIdx.x;

        index_t threadOffset = threadId * Config::THREAD_TILE_ROWS;

        return output_buffer + blockOffset + threadOffset + row;
    }

    __device__ RealType* get_back_prop_thread_output_pointer(RegisterState& register_state,
        index_t row) {

        RealType* output_buffer = register_state.activation_scratch -
            register_state.scratch_input_to_output_offset +
            Config::GRID_TILE_ROWS;

        index_t blockId = blockIdx.x;

        index_t blockOffset = blockId * Config::BLOCK_TILE_ROWS;

        index_t threadId = threadIdx.x;

        index_t threadOffset = threadId * Config::THREAD_TILE_ROWS;

        return output_buffer + blockOffset + threadOffset + row;
    }

    __device__ RealType* get_output_pointer(RegisterState& register_state) {
        return register_state.input_base_pointer;
    }

private:
    __device__ void advance_pointers(RegisterState& register_state) {
        register_state.input_base_pointer += register_state.iteration_step;

        register_state.activation_scratch += register_state.scratch_step_size;
    }

    __device__ void advance_pointers_back_prop(RegisterState& register_state) {
        register_state.input_base_pointer -= register_state.iteration_step;
        register_state.back_prop_activation_base_pointer -= register_state.iteration_step;

        register_state.activation_scratch -= register_state.scratch_step_size;
    }

    __device__ void advance_shared_pointers(RegisterState& register_state) {
        register_state.shared_base ^= Config::SHARED_BUFFER_SIZE;
    }

    __device__ index_t get_scratch_offset(index_t offset) {
        index_t layer     = offset / parameters.layer_size;
        index_t remainder = offset % parameters.layer_size;

        return layer * Config::GRID_TILE_ROWS + remainder;;
    }

private:
    const PersistentEngineParameters<Config> parameters;
    GpuSynchronizer synchronizer;

};


template<typename Config>
__launch_bounds__(Config::THREADS_PER_BLOCK, Config::BLOCKS_PER_SM)
__global__ void forward_prop_recurrent_kernel(PersistentEngineParameters<Config> parameters) {
    PersistentEngine<Config> engine(parameters);

    engine.run_forward();
}

template<typename Config>
__launch_bounds__(Config::THREADS_PER_BLOCK, Config::BLOCKS_PER_SM)
__global__ void back_prop_recurrent_deltas_kernel(PersistentEngineParameters<Config> parameters) {
    PersistentEngine<Config> engine(parameters);

    engine.run_back_prop_deltas();
}

}
}



