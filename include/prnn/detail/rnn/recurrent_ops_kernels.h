#pragma once

// Persistent RNN Includes
#include <prnn/detail/rnn/recurrent_ops_config.h>
#include <prnn/detail/rnn/synchronizer.h>

#include <prnn/detail/util/atomics.h>

#define DEBUG_RECURRENT_OPS 0
#define ATOMIC_INCREMENT 1
#define USE_BARRIER 1
#define SHOULD_SPIN 1
#define INITIALIZE_OUTPUT_ACCUMULATORS 1
#define REDUCE_ACCUMULATORS 1
#define REDUCE_ADDRESS_MATH 1
#define BARRIER_ALWAYS_FAILS 0
#define WAIT_FOREVER 0

#if DEBUG_RECURRENT_OPS

#define dprintf(...) do { if( blockIdx.x == 0 && blockIdx.y == 0 ) \
    { std::printf(__VA_ARGS__); } } while(0)

#define t0printf(...) do { if(threadIdx.x == 0 && (threadIdx.y == 0) && \
    blockIdx.x == 0 && blockIdx.y == 0 ) { std::printf(__VA_ARGS__); } } while(0)

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
      expanded_layer_size(expand_size(config.handle.layerSize)),
      mini_batch_size(config.handle.miniBatchSize),
      timesteps(config.handle.timesteps),
      is_fully_covered(config.handle.layerSize % Config::BLOCK_TILE_COLUMNS == 0),
      synchronizer(synchronizer_),
      config(config)
    {
        iteration_step = layer_size;
        input_to_output_offset = layer_size * (mini_batch_size);
        scratch_input_to_output_offset = Config::EXPANDED_GRID_TILE_ROWS * (mini_batch_size);
        iterations = mini_batch_size * (timesteps - 1);

        first_iteration = synchronizer_.get_current_phase();

        scratch_step_size = Config::EXPANDED_GRID_TILE_ROWS;

        reduction_threads_per_value = (layer_size + Config::BLOCK_TILE_COLUMNS - 1) /
            Config::BLOCK_TILE_COLUMNS;
    }

    index_t expand_size(index_t size) const {
        return size * Config::CACHE_LINE_USAGE;
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
    index_t         expanded_layer_size;
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
    RealType reduction_threads_per_value;

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

    typedef typename Config::ThreadTileWeights       ThreadTileWeights;
    typedef typename Config::ThreadTileInputs        ThreadTileInputs;
    typedef typename Config::ThreadTileAccumulators  ThreadTileAccumulators;
    typedef typename Config::DataLoadingBuffer       DataLoadingBuffer;
    typedef typename Config::ActivationLoadingBuffer ActivationLoadingBuffer;
    typedef typename Config::SharedDataStorage       SharedDataStorage;

    typedef typename Config::ThreadTileOutputAccumulators ThreadTileOutputAccumulators;

    typedef typename Config::GlobalAccessType GlobalAccessType;
    typedef typename Config::GlobalStoreType  GlobalStoreType;
    typedef typename Config::SharedAccessType SharedAccessType;
    typedef typename Config::WeightAccessType WeightAccessType;

    typedef typename Config::SharedInputStoreType   SharedInputStoreType;
    typedef typename Config::SharedOutputStoreType  SharedOutputStoreType;

    typedef typename Config::ActivationAccessType ActivationAccessType;
    typedef typename Config::SharedAccumulatorStoreType SharedAccumulatorStoreType;
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
          should_check_barrier(false),
          skip_connection_scale(parameters.skip_connection_scale),
          layer_size(parameters.layer_size),
          expanded_layer_size(parameters.expanded_layer_size),
          input_to_output_offset(parameters.input_to_output_offset),
          scratch_input_to_output_offset(parameters.scratch_input_to_output_offset),
          iteration_step(parameters.iteration_step),
          iterations(parameters.iterations),
          iteration(parameters.first_iteration),
          first_iteration(parameters.first_iteration),
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
                    Config::EXPANDED_GRID_TILE_COLUMNS * iteration;
            }
            else {
                back_prop_activation_base_pointer = parameters.back_prop_activations +
                    parameters.first_iteration * parameters.layer_size;
                input_base_pointer = parameters.activations +
                    parameters.first_iteration * parameters.layer_size;

                activation_scratch = parameters.activation_scratch +
                    Config::EXPANDED_GRID_TILE_COLUMNS * parameters.first_iteration;
            }
        }

    public:
        RealType* activation_scratch;

    public:
        index_t shared_base; // 1 reg

    public:
        bool barrier_success;
        bool should_check_barrier;

    public:
        RealType* back_prop_activation_base_pointer;
        RealType* input_base_pointer; // 2 regs

    public:
        RealType skip_connection_scale; // const
        index_t layer_size; // const
        index_t expanded_layer_size; // const

    public:
        index_t input_to_output_offset; // const
        index_t scratch_input_to_output_offset; // const
        index_t iteration_step; // const
        index_t iterations; // const

    public:
        index_t iteration; // 1 reg
        index_t first_iteration; // const

    public:
        index_t scratch_step_size; // const

    public:
        RealType reduction_threads_per_value; // const

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

        DataLoadingBuffer data_buffer; // 4 regs
        ThreadTileAccumulators accumulators; // 9 regs
        ThreadTileOutputAccumulators output_accumulators; // 3 regs

        RegisterState register_state(parameters);

        __shared__ SharedDataStorage shared_state;

        initialize_shared_state(shared_state);

        ThreadTileWeights weights;

        load_weights(weights);

        warm_start(register_state, shared_state, weights, data_buffer, accumulators,
            output_accumulators);

        if(register_state.barrier_success)
        {
            for(; register_state.iteration < register_state.iterations; ++register_state.iteration)
            {
                t0printf("Thread (%d, %d, %d, %d) - Starting iteration %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);

                perform_iteration(register_state, shared_state, weights, data_buffer, accumulators,
                    output_accumulators);

                if(!register_state.barrier_success)
                {
                    t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing"
                        " out of main loop.\n",
                        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
                    register_state.iteration -= 1;
                    break;
                }
            }
        }

        if(register_state.barrier_success)
        {
            clean_up(register_state, shared_state, weights, data_buffer, accumulators,
                output_accumulators);
        }

        if(!register_state.barrier_success)
        {
            t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out of kernel, "
                "restart at %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);
            #if USE_BARRIER
            synchronizer.set_concurrent_execution_failed();
            synchronizer.set_phase(register_state.iteration);
            #endif
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
        ActivationLoadingBuffer activation_buffer;
        ThreadTileAccumulators accumulators;
        ThreadTileOutputAccumulators output_accumulators;

        RegisterState register_state(parameters);

        __shared__ SharedDataStorage shared_state;

        initialize_shared_state(shared_state);

        ThreadTileWeights weights;

        load_transposed_weights(weights);

        warm_start_back_prop(register_state, shared_state, weights,
            data_buffer, activation_buffer, accumulators, output_accumulators);

        if(register_state.barrier_success)
        {
            for(; register_state.iteration < register_state.iterations; ++register_state.iteration)
            {
                t0printf("Thread (%d, %d, %d, %d) - Starting iteration %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);

                perform_back_prop_iteration(register_state, shared_state,
                    weights, data_buffer, activation_buffer,
                    accumulators, output_accumulators);

                if (!register_state.barrier_success) {
                    t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out"
                        " of main loop.\n",
                        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
                    break;
                }
            }
        }

        if(register_state.barrier_success)
        {
            clean_up_back_prop(register_state, shared_state,
                weights, data_buffer, activation_buffer,
                accumulators, output_accumulators);
        }

        if(!register_state.barrier_success)
        {
            printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out of kernel at %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, (register_state.iteration - 1));
            synchronizer.set_concurrent_execution_failed();
            synchronizer.set_phase(register_state.iteration - 1);
        }
    }

private:
    __device__ void perform_iteration(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators,
        bool save_input = true,
        bool load_output = true,
        bool stage_one = true,
        bool stage_two = true,
        bool stage_three = true)
    {
        ThreadTileInputs thread_inputs;

        // 0
        if(stage_one)
        {
            if(!check_for_critical_barrier_failure(register_state, shared_state))
            {
                return;
            }
        }

        // 0
        if(stage_one)
        {
            load_input(register_state, data_buffer, load_output);
        }

        // 1
        if(stage_two)
        {
            load_thread_tile_inputs(register_state, shared_state, thread_inputs);
            initialize_accumulators(accumulators);
            perform_thread_tile_math(accumulators, weights, thread_inputs);
        }

        // 2
        if(stage_three)
        {
            reduce_thread_tile_shared(register_state, shared_state, output_accumulators);
        }

        // 2
        if(stage_three)
        {
            store_accumulators(register_state, output_accumulators);
        }

        // 1
        if(stage_two)
        {
            initialize_output_accumulators(register_state, shared_state, output_accumulators);
        }

        advance_shared_pointers(register_state);

        // 1
        if(stage_two)
        {
            store_accumulators_to_shared(register_state, shared_state, accumulators);
        }

        // 0
        if(stage_one)
        {
            wait_for_barrier(register_state, shared_state, data_buffer);
        }

        // 0
        if(stage_one)
        {
            format_input(register_state, shared_state, data_buffer, save_input);
        }

        synchronize_block();
        advance_pointers(register_state);
    }

    __device__ void perform_back_prop_iteration(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        ActivationLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators,
        bool save_input = true,
        bool load_output = true,
        bool stage_one = true,
        bool stage_two = true,
        bool stage_three = true)
    {
        ThreadTileInputs thread_inputs;

        // 0
        if(stage_one)
        {
            if(!check_for_critical_barrier_failure(register_state, shared_state))
            {
                return;
            }
        }

        // 0
        if(stage_one)
        {
            load_input(register_state, data_buffer, load_output);
            load_back_prop_activations(register_state, activation_buffer);
        }

        // 1
        if(stage_two)
        {
            load_thread_tile_inputs(register_state, shared_state, thread_inputs);
            initialize_accumulators(accumulators);
            perform_thread_tile_math(accumulators, weights, thread_inputs);
        }

        // 2
        if(stage_three)
        {
            reduce_thread_tile_shared(register_state, shared_state, output_accumulators);
        }

        // 2
        if(stage_three)
        {
            store_accumulators_back_prop(register_state, output_accumulators);
        }

        // 1
        if(stage_two)
        {
            initialize_output_accumulators(register_state, shared_state, output_accumulators);
        }

        advance_shared_pointers(register_state);

        // 1
        if(stage_two)
        {
            store_accumulators_to_shared(register_state, shared_state, accumulators);
        }

        // 0
        if(stage_one)
        {
            wait_for_barrier(register_state, shared_state, data_buffer);
        }

        // 0
        if(stage_one)
        {
            format_input_back_prop(register_state, shared_state, data_buffer, activation_buffer,
                save_input);
        }

        synchronize_block();
        advance_pointers_back_prop(register_state);
    }

private:
    __device__ void warm_start(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights, DataLoadingBuffer& data_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators) {

        for(; register_state.iteration < parameters.first_iteration + 2;
            ++register_state.iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Warm starting iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                register_state.iteration);

            perform_iteration(register_state, shared_state, weights, data_buffer,
                accumulators, output_accumulators, !is_restarted(), true,
                true, register_state.iteration > parameters.first_iteration, false);

            if(!register_state.barrier_success)
            {
                register_state.iteration -= 1;
                t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out of "
                    "warmup loop at %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);
                break;
            }
        }
    }

    __device__ void warm_start_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights,
        DataLoadingBuffer& data_buffer,
        ActivationLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators) {

        for(; register_state.iteration < parameters.first_iteration + 2;
            ++register_state.iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Warm starting back prop iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                register_state.iteration);

            perform_back_prop_iteration(register_state, shared_state, weights, data_buffer,
                activation_buffer, accumulators, output_accumulators, !is_restarted(), true,
                true, register_state.iteration > parameters.first_iteration, false);

            if(!register_state.barrier_success)
            {
                register_state.iteration -= 1;
                t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out of "
                    "warmup loop at %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);
                break;
            }
        }
    }

private:
    __device__ void clean_up(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights,
        DataLoadingBuffer& data_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators) {

        for(; register_state.iteration < register_state.iterations + parameters.mini_batch_size;
            ++register_state.iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Clean up iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);

            bool should_store_accumulators = register_state.iteration
                < register_state.iterations + 2;

            perform_iteration(register_state, shared_state,
                weights, data_buffer, accumulators, output_accumulators,
                true, false, true, true, should_store_accumulators);

            if(!register_state.barrier_success)
            {
                register_state.iteration -= 1;
                t0printf("Thread (%d, %d, %d, %d) - Barrier failed, bailing out of "
                    "cleanup loop at %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);
                break;
            }
        }
    }

    __device__ void clean_up_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileWeights& weights,
        DataLoadingBuffer& data_buffer,
        ActivationLoadingBuffer& activation_buffer,
        ThreadTileAccumulators& accumulators,
        ThreadTileOutputAccumulators& output_accumulators) {

        for(; register_state.iteration < parameters.iterations + parameters.mini_batch_size;
            ++register_state.iteration) {

            t0printf("Thread (%d, %d, %d, %d) - Clean up back prop iteration %d.\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, register_state.iteration);

            bool should_store_accumulators =
                register_state.iteration < register_state.iterations + 2;

            perform_back_prop_iteration(register_state, shared_state,
                weights, data_buffer, activation_buffer,
                accumulators, output_accumulators,
                true, false, true, true, should_store_accumulators);
        }
    }

private:
    __device__ void initialize_shared_state(SharedDataStorage& shared_state)
    {
        shared_state.data[Config::SHARED_BARRIER_OFFSET] = 0;
        shared_state.data[Config::SHARED_BUFFER_SIZE + Config::SHARED_BARRIER_OFFSET] = 0;

        synchronize_block();
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

        index_t expanded_layer_size = parameters.expanded_layer_size;

        for (index_t offset = id;
            offset < expanded_layer_size * parameters.mini_batch_size;
            offset += size) {

            index_t position_in_layer = offset % expanded_layer_size;
            index_t layer_id          = offset / expanded_layer_size;

            index_t compressed_position_in_layer = compress_id(position_in_layer);

            index_t local_index   = layer_id * parameters.layer_size + compressed_position_in_layer;
            index_t scratch_index = layer_id * Config::EXPANDED_GRID_TILE_COLUMNS +
                position_in_layer;

            bool is_in_range = compressed_position_in_layer < parameters.layer_size;

            RealType value = 0.0;

            if (is_in_range) {
                value = parameters.activations[local_index];
            }

            if (is_barrier_id(offset)) {
                value = Config::THREADS_PER_GLOBAL_REDUCTION;
            }

            dprintf("Thread (%d, %d, %d, %d) - Warm starting scratch "
                "offset[%d] (%p) = activations[%d] (%p) (%f) \n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                scratch_index, parameters.activation_scratch + scratch_index,
                local_index, parameters.activations + local_index, (float)value);

            atomic_store_relaxed(parameters.activation_scratch[scratch_index], value);
        }
    }

private:
    __device__ void populate_scratch_back_prop() {
        index_t id   = thread_id_in_grid();
        index_t size = grid_size();

        auto* deltas_base = parameters.get_deltas() +
            (parameters.timesteps - 1) * parameters.mini_batch_size * parameters.layer_size;
        auto* deltas_scratch_base = parameters.get_deltas_scratch() +
            (parameters.timesteps - 1) * parameters.mini_batch_size *
            Config::EXPANDED_GRID_TILE_ROWS;

        index_t expanded_layer_size = parameters.expanded_layer_size;

        for (index_t offset = id;
            offset < expanded_layer_size * parameters.mini_batch_size;
            offset += size) {

            index_t position_in_layer = offset % expanded_layer_size;
            index_t layer_id          = offset / expanded_layer_size;

            index_t compressed_position_in_layer = compress_id(position_in_layer);

            index_t local_index   = layer_id * parameters.layer_size + compressed_position_in_layer;
            index_t scratch_index = layer_id * Config::EXPANDED_GRID_TILE_COLUMNS +
                position_in_layer;

            bool is_in_range = compressed_position_in_layer < parameters.layer_size;

            RealType value = 0.0;

            if (is_in_range) {
                value = deltas_base[local_index];
            }

            if (is_barrier_id(offset)) {
                value = Config::THREADS_PER_GLOBAL_REDUCTION;
            }

            dprintf("Thread (%d, %d, %d, %d) - Warm starting scratch "
                "offset[%d] (%p) = deltas[%d] (%p) (%f) \n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                scratch_index,
                deltas_scratch_base + scratch_index,
                local_index, deltas_base + local_index, (float)value);

            atomic_store_relaxed(deltas_scratch_base[scratch_index], value);
        }
    }

private:
    __device__ void load_weights(ThreadTileWeights& weights) {
        safe_load_weights(weights, false);
    }

    __device__ void safe_load_weights(ThreadTileWeights& weights, bool transpose) {

        index_t thread_tile_base_row = threadIdx.x +
            get_block_id_x() * Config::BLOCK_TILE_ROWS;
        index_t thread_tile_base_column = threadIdx.y * Config::VALUES_PER_SHARED_LOAD +
            get_block_id_y() * Config::BLOCK_TILE_COLUMNS;

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
                    index_t data_row = row * Config::THREADS_PER_COLUMN;

                    index_t current_row    = data_row    + thread_tile_base_row;
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
                index_t data_row = row * Config::THREADS_PER_COLUMN;

                index_t current_row    = data_row    + thread_tile_base_row;
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

    __device__ void load_transposed_weights(ThreadTileWeights& weights)
    {
        safe_load_weights(weights, true);
    }

private:
    __device__ void format_input(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer, bool save_input)
    {
        compress_data_buffer(data_buffer);

        apply_nonlinearity(data_buffer);

        if(save_input)
        {
            store_nonlinear_input_global(register_state, data_buffer);
        }

        store_nonlinear_input_shared(register_state, shared_state, data_buffer);
    }

    __device__ void format_input_back_prop(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer, ActivationLoadingBuffer& activation_buffer,
        bool save_input)
    {
        compress_data_buffer(data_buffer);

        apply_back_prop_nonlinearity(data_buffer, activation_buffer);

        if(save_input)
        {
            store_nonlinear_input_global(register_state, data_buffer);
        }

        store_nonlinear_input_shared(register_state, shared_state, data_buffer);
    }

    // noinline so that the uncommon case doesn't polute the main loop
    __device__ __noinline__ void external_load_input(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer) {

        load_input(register_state, data_buffer, true, true);
        detect_barrier_success(register_state, shared_state, data_buffer, true);
    }

    __device__ void load_input(RegisterState& register_state, DataLoadingBuffer& data_buffer,
        bool load_output = true, bool is_retry = false) {

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_GLOBAL_LOAD) {

            predicated_load_vector(register_state, data_buffer.data[i], i, load_output, is_retry);
        }
    }

    __device__ void store_accumulators_to_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileAccumulators& accumulators) {

        UNROLL
        for(index_t i = 0; i < Config::THREAD_TILE_ROWS; ++i)
        {
            store_thread_accumulators_to_shared(register_state,
                shared_state, accumulators.data[i], i);
        }
    }

    __device__ void initialize_output_accumulators(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileOutputAccumulators& accumulators) {

        UNROLL
        for (index_t i = 0; i < Config::OUTPUTS_PER_THREAD; ++i) {
            initialize_output_accumulator(register_state, shared_state, accumulators.data[i], i);
        }
    }

    __device__ void reduce_thread_tile_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileOutputAccumulators& accumulators) {

        UNROLL
        for (index_t i = 0; i < Config::OUTPUTS_PER_THREAD; ++i) {
            reduce_accumulator(register_state, shared_state, accumulators.data[i], i);
        }
    }

    __device__ void load_back_prop_activations(RegisterState& register_state,
        ActivationLoadingBuffer& activation_buffer)
    {

        UNROLL
        for (index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_ACTIVATION_LOAD)
        {
            predicated_load_back_prop_activation_vector(register_state,
                activation_buffer.data[i], i);
        }
    }

    __device__ void wait_for_barrier(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer)
    {
        detect_barrier_success(register_state, shared_state, data_buffer);

        #if USE_BARRIER
        #if BARRIER_ALWAYS_FAILS
        register_state.barrier_success =
            register_state.iteration < (register_state.first_iteration + 2);
        #else
        #if SHOULD_SPIN
        if(!register_state.barrier_success)
        {
            DataLoadingBuffer temp_buffer;
            RegisterState temp_state = register_state;

            spin_on_barrier_failure(temp_state, shared_state, temp_buffer);
            data_buffer = temp_buffer;
            register_state.barrier_success = temp_state.barrier_success;
        }
        #endif
        #endif

        index_t shared_offset = register_state.shared_base + Config::SHARED_BARRIER_OFFSET;

        if(!register_state.barrier_success)
        {
            shared_state.data[shared_offset] = RealType(1.0);
        }
        #endif
    }

    __device__ void detect_barrier_success(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer,
        bool is_retry = false)
    {
        register_state.barrier_success = true;

        #if USE_BARRIER
        bool performing_check = register_state.should_check_barrier;

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD; ++i)
        {
            register_state.barrier_success &=
                (!performing_check) || check_barrier(register_state, data_buffer.data[i], i,
                    is_retry);
        }
        #endif
    }

    __device__ void apply_back_prop_nonlinearity(DataLoadingBuffer& data_buffer,
        ActivationLoadingBuffer& activation_buffer)
    {

        UNROLL
        for (index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD; ++i)
        {
            if(is_input_thread())
            {
                data_buffer.data[i] = parameters.config.apply_activation_derivative(
                    activation_buffer.data[i], data_buffer.data[i]);
            }
        }
    }

    __device__ void apply_nonlinearity(DataLoadingBuffer& data_buffer)
    {
        UNROLL
        for(index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD; ++i)
        {
            if (is_input_thread())
            {
                data_buffer.data[i] = parameters.config.apply_activation_function(
                    data_buffer.data[i]);
            }
        }
    }

    __device__ void compress_data_buffer(DataLoadingBuffer& data_buffer)
    {
        UNROLL
        for(index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD; ++i)
        {
            if(is_input_thread())
            {
                data_buffer.data[i] = data_buffer.data[i * Config::CACHE_LINE_USAGE];
            }
        }
    }

    __device__ void store_nonlinear_input_global(RegisterState& register_state,
        DataLoadingBuffer& data_buffer)
    {
        UNROLL
        for (index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_GLOBAL_STORE)
        {
            predicated_store_vector(register_state, data_buffer.data[i], i);
        }
    }

    __device__ void store_nonlinear_input_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer)
    {
        UNROLL
        for (index_t i = 0; i < Config::USEFUL_GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_INPUT_SHARED_STORE)
        {
            predicated_store_input_vector_shared(register_state,
                shared_state, data_buffer.data[i], i);
        }

        UNROLL
        for (index_t i = 0; i < Config::GLOBAL_VALUES_PER_THREAD;
            i += Config::VALUES_PER_OUTPUT_SHARED_STORE)
        {
            predicated_store_output_vector_shared(register_state,
                shared_state, data_buffer.data[i], i);
        }
    }

    __device__ bool check_for_critical_barrier_failure(RegisterState& register_state,
        SharedDataStorage& shared_state)
    {
        index_t shared_offset = register_state.shared_base + Config::SHARED_BARRIER_OFFSET;

        register_state.barrier_success = shared_state.data[shared_offset] == 0.0;

        return register_state.barrier_success;
    }

    __device__ __noinline__ void spin_on_barrier_failure(RegisterState& register_state,
        SharedDataStorage& shared_state,
        DataLoadingBuffer& data_buffer)
    {
        #if USE_BARRIER
        #if WAIT_FOREVER
        for(index_t i = 0; true; ++i)
        #else
        for(index_t i = 0; i < Config::BARRIER_WAIT_COUNT; ++i)
        #endif
        {
            external_load_input(register_state, shared_state, data_buffer);

            if(register_state.barrier_success)
            {
                t0printf("Thread (%d, %d, %d, %d) - Barrier succeeded on retry %d.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i);
                return;
            }
        }
        #endif
    }

    __device__ void synchronize_block() {
        __syncthreads();
    }

    __device__ void initialize_accumulators(ThreadTileAccumulators& accumulators) {

        UNROLL
        for (index_t row = 0; row < Config::THREAD_TILE_ROWS;
            row += Config::VALUES_PER_OUTPUT_SHARED_LOAD) {

            set_accumulators_to_zero(accumulators, row);
        }
    }

private:
    __device__ bool is_leader_thread() const
    {
        return threadIdx.y == 0;
    }

    __device__ index_t get_block_id_x() const
    {
        return blockIdx.x;
    }

    __device__ index_t get_block_id_y() const
    {
        return blockIdx.y;
    }

    __device__ bool is_input_thread() const
    {
        return getLinearThreadId() < Config::INPUT_LOAD_GROUP_SIZE;
    }

    __device__ void predicated_load_vector(RegisterState& register_state,
        RealType& value, index_t value_offset, bool load_output, bool is_retry)
    {
        index_t block_offset = is_input_thread() ?
            get_block_id_y() * Config::EXPANDED_BLOCK_TILE_COLUMNS :
            get_block_id_x() * Config::BLOCK_TILE_ROWS;

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
        for(int i = 0; i < Config::VALUES_PER_GLOBAL_LOAD; ++i) {
            loaded_data.data[i] = 0;
        }

        bool condition = (is_input_thread() && (io_offset < register_state.expanded_layer_size)) ||
            (!is_input_thread() && load_output && (io_offset < register_state.layer_size) &&
                (thread_offset + value_offset < Config::BLOCK_TILE_ROWS));

        register_state.should_check_barrier = condition && is_input_thread();

        predicated_atomic_global_load_relaxed(loaded_data, *reinterpret_cast<GlobalAccessType*>(
            load_base + offset), condition);

        for(int i = 0; i < Config::VALUES_PER_GLOBAL_LOAD; ++i)
        {
            if(condition)
            {
                dprintf("Thread (%d, %d, %d, %d) - Loading %s activation[%d] "
                    "(%d block, %d thread, %d io) (%p) = %f (%s)\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    is_input_thread() ? "input" : "output",
                    offset + i, block_offset, thread_offset, io_offset + i,
                    load_base + offset + i,
                    (float)loaded_data.data[i],
                    condition ? "enabled" : "disabled");
            }
        }

        reinterpret_cast<GlobalAccessType&>(value) = loaded_data;
    }

    __device__ void predicated_load_back_prop_activation_vector(RegisterState& register_state,
        RealType& value, index_t value_offset)
    {
        index_t  block_offset = get_block_id_y() * Config::BLOCK_TILE_COLUMNS;
        index_t thread_offset = get_thread_id_in_load_group() *
            Config::USEFUL_GLOBAL_VALUES_PER_THREAD;

        RealType* load_base = register_state.back_prop_activation_base_pointer;

        index_t offset = value_offset + thread_offset + block_offset;

        ActivationAccessType loaded_data;

        UNROLL
        for (int i = 0; i < Config::VALUES_PER_ACTIVATION_LOAD; ++i)
        {
            loaded_data.data[i] = 0;
        }

        bool condition = (offset < register_state.layer_size) && is_input_thread();

        predicated_atomic_global_load_relaxed(loaded_data,
            reinterpret_cast<const ActivationAccessType&>(*(load_base + offset)), condition);

        for(int i = 0; i < Config::VALUES_PER_ACTIVATION_LOAD; ++i)
        {
            if(condition)
            {
                dprintf("Thread (%d, %d, %d, %d) - Loading back prop activation[%d] "
                    "(%d block, %d thread, %d io) (%p) = %f (%s)\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset, block_offset, thread_offset, offset + i,
                    load_base + offset + i,
                    (float)loaded_data.data[i],
                    condition ? "enabled" : "disabled");
            }
        }

        reinterpret_cast<ActivationAccessType&>(value) = loaded_data;
    }

    __device__ void store_thread_accumulators_to_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        const RealType& accumulator, index_t value_offset)
    {
        index_t row_offset = value_offset * Config::THREADS_PER_COLUMN + threadIdx.x;

        index_t offset = row_offset + threadIdx.y * Config::BLOCK_TILE_ROWS;

        index_t shared_offset = register_state.shared_base +
            offset + Config::SHARED_REDUCE_OFFSET;

        if(row_offset < register_state.layer_size &&
            threadIdx.y * Config::VALUES_PER_SHARED_LOAD < register_state.layer_size)
        {
            dprintf("Thread (%d, %d, %d, %d) - Storing accumulator[%d] %f to shared[%d]\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                value_offset, (float)accumulator, shared_offset);
        }

        shared_state.data[shared_offset] = accumulator;
    }

    __device__ void reduce_accumulator(RegisterState& register_state,
        SharedDataStorage& shared_state,
        RealType& accumulator, index_t value_offset)
    {
        #if REDUCE_ACCUMULATORS
        index_t stride = Config::BLOCK_TILE_ROWS;

        index_t base_offset = getLinearThreadId() +
            Config::THREADS_PER_BLOCK * value_offset;

        index_t offset = register_state.shared_base + base_offset + Config::SHARED_REDUCE_OFFSET;

        RealType value = shared_state.data[offset];

        if(base_offset < Config::BLOCK_TILE_ROWS)
        {
            if(base_offset < register_state.layer_size)
            {
                dprintf("Thread (%d, %d, %d, %d) - "
                    "Updating output (reduce) accumulator[%d] %f = "
                    "current value %f + shared[%d] %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    base_offset, (float)(accumulator + value), (float)accumulator,
                    offset, (float)value);
            }
        }

        offset += stride;

        UNROLL
        for(index_t i = 1; i < Config::THREADS_PER_ROW; ++i, offset += stride)
        {
            value += shared_state.data[offset];

            if(base_offset < Config::BLOCK_TILE_ROWS)
            {
                if(base_offset < register_state.layer_size)
                {
                    dprintf("Thread (%d, %d, %d, %d) - "
                        "Updating output (reduce) accumulator[%d] %f = "
                        "current value %f + shared[%d] %f\n",
                        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                        base_offset, (float)(accumulator + value), (float)accumulator,
                        offset, (float)value);
                }
            }
        }

        accumulator += value;
        #endif
    }

    __device__ void initialize_output_accumulator(RegisterState& register_state,
    SharedDataStorage& shared_state,
    RealType& accumulator, index_t value_offset)
    {
        accumulator = 0.0;

        #if INITIALIZE_OUTPUT_ACCUMULATORS
        index_t base_offset = getLinearThreadId() +
            Config::THREADS_PER_BLOCK * value_offset;

        index_t output_offset = base_offset + register_state.shared_base +
            Config::SHARED_OUTPUT_OFFSET;

        if(base_offset < Config::BLOCK_TILE_ROWS && get_block_id_y() == 0)
        {
            accumulator = shared_state.data[output_offset];

            if(base_offset < register_state.layer_size)
            {
                dprintf("Thread (%d, %d, %d, %d) - Initializing output accumulator[%d] %f = "
                    "shared_output[%d] %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    base_offset, (float)accumulator, output_offset,
                    (float)shared_state.data[output_offset]);
            }
        }

        index_t global_output_base = get_block_id_x() * Config::BLOCK_TILE_ROWS;
        index_t global_input_base  = get_block_id_y() * Config::BLOCK_TILE_COLUMNS;

        index_t global_output_offset = base_offset + global_output_base;

        index_t shared_input_offset = global_output_offset - global_input_base;

        index_t input_offset = shared_input_offset + register_state.shared_base +
            Config::SHARED_INPUT_OFFSET;

        bool input_in_range = shared_input_offset < Config::BLOCK_TILE_COLUMNS &&
            global_output_offset >= global_input_base;

        if(input_in_range)
        {
            auto updated_accumulator = register_state.skip_connection_scale *
                shared_state.data[input_offset] + accumulator;

            if(shared_input_offset < register_state.layer_size)
            {
                dprintf("Thread (%d, %d, %d, %d) - Updating output accumulator[%d] %f = "
                    "accumultor %f + skip_connection_scale %f * shared_input[%d] %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    base_offset, (float)updated_accumulator, (float) accumulator,
                    register_state.skip_connection_scale, input_offset,
                    (float)shared_state.data[input_offset]);
            }

            accumulator = updated_accumulator;
        }
        #endif
    }

    __device__ void predicated_store_vector(RegisterState& register_state,
        RealType& data, index_t value_offset)
    {
        RealType* output_base = get_output_pointer(register_state);

        index_t block_offset  = get_block_id_y() * Config::BLOCK_TILE_COLUMNS;
        index_t thread_offset = getLinearThreadId() *
            Config::USEFUL_GLOBAL_VALUES_PER_THREAD;

        GlobalStoreType stored_data;

        for(int i = 0; i < Config::VALUES_PER_GLOBAL_STORE; ++i)
        {
            stored_data.data[i] = (&data)[i + value_offset];
        }

        index_t offset = thread_offset + value_offset + block_offset;

        bool condition = get_block_id_x() == 0 &&
            (offset < register_state.layer_size) && is_input_thread();

        predicated_atomic_global_store_relaxed(
            reinterpret_cast<GlobalStoreType&>(*(output_base + offset)),
            stored_data, condition);

        if(condition)
        {
            for(int i = 0; i < Config::VALUES_PER_GLOBAL_STORE; ++i)
            {
                dprintf("Thread (%d, %d, %d, %d) - Saving final activation[%d] "
                    "(%d block, %d thread, %d value) (%p) = %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset + i, block_offset, thread_offset, value_offset + i,
                    output_base + offset + i,
                    (float)reinterpret_cast<GlobalAccessType&>(data).data[i]);
            }
        }
    }

    __device__ void predicated_store_input_vector_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        const RealType& data, index_t value_offset)
    {
        index_t thread_offset = getLinearThreadId() * Config::USEFUL_GLOBAL_VALUES_PER_THREAD;

        SharedInputStoreType stored_data;

        for(int i = 0; i < Config::VALUES_PER_INPUT_SHARED_STORE; ++i)
        {
            stored_data.data[i] = (&data)[i + value_offset];
        }

        index_t shared_offset = thread_offset + value_offset;

        index_t offset = shared_offset + register_state.shared_base;

        if(is_input_thread())
        {
            reinterpret_cast<SharedInputStoreType&>(shared_state.data[offset]) = stored_data;
        }

        for(int i = 0; i < Config::VALUES_PER_INPUT_SHARED_STORE; ++i)
        {
            if(is_input_thread() && (shared_offset + i) < register_state.layer_size)
            {
                dprintf("Thread (%d, %d, %d, %d) - Storing loaded input value to shared [%d] "
                    "(block offset %d) = %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset + i, shared_offset + i,
                    (float)((&data)[i]));
            }
        }
    }

    __device__ void predicated_store_output_vector_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        const RealType& data, index_t value_offset)
    {
        index_t thread_offset = get_thread_id_in_load_group() * Config::GLOBAL_VALUES_PER_THREAD;

        SharedOutputStoreType stored_data;

        for(int i = 0; i < Config::VALUES_PER_OUTPUT_SHARED_STORE; ++i)
        {
            stored_data.data[i] = (&data)[i + value_offset];
        }

        index_t shared_offset = thread_offset + value_offset;

        index_t offset = shared_offset + register_state.shared_base + Config::SHARED_OUTPUT_OFFSET;

        bool condition = !is_input_thread() && shared_offset < Config::BLOCK_TILE_ROWS;

        if(condition)
        {
            reinterpret_cast<SharedOutputStoreType&>(shared_state.data[offset]) = stored_data;
        }

        for(int i = 0; i < Config::VALUES_PER_OUTPUT_SHARED_STORE; ++i)
        {
            if(condition &&
                ((get_block_id_x() * Config::BLOCK_TILE_ROWS + shared_offset + i)
                < register_state.layer_size))
            {
                dprintf("Thread (%d, %d, %d, %d) - Storing loaded output value to shared [%d] "
                    "(block offset %d) = %f\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    offset + i, shared_offset + i,
                    (float)((&data)[i]));
            }
        }
    }

    __device__ index_t get_thread_id_in_load_group() const
    {
        index_t myThreadId = getLinearThreadId();

        bool isInInputLoadGroup = myThreadId < Config::INPUT_LOAD_GROUP_SIZE;

        return isInInputLoadGroup ? myThreadId : myThreadId - Config::INPUT_LOAD_GROUP_SIZE;
    }

    __device__ index_t getLinearThreadId() const
    {
        return threadIdx.x + threadIdx.y * Config::THREADS_PER_COLUMN;
    }

    __device__ bool is_barrier_thread() const
    {
        return is_barrier_id(getLinearThreadId());
    }

    __device__ bool is_barrier_id(index_t threadId) const
    {
        index_t segmentOffset = threadId % Config::CACHE_LINE_USAGE;

        return 1 == segmentOffset;
    }

    __device__ index_t get_compressed_input_linear_thread_id() const
    {
        return compress_id(getLinearThreadId());
    }

    __device__ index_t compress_id(index_t threadId) const
    {
        return threadId / Config::CACHE_LINE_USAGE;
    }

    __device__ bool check_barrier(RegisterState& register_state,
        const RealType& value, index_t value_offset, bool is_retry) const
    {
        index_t offset = value_offset +
            Config::GLOBAL_VALUES_PER_THREAD * get_thread_id_in_load_group();

        bool result = is_barrier_id(offset) ?
            value >= register_state.reduction_threads_per_value : true;

        if (is_barrier_id(offset)) {
            dprintf("Thread (%d, %d, %d, %d) - Checking barrier counter %f against "
                "requirement %f (%s)\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                (float)value, (float)register_state.reduction_threads_per_value,
                result ? "success" : "failed");
        }

        return result;
    }

    __device__ void set_accumulators_to_zero(ThreadTileAccumulators& accumulators, index_t start)
    {
        for (index_t row = 0; row < Config::VALUES_PER_OUTPUT_SHARED_LOAD; ++row)
        {
            accumulators.data[start + row] = 0;
        }
    }

    __device__ void load_output_data_from_shared(RegisterState& register_state,
        SharedDataStorage& shared_state,
        ThreadTileAccumulators& accumulators, index_t row)
    {
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
        ThreadTileAccumulators& accumulators, index_t row)
    {
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
        ThreadTileInputs& thread_inputs)
    {
        index_t thread_offset = register_state.shared_base +
            Config::VALUES_PER_SHARED_LOAD * threadIdx.y;

        index_t thread_offset_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;

        UNROLL
        for (index_t column = 0; column < Config::THREAD_TILE_COLUMNS;
            column += Config::VALUES_PER_SHARED_LOAD, thread_offset += thread_offset_step)
        {
            auto value = reinterpret_cast<SharedAccessType&>(
                shared_state.data[thread_offset]);

            for(index_t i = 0; i < Config::VALUES_PER_SHARED_LOAD; ++i)
            {
                t0printf("Thread (%d, %d, %d, %d) - Loading tile inputs from shared memory "
                    "at %d = %f.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                    thread_offset + i, value.data[i]);
            }

            reinterpret_cast<SharedAccessType&>(thread_inputs.data[column]) = value;
        }
    }

    __device__ void perform_thread_tile_math(ThreadTileAccumulators& accumulators,
        ThreadTileWeights& weights, ThreadTileInputs& thread_inputs)
    {
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
        index_t row, index_t column)
    {
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
        index_t row, index_t column)
    {
        UNROLL
        for (index_t r = 0; r < Config::SIMD; ++r)
        {
            index_t row_base = threadIdx.x * Config::THREAD_TILE_ROWS +
                get_block_id_x() * Config::BLOCK_TILE_ROWS;
            index_t column_base = threadIdx.y * Config::VALUES_PER_SHARED_LOAD +
                get_block_id_y() * Config::BLOCK_TILE_COLUMNS;

            index_t thread_offset_step = Config::VALUES_PER_SHARED_LOAD * Config::THREADS_PER_ROW;
            index_t column_segment = column / Config::VALUES_PER_SHARED_LOAD;
            index_t column_offset  = column_segment * thread_offset_step;

            bool condition = (row_base + row + r < parameters.layer_size) &&
                (column_base + column + column_offset < parameters.layer_size);

            if (condition)
            {
                t0printf("Thread (%d, %d, %d, %d) - ffma accumulator[%d] %f = weight[%d, %d] %f * "
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
        ThreadTileOutputAccumulators& accumulators)
    {
        #if REDUCE_ADDRESS_MATH
        RealType* outputBuffer = register_state.activation_scratch;

        index_t bufferOffset =
            register_state.scratch_input_to_output_offset -
            2 * Config::EXPANDED_GRID_TILE_ROWS;

        index_t blockId = blockIdx.x;

        index_t blockOffset = blockId * Config::EXPANDED_BLOCK_TILE_ROWS;

        index_t threadId = 2 * getLinearThreadId();

        index_t threadOffset = threadId;
        #else
        RealType* outputBuffer = register_state.activation_scratch;
        index_t threadOffset = 0;
        index_t blockOffset = 0;
        index_t bufferOffset;
        #endif

        atomic_increment(outputBuffer, register_state, accumulators, blockOffset, threadOffset, bufferOffset);
    }

    __device__ void store_accumulators_back_prop(RegisterState& register_state,
        ThreadTileOutputAccumulators& accumulators)
    {
        #if REDUCE_ADDRESS_MATH
        RealType* outputBuffer = register_state.activation_scratch;

        index_t bufferOffset = 2 * Config::EXPANDED_GRID_TILE_ROWS -
            register_state.scratch_input_to_output_offset;

        index_t blockId = blockIdx.x;

        index_t blockOffset = blockId * Config::EXPANDED_BLOCK_TILE_ROWS;

        index_t threadId = 2 * getLinearThreadId();

        index_t threadOffset = threadId;
        #else
        RealType* outputBuffer = register_state.activation_scratch;
        index_t threadOffset = 0;
        index_t blockOffset = 0;
        index_t bufferOffset = 0;
        #endif

        atomic_increment(outputBuffer, register_state, accumulators, blockOffset, threadOffset, bufferOffset);
    }

    __device__ void atomic_increment(RealType* output_pointer,
        RegisterState& register_state,
        ThreadTileOutputAccumulators& accumulators, index_t blockOffset, index_t threadOffset, index_t bufferOffset) {

        #if ATOMIC_INCREMENT
        UNROLL
        for (index_t row = 0; row < Config::OUTPUTS_PER_THREAD; row += 1)
        {
            index_t offsetInLayer = threadOffset + row * 2 * Config::THREADS_PER_BLOCK;
            index_t offset = offsetInLayer + blockOffset + bufferOffset;

            bool condition = (offsetInLayer + blockOffset) < register_state.expanded_layer_size &&
                offsetInLayer < Config::EXPANDED_BLOCK_TILE_ROWS;

            #if DEBUG_RECURRENT_OPS
            if(condition)
            {
                auto result = atomic_increment_relaxed(output_pointer[offset],
                    accumulators.data[row]);

                dprintf("Thread (%d, %d, %d, %d) - atomic increment %d (%p) "
                    "%f = accumulator %f + original value %f.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row,
                    output_pointer + offset,
                    (float)(result + accumulators.data[row]),
                    (float)accumulators.data[row],
                    (float)result);
            }

            if(condition)
            {
                auto result = atomic_increment_relaxed(output_pointer[offset + 1],
                    RealType(1.0));

                dprintf("Thread (%d, %d, %d, %d) - atomic increment barrier %d (%p) "
                    "%f = accumulator %f + original value %f.\n",
                    blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row,
                    output_pointer + offset + 1,
                    (float)(result + 1.0f),
                    (float)1.0f,
                    (float)result);

            }
            #else
            predicated_atomic_increment_reduce_relaxed(output_pointer[offset],
                accumulators.data[row], condition);

            predicated_atomic_increment_reduce_relaxed(output_pointer[offset + 1],
                RealType(1.0), condition);
            #endif
        }
        #endif
    }

    __device__ RealType* get_output_pointer(RegisterState& register_state) {
        return register_state.input_base_pointer;
    }

private:
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



