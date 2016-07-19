#pragma once

// Persistent RNN Includes
#include <prnn/detail/types/fixed_point.h>
#include <prnn/detail/types/float16.h>

#include <prnn/detail/rnn/recurrent_ops_handle.h>

// Standard Library Includes
#include <cstdint>

namespace prnn {
namespace rnn {

typedef int32_t index_t;
typedef prnn::types::float16 float16;
typedef prnn::RecurrentLayerDirection RecurrentLayerDirection;

__device__ constexpr index_t align(index_t address, index_t alignment)
{
    return address % alignment == 0 ? address : address + alignment - address % alignment;
}

__device__ constexpr index_t get_max(index_t l, index_t r)
{
    return l < r ? r : l;
}

__device__ constexpr index_t get_min(index_t left, index_t right)
{
    return left < right ? left : right;
}

template<
    index_t StreamingMultiprocessors = 24,
    index_t GridTileRows = 1088,
    index_t GridTileColumns = 1088,
    index_t BlockTileRows = 224,
    index_t BlockTileColumns = 224,
    index_t ThreadTileRows = 14,
    index_t ThreadTileColumns = 14,
    int Direction = prnn::RECURRENT_FORWARD,
    typename RealType = float>
class TileConfig {
public:
    enum {
        STREAMING_MULTIPROCESSORS = StreamingMultiprocessors
    };

    enum {
        GRID_TILE_ROWS    = GridTileRows,
        GRID_TILE_COLUMNS = GridTileColumns
    };

    enum {
        BLOCK_TILE_ROWS    = BlockTileRows,
        BLOCK_TILE_COLUMNS = BlockTileColumns
    };

    enum {
        THREAD_TILE_ROWS    = ThreadTileRows,
        THREAD_TILE_COLUMNS = ThreadTileColumns
    };

    enum {
        THREAD_TILE_SIZE = THREAD_TILE_ROWS * THREAD_TILE_COLUMNS,
        BLOCK_TILE_SIZE  = BLOCK_TILE_ROWS  * BLOCK_TILE_COLUMNS,
        GRID_TILE_SIZE   = GRID_TILE_ROWS   * GRID_TILE_COLUMNS
    };

    enum {
        THREADS_PER_BLOCK = (BLOCK_TILE_SIZE + THREAD_TILE_SIZE - 1) / THREAD_TILE_SIZE
    };

    enum {
        BLOCKS_PER_GRID = (GRID_TILE_SIZE + BLOCK_TILE_SIZE - 1) / BLOCK_TILE_SIZE
    };

    enum {
        THREADS_PER_ROW = BLOCK_TILE_COLUMNS / THREAD_TILE_COLUMNS
    };

    enum {
        THREADS_PER_COLUMN = BLOCK_TILE_ROWS / THREAD_TILE_ROWS
    };

    enum {
        BLOCKS_PER_SM = (BLOCKS_PER_GRID + STREAMING_MULTIPROCESSORS - 1) /
            STREAMING_MULTIPROCESSORS
    };

    enum {
        CACHE_LINE_SIZE = 32
    };

    enum {
        VALUES_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(RealType)
    };

    enum {
        USABLE_VALUES_PER_CACHE_LINE = VALUES_PER_CACHE_LINE / 2
    };

    enum {
        EXPANDED_GRID_TILE_ROWS = GRID_TILE_ROWS * 2,
        EXPANDED_GRID_TILE_COLUMNS = GRID_TILE_COLUMNS * 2,
        EXPANDED_BLOCK_TILE_ROWS = BLOCK_TILE_ROWS * 2,
        EXPANDED_BLOCK_TILE_COLUMNS = BLOCK_TILE_COLUMNS * 2
    };

    enum {
        MAXIMUM_LAYER_SIZE = get_min(GRID_TILE_ROWS, GRID_TILE_COLUMNS),
        EXPANDED_LAYER_SIZE = get_min(EXPANDED_GRID_TILE_ROWS, EXPANDED_GRID_TILE_COLUMNS)
    };

    enum {
        DIRECTION = Direction
    };

};

class None {};

__device__ constexpr index_t get_next_power_of_two(
        index_t value,
        index_t maxb = sizeof(index_t)*8,
        index_t curb = 1
        ) {
    return maxb <= curb
            ? value
            : get_next_power_of_two( ((value-1) | ((value-1)>>curb))+1, maxb, curb << 1 )
            ;
}

__device__ constexpr index_t get_log2(index_t n, index_t p = 0)
{
    return (n <= 1) ? p : get_log2(n / 2, p + 1);
}

__device__ constexpr index_t evenly_divisible(index_t n, index_t d)
{
    return n % d == 0 ? d : evenly_divisible(n, d/2);
}

template<int bytes>
class GetAlignedType
{
public:
    typedef uint8_t type[bytes];
};

template<>
class GetAlignedType<8>
{
public:
    typedef float2 type;
};

template<>
class GetAlignedType<16>
{
public:
    typedef float4 type;
};

template<int bytes>
class GetIntType
{
public:
    typedef int64_t type;
};

template<>
class GetIntType<2>
{
public:
    typedef int16_t type;
};

template<>
class GetIntType<4>
{
public:
    typedef int32_t type;
};

template<typename T>
class GetSIMD
{
public:
    enum {
        value = 1
    };
};

template<>
class GetSIMD<float16>
{
public:
    enum {
        value = 2
    };
};

class RecurrentOpsDeviceHandle
{
public:
    RecurrentOpsDeviceHandle(const RecurrentOpsHandle& handle) :

        layerSize(handle.layerSize),
        miniBatchSize(handle.miniBatchSize),
        timesteps(handle.timesteps),
        skipConnectionScale(handle.skipConnectionScale),
        direction(handle.direction)
    {}

public:
    size_t layerSize;
    size_t miniBatchSize;
    size_t timesteps;
    double skipConnectionScale;

public:
    RecurrentLayerDirection direction;
};

template<
    typename RealType_,
    typename ActivationFunction = None,
    typename Config = TileConfig<> >
class RecurrentConfig {
public:
    typedef RealType_ RealType;

public:
    enum {
        GRID_TILE_ROWS    = Config::GRID_TILE_ROWS,
        GRID_TILE_COLUMNS = Config::GRID_TILE_COLUMNS
    };

    enum {
        EXPANDED_GRID_TILE_ROWS     = Config::EXPANDED_GRID_TILE_ROWS,
        EXPANDED_GRID_TILE_COLUMNS  = Config::EXPANDED_GRID_TILE_COLUMNS,
        EXPANDED_BLOCK_TILE_ROWS    = Config::EXPANDED_BLOCK_TILE_ROWS,
        EXPANDED_BLOCK_TILE_COLUMNS = Config::EXPANDED_BLOCK_TILE_COLUMNS
    };

    enum {
        BLOCK_TILE_ROWS    = Config::BLOCK_TILE_ROWS,
        BLOCK_TILE_COLUMNS = Config::BLOCK_TILE_COLUMNS
    };

    enum {
        THREAD_TILE_ROWS    = Config::THREAD_TILE_ROWS,
        THREAD_TILE_COLUMNS = Config::THREAD_TILE_COLUMNS
    };

    enum {
        THREAD_TILE_SIZE = Config::THREAD_TILE_SIZE,
        BLOCK_TILE_SIZE  = Config::BLOCK_TILE_SIZE
    };

    enum {
        BLOCKS_PER_GRID = Config::BLOCKS_PER_GRID,
        BLOCKS_PER_SM   = Config::BLOCKS_PER_SM
    };

    enum {
        THREADS_PER_ROW = Config::THREADS_PER_ROW
    };

    enum {
        THREADS_PER_COLUMN = Config::THREADS_PER_COLUMN
    };

    enum {
        CACHE_LINE_SIZE = Config::CACHE_LINE_SIZE,
        VALUES_PER_CACHE_LINE = Config::VALUES_PER_CACHE_LINE,
        USABLE_VALUES_PER_CACHE_LINE = Config::USABLE_VALUES_PER_CACHE_LINE,
        CACHE_LINE_USAGE = VALUES_PER_CACHE_LINE / USABLE_VALUES_PER_CACHE_LINE
    };

    enum {
        THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK,
        COMPRESSED_THREADS_PER_BLOCK = THREADS_PER_BLOCK / CACHE_LINE_USAGE
    };

    enum {
        VALUES_PER_SHARED_LOAD = ((16 + sizeof(RealType) - 1) / sizeof(RealType))
    };

    enum {
        THREAD_TILE_COLUMN_SEGMENTS = THREAD_TILE_COLUMNS / VALUES_PER_SHARED_LOAD,
        THREAD_TILE_COLUMN_SEGMENT_REMAINDER = THREAD_TILE_COLUMNS % VALUES_PER_SHARED_LOAD
    };

    enum {
        BARRIER_STATUS_SIZE = 1
    };

    enum {
        UNALIGNED_GLOBAL_VALUES_PER_THREAD = (BLOCK_TILE_ROWS + EXPANDED_BLOCK_TILE_COLUMNS +
            THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        GLOBAL_VALUES_PER_THREAD = align(UNALIGNED_GLOBAL_VALUES_PER_THREAD, VALUES_PER_SHARED_LOAD),
        USEFUL_GLOBAL_VALUES_PER_THREAD = GLOBAL_VALUES_PER_THREAD / CACHE_LINE_USAGE
    };

    static_assert(THREAD_TILE_COLUMN_SEGMENT_REMAINDER == 0,
        "No support for thread tiles that are not evenly divisible by the shared load size yet.");

    static_assert(GRID_TILE_ROWS % GLOBAL_VALUES_PER_THREAD == 0,
        "Grid size must be divisible by the minimum load size");

    static_assert(GLOBAL_VALUES_PER_THREAD % USEFUL_GLOBAL_VALUES_PER_THREAD == 0,
        "Global values per thread must be evenly divisible by useful values");

    enum {
        SHARED_INPUT_BUFFER_SIZE = EXPANDED_BLOCK_TILE_COLUMNS,
        SHARED_OUTPUT_BUFFER_SIZE = EXPANDED_BLOCK_TILE_ROWS
    };

    enum {
        SHARED_REDUCE_OFFSET = SHARED_INPUT_BUFFER_SIZE + SHARED_OUTPUT_BUFFER_SIZE,
        SHARED_OUTPUT_OFFSET = SHARED_INPUT_BUFFER_SIZE,
        SHARED_INPUT_OFFSET = 0,
        SHARED_BARRIER_OFFSET = SHARED_INPUT_BUFFER_SIZE + SHARED_OUTPUT_BUFFER_SIZE +
            THREADS_PER_ROW * BLOCK_TILE_ROWS
    };

    enum {
        SHARED_BUFFER_SIZE = get_next_power_of_two(SHARED_BARRIER_OFFSET + 1)
    };

    enum {
        VALUES_PER_INPUT_SHARED_STORE = get_min((16 + sizeof(RealType) - 1) / sizeof(RealType),
            USEFUL_GLOBAL_VALUES_PER_THREAD),
        VALUES_PER_OUTPUT_SHARED_STORE = get_min((16 + sizeof(RealType) - 1) / sizeof(RealType),
            GLOBAL_VALUES_PER_THREAD)
    };

    enum {
        VALUES_PER_GLOBAL_LOAD = get_min(GLOBAL_VALUES_PER_THREAD,
            (16 + sizeof(RealType) - 1) / sizeof(RealType)),
        VALUES_PER_GLOBAL_STORE = get_min((16 + sizeof(RealType) - 1) / sizeof(RealType),
            USEFUL_GLOBAL_VALUES_PER_THREAD),
        VALUES_PER_ACTIVATION_LOAD = get_min(USEFUL_GLOBAL_VALUES_PER_THREAD,
            (16 + sizeof(RealType) - 1) / sizeof(RealType))
    };

    enum {
        VALUES_PER_WEIGHT_LOAD = evenly_divisible(THREAD_TILE_ROWS,
            (16 + sizeof(RealType) - 1) / sizeof(RealType))
    };

    enum {
        VALUES_PER_OUTPUT_SHARED_LOAD = evenly_divisible(THREAD_TILE_ROWS,
            (16 + sizeof(RealType) - 1) / sizeof(RealType))
    };

    enum {
        INPUT_LOAD_GROUP_SIZE  = EXPANDED_BLOCK_TILE_COLUMNS / GLOBAL_VALUES_PER_THREAD,
        OUTPUT_LOAD_GROUP_SIZE = BLOCK_TILE_ROWS / GLOBAL_VALUES_PER_THREAD
    };

    enum {
        OUTPUTS_PER_THREAD = (BLOCK_TILE_ROWS + THREADS_PER_BLOCK - 1) /
            THREADS_PER_BLOCK
    };

    static_assert(INPUT_LOAD_GROUP_SIZE + OUTPUT_LOAD_GROUP_SIZE <= THREADS_PER_BLOCK,
        "Incorrect load group sizes.");

    enum {
        WARP_SIZE = 32
    };

    enum {
        SIMD = GetSIMD<RealType>::value
    };

    enum {
        SHARED_REDUCE_STORE_VALUES_PER_THREAD = evenly_divisible(THREAD_TILE_ROWS,
            (16 + sizeof(RealType) - 1) / sizeof(RealType))
    };

    enum {
        FIXED_POINT_BITS = sizeof(RealType) * 8
    };

    enum {
        THREADS_PER_GLOBAL_REDUCTION = ((GRID_TILE_COLUMNS + BLOCK_TILE_COLUMNS - 1) /
            BLOCK_TILE_COLUMNS)
    };

    enum {
        FIXED_POINT_COUNTER_BITS = get_log2(THREADS_PER_GLOBAL_REDUCTION) + 1
    };

    enum {
        FIXED_POINT_INTEGER_BITS = get_min((FIXED_POINT_BITS - FIXED_POINT_COUNTER_BITS) / 2, 7)
    };

    enum {
        FIXED_POINT_FRACTIONAL_BITS = FIXED_POINT_BITS - FIXED_POINT_COUNTER_BITS -
            FIXED_POINT_INTEGER_BITS
    };

    enum {
        DIRECTION = Config::DIRECTION
    };

    enum {
        BARRIER_WAIT_COUNT = 3333 // about 10us
    };

public:
    typedef typename GetIntType<sizeof(RealType)>::type IntType;
    typedef prnn::types::fixed_point<IntType, FIXED_POINT_FRACTIONAL_BITS> FixedPointType;

public:
    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    ThreadTileWeights {
    public:
        RealType data[THREAD_TILE_ROWS][THREAD_TILE_COLUMNS];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    ThreadTileAccumulators {
    public:
        RealType data[THREAD_TILE_ROWS];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    ThreadTileOutputAccumulators {
    public:
        RealType data[OUTPUTS_PER_THREAD];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    ThreadTileInputs {
    public:
        RealType data[THREAD_TILE_COLUMNS];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    DataLoadingBuffer {
    public:
        RealType data[GLOBAL_VALUES_PER_THREAD];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    ActivationLoadingBuffer {
    public:
        RealType data[USEFUL_GLOBAL_VALUES_PER_THREAD];
    };

    class
    #if defined(__CUDACC__)
    __align__(16)
    #endif
    SharedDataStorage
    {
    public:
        RealType data[2 * SHARED_BUFFER_SIZE];
    };

    union ActivationAccessType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_ACTIVATION_LOAD>::type aligned_data;
        RealType data[VALUES_PER_ACTIVATION_LOAD];
    };

    union GlobalAccessType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_GLOBAL_LOAD>::type aligned_data;
        RealType data[VALUES_PER_GLOBAL_LOAD];
    };

    union SharedAccessType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_SHARED_LOAD>::type aligned_data;
        RealType data[VALUES_PER_SHARED_LOAD];
    };

    union OutputSharedAccessType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_SHARED_LOAD>::type aligned_data;
        RealType data[VALUES_PER_OUTPUT_SHARED_LOAD];
    };

    union GlobalStoreType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_GLOBAL_STORE>::type aligned_data;
        RealType data[VALUES_PER_GLOBAL_STORE];
    };

    union SharedInputStoreType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_INPUT_SHARED_STORE>::type aligned_data;
        RealType data[VALUES_PER_INPUT_SHARED_STORE];
    };

    union SharedOutputStoreType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_OUTPUT_SHARED_STORE>::type aligned_data;
        RealType data[VALUES_PER_OUTPUT_SHARED_STORE];
    };

    union SharedAccumulatorStoreType {
        typename GetAlignedType<sizeof(RealType)*SHARED_REDUCE_STORE_VALUES_PER_THREAD>::type
            aligned_data;
        RealType data[SHARED_REDUCE_STORE_VALUES_PER_THREAD];
    };

    union WeightAccessType {
        typename GetAlignedType<sizeof(RealType)*VALUES_PER_WEIGHT_LOAD>::type aligned_data;
        RealType data[VALUES_PER_WEIGHT_LOAD];
    };

public:
    RecurrentConfig(ActivationFunction f, RecurrentOpsDeviceHandle handle)
        : activationFunction(f), handle(handle) {}

    RecurrentConfig(RecurrentOpsDeviceHandle handle) : handle(handle) {}

public:
    __device__ RealType apply_activation_function(RealType v) const {
        return activationFunction(v);
    }

    __device__ RealType apply_activation_derivative(RealType a, RealType d) const {
        return activationFunction(a, d);
    }

    __device__ index_t get_timestep(index_t logical_timestep, index_t timesteps) const {
        if (DIRECTION == RECURRENT_FORWARD) {
            return logical_timestep;
        }
        else {
            return timesteps - logical_timestep - 1;
        }
    }

    __device__ bool is_last_timestep(index_t timestep, index_t timesteps) const {
        if (DIRECTION == RECURRENT_FORWARD) {
            return timestep == (timesteps - 1);
        }
        else {
            return timestep == 0;
        }
    }

    __device__ constexpr bool is_reversed() const {
        return DIRECTION == RECURRENT_REVERSE;
    }

public:
    __device__ constexpr bool is_always_fully_covered() const {
        return false;
    }

public:
    ActivationFunction activationFunction;
    RecurrentOpsDeviceHandle handle;

};


template <typename T, typename TileConfiguration = TileConfig<>>
class RecurrentArchitectureParameters {
public:
    typedef TileConfiguration TileParameters;
    typedef T RealType;

public:
    RecurrentArchitectureParameters(RecurrentOpsHandle handle)
        : handle(handle) {}

public:
    dim3 blocks() const {
        int block_rows    = (handle.layerSize + TileParameters::BLOCK_TILE_ROWS    - 1) /
            TileParameters::BLOCK_TILE_ROWS;
        int block_columns = (handle.layerSize + TileParameters::BLOCK_TILE_COLUMNS - 1) /
            TileParameters::BLOCK_TILE_COLUMNS;

        return dim3(block_rows, block_columns, 1);
    }

    index_t block_count() const {
        return blocks().x * blocks().y;
    }

    dim3 threads() const {
        return dim3(TileParameters::THREADS_PER_COLUMN, TileParameters::THREADS_PER_ROW, 1);
    }

    index_t thread_count() const {
        return threads().x * threads().y;
    }

    index_t activations_per_block() const {
        return TileParameters::BLOCK_TILE_ROWS;
    }

    index_t activations_per_grid() const {
        return TileParameters::GRID_TILE_ROWS;
    }

    index_t scratch_activations_per_grid() const {
        return TileParameters::EXPANDED_GRID_TILE_ROWS;
    }

public:
    bool is_supported() const {
        return handle.layerSize <= TileParameters::GRID_TILE_COLUMNS;
    }

public:
    RecurrentOpsHandle handle;

};

}
}


