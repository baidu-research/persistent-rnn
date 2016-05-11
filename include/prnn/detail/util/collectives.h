
#pragma once

#include <prnn/detail/assert.h>

namespace detail {

class ThreadGroup {
public:
    enum Type {
        ALIGNED = 0x1,
        WARP    = 0x2,
        CTA     = 0x3,
        GRID    = 0x4,
        GRID2D  = 0x5
    };

public:
    __device__ ThreadGroup(Type t) { set_type(t); }
    __device__ ThreadGroup(Type t, size_t s) { set_type(t); set_size(s); }

public:
    __device__ void set_type(Type t) { state_.type = t; }
    __device__ Type get_type() const { return static_cast<Type>(state_.type); }

public:
    __device__ void set_size(size_t s) { state_.size = s; }
    __device__ size_t get_size() const { return state_.size; }

public:
    __device__ bool is_warp() const { return get_type() == WARP; }
    __device__ bool is_cta()  const { return get_type() == CTA;  }
    __device__ bool is_grid() const { return get_type() == GRID; }
    __device__ bool is_2d_grid() const { return get_type() == GRID2D; }

    __device__ bool is_aligned() const { return get_type() == ALIGNED; }

private:
    class State {
    public:
        uint32_t type     : 3;
        uint32_t size     : 12;
        uint32_t reserved : 17;
    };

private:
    State state_;

};

}

__device__ inline detail::ThreadGroup this_thread_block() {
    return detail::ThreadGroup(detail::ThreadGroup::CTA);
}

__device__ inline detail::ThreadGroup this_grid() {
    return detail::ThreadGroup(detail::ThreadGroup::GRID);
}

__device__ inline detail::ThreadGroup this_2d_grid() {
    return detail::ThreadGroup(detail::ThreadGroup::GRID2D);
}

__device__ inline detail::ThreadGroup aligned_thread_group(size_t size) {
    return detail::ThreadGroup(detail::ThreadGroup::ALIGNED, size);
}

__device__ inline void sync(detail::ThreadGroup g) {
    if (g.is_cta()) {
        __syncthreads();
    }
    else if (g.is_warp()) {
        // noop
    }
    else if (g.is_aligned()) {
        // noop
    }
    else {
        // TODO: implement global sync
        MAJEL_ASSERT(false && "not implemented.");
    }
}

__device__ inline size_t size(detail::ThreadGroup group) {
    if (group.is_cta()) {
        return blockDim.x;
    }
    else if (group.is_warp()) {
        return warpSize;
    }
    else if (group.is_aligned()) {
        return group.get_size();
    }
    else if (group.is_2d_grid()) {
        return gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
    else {
        return gridDim.x * blockDim.x;
    }
}

__device__ inline size_t id(detail::ThreadGroup group) {
    if (group.is_cta()) {
        return threadIdx.x;
    }
    else if (group.is_warp()) {
        return threadIdx.x % 32;
    }
    else if (group.is_aligned()) {
        return threadIdx.x % size(group);
    }
    else if (group.is_2d_grid()) {
        return threadIdx.x +
            threadIdx.y * (blockDim.x) +
            blockIdx.x * (blockDim.x * blockDim.y) +
            blockIdx.y * (gridDim.x * blockDim.x * blockDim.y);
    }
    else {
        return threadIdx.x + blockDim.x * blockIdx.x;
    }
}

__device__ inline bool is_leader(detail::ThreadGroup group) {
    return id(group) == 0;
}

template<typename T>
__device__ inline T gather_down(T value, size_t index, detail::ThreadGroup g) {
    if (g.is_cta()) {
        // TODO: be smarter about this
        __shared__ T storage[1024];

        storage[id(g)] = value;

        sync(g);

        T result = T();

        if (id(g) + index < size(g)) {
            result = storage[id(g) + index];
        }

        sync(g);

        return result;

    }
    else if (g.is_warp()) {
        return __shfl_down(value, index);
    }
    else if (g.is_aligned()) {
        return __shfl_down(value, index, size(g));
    }
    else {
        // TODO: implement global gather
        MAJEL_ASSERT(false && "not implemented");
    }

    return value;
}

template<typename T, typename F>
__device__ inline T reduce(T value, F function, detail::ThreadGroup group) {

    for (uint32_t offset = size(group) / 2; offset > 0; offset /= 2) {
        T neighbors_value = gather_down(value, offset, group);

        value = function(value, neighbors_value);
    }

    return value;
}


