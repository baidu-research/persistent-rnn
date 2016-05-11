#pragma once

// Persistent RNN Includes
#include <prnn/detail/util/abort.h>
#include <prnn/detail/util/atomics.h>
#include <prnn/detail/util/collectives.h>

namespace prnn {
namespace detail {
namespace rnn {

class GpuTimer {

public:
    __device__ void start() {
        timestamp_ = clock();
    }

    __device__ index_t clocks() {
        return clock() - timestamp_;
    }

private:
    index_t timestamp_;

};

template<typename T>
__global__ void zero_counters(T* counter, index_t size) {
    for (index_t i = threadIdx.x; i < size; i += blockDim.x) {
        counter[i] = 0;
    }
}

inline void zero_counters(index_t* counters, index_t size, cudaStream_t stream) {
    zero_counters<index_t><<<1, 128, 0, stream>>>(counters, size);
}

class Synchronizer {
public:
    Synchronizer(index_t blocks, cudaStream_t stream, Place place)
    : counters_(make_dim(3), place),
      blocks_(blocks), stream_(stream), current_phase_(0), not_finished_(true) {

        zero_counters(counters_.raw_ptr(), 3, stream);

        participating_count_   = counters_.raw_ptr() + 0;
        barrier_failed_flag_   = counters_.raw_ptr() + 1;
        current_phase_counter_ = counters_.raw_ptr() + 2;
    }

public:
    ArrayView<index_t, 1> get_counters() const { return counters_; }

public:
    index_t get_blocks() const {
        return blocks_;
    }

    void check_for_failure() {

        if (!get_failed_()) {
          not_finished_ = false;
          return;
        }

        current_phase_ = get_current_phase_();
    }

    void reset_failed_flag() {
        index_t failed = 0;

        majel::gpu::detail::memcpy(get_barrier_failed_flag(), &failed,
            sizeof(index_t), cudaMemcpyHostToDevice, stream_);
    }

    bool not_finished() const {
        return not_finished_;
    }

    index_t get_current_phase() const {
        return current_phase_;
    }

public:
    index_t* get_participating_count() const {
        return participating_count_;
    }

    index_t* get_barrier_failed_flag() const {
        return barrier_failed_flag_;
    }

    index_t* get_current_phase_counter() const {
        return current_phase_counter_;
    }

private:
    bool get_failed_() const {
        index_t failed = 0;

        majel::gpu::detail::memcpy(&failed, get_barrier_failed_flag(),
            sizeof(index_t), cudaMemcpyDeviceToHost, stream_);

        return failed != 0;
    }

    index_t get_current_phase_() const {
        index_t current_phase = 0;

        majel::gpu::detail::memcpy(&current_phase, get_current_phase_counter(),
            sizeof(index_t), cudaMemcpyDeviceToHost, stream_);

        return current_phase;
    }

private:
    Array<index_t, 1> counters_;

private:
    index_t* participating_count_;
    index_t* barrier_failed_flag_;
    index_t* current_phase_counter_;

private:
    index_t blocks_;

private:
    cudaStream_t stream_;

private:
    index_t current_phase_;
    bool not_finished_;
};

class GpuSynchronizer {
public:
    __host__ GpuSynchronizer(Synchronizer s)
    : counters_(s.get_counters()),
      participating_count_(s.get_participating_count()),
      barrier_failed_flag_(s.get_barrier_failed_flag()),
      current_phase_(s.get_current_phase_counter()),
      blocks_(s.get_blocks()) {

    }

public:
    __device__ index_t get_phase() {
        return *current_phase_;
    }

    __device__ void set_phase(index_t phase) {
        *current_phase_ = phase;
    }

public:
    __device__ void set_concurrent_execution_failed() {
        *barrier_failed_flag_ = 1;
    }

private:
    static constexpr __device__ index_t timeout_() {
        return 32000; // about 32us
    }

private:
    ArrayView<index_t, 1> counters_;

private:
    index_t* participating_count_;
    index_t* barrier_failed_flag_;
    index_t* current_phase_;

private:
    index_t blocks_;

};

}
}
}


