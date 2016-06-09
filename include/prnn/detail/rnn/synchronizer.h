#pragma once

// Persistent RNN Includes
#include <prnn/detail/util/abort.h>
#include <prnn/detail/util/atomics.h>

#include <prnn/detail/parallel/cuda_runtime_library.h>

namespace prnn {
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

class Synchronizer {
public:
    Synchronizer(index_t blocks, cudaStream_t stream, index_t* counters)
    : counters_(counters), blocks_(blocks), stream_(stream),
      current_phase_(0), not_finished_(true) {

        participating_count_   = counters + 0;
        barrier_failed_flag_   = counters + 1;
        current_phase_counter_ = counters + 2;
    }

public:
    index_t* get_counters() const { return counters_; }

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

        prnn::parallel::CudaRuntimeLibrary::cudaMemcpyAsync(get_barrier_failed_flag(), &failed,
            sizeof(index_t), prnn::parallel::CudaRuntimeLibrary::cudaMemcpyDefault, stream_);
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

        prnn::parallel::CudaRuntimeLibrary::cudaMemcpyAsync(&failed,
            get_barrier_failed_flag(), sizeof(index_t),
            prnn::parallel::CudaRuntimeLibrary::cudaMemcpyDefault, stream_);

        return failed != 0;
    }

    index_t get_current_phase_() const {
        index_t current_phase = 0;

        prnn::parallel::CudaRuntimeLibrary::cudaMemcpyAsync(&current_phase,
            get_current_phase_counter(), sizeof(index_t),
            prnn::parallel::CudaRuntimeLibrary::cudaMemcpyDefault, stream_);

        return current_phase;
    }

private:
    index_t* counters_;

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
        return current_phase_[block_id()];
    }

    __device__ void set_phase(index_t phase) {
        current_phase_[block_id()] = phase;
    }

public:
    __device__ void set_concurrent_execution_failed() {
        *barrier_failed_flag_ = 1;
    }

public:
    __device__ index_t block_id() const
    {
        return blockIdx.x + blockDim.x * blockIdx.y;
    }

private:
    static constexpr __device__ index_t timeout_() {
        return 32000; // about 32us
    }

private:
    index_t* counters_;

private:
    index_t* participating_count_;
    index_t* barrier_failed_flag_;
    index_t* current_phase_;

private:
    index_t blocks_;

};

}
}


