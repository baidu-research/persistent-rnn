![Baidu Logo](/doc/baidu-research-logo-small.png)

# PRNN

A fast implementation of recurrent neural network layers in CUDA.

## Introduction

For a GPU, the largest source of on-chip memory is distributed among the individual register files
of thousands of threads. For example, the NVIDIA TitanX GPU has 6.3 MB of register file memory,
which is enough to store a recurrent layer with approximately 1200 activations. Persistent kernels
exploit this register file memory to cache recurrent weights and reuse them over multiple timesteps.

Avoiding reloading layer weights multiple times makes persistent kernels very efficient at low batch sizes.

## Performance

![TitanX Performance](/doc/mb-scaling.png)

Performance of the CUDA implementation is approximately 15x faster than an RNN implementation using CUBLAS for GEMM operations at a mini-batch size of 4.  There is still some room to improve performance compared with our previous implementation, which was written directly in assembly.  A CUDA implementation has the advantage that it is much easier to support more types of GPUs.

## Limitations in the current release

 * No support for GRU or LSTM layer types
  * These may be added in the future.  We would welcome pull requests that add support. 
 * The maximum layer size is determined by the selected GPU
  * 1152 for TitanX/M40
  * 1792 for GP100
    * 2432 in 16-bit floating point
 * Only the following GPUs are supported (TitanX/M40, Geforce 1080 GTX, GP100)
  * Support for new GPUs can be enabled by determining good tile sizes and adding them to recurrent_ops.cu .
 * The layer size must be a multiple of 4.
 * The mini-batch size must be 3 or greater.
 * The input data must be 16-byte aligned.
 
## Interface

The C language interface is in [`include/persistent_rnn.h`](include/persistent_rnn.h).
It supports GPU execution, and you can specify the CUDA stream if running on the GPU. We
took care to ensure that the library does not perform memory allocation internally, in
order to avoid synchronizations and overheads caused by memory allocation.

The interface is modeled after the recurrent layer interface in cuDNN v5 to make it easy to
integrate with frameworks that already interface with cuDNN.  The cuDNN interface is more
general than the capabilities provided by this library, and API calls that attempt to use unsupported
features will return PRNN_NOT_SUPPORTED, see the limitations section for details.

## Implementation

See [`include/prnn/detail/rnn/recurrent_ops_kernels.h`](include/prnn/detail/rnn/recurrent_ops_kernels.h) for
the kernel implementations.

## Compilation

prnn has been tested on Ubuntu 14.04 and OSX 10.10.  Windows is not supported
at this time.

First get the code:

```
git clone https://github.com/baidu-research/persistent-rnn.git
cd persistent-rnn
```

create a build directory:

```
mkdir build_local
```

run scons and build:

```
scons mode=release install=true
```

The C library should now be built along with test executables.  

## Contributing

We welcome improvements from the community, please feel free to submit pull
requests.
