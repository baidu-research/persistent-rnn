# persistent-rnn

## Persistent RNNs

For a GPU, the largest source of on-chip memory is distributed among the individual register files
of thousands of threads. For example, the NVIDIA TitanX GPU has 6.3 MB of register file memory,
which is enough to store a recurrent layer with approximately 1200 activations. Persistent kernels
exploit this register file memory to cache recurrent weights and reuse them over multiple timesteps.

## Limitations in the current release

 * No support for GRU or LSTM layer types
  * These may be added in the future.  We would welcome pull requests that add support. 
 * The maximum layer size is determined by the selected GPU
  * 1088 for TitanX/M40
  * 1460 for Geforce 1080
  * 1880 for GP100 (2720 in fp16)
 * Only the following GPUs are supported (TitanX/M40, Geforce 1080 GTX, GP100)
  * Support for other GPUs can be enabled by determining good tile sizes and adding them to the file recurrent_ops.cu file.
 * The layer size must be a multiple of 4.
 * The mini-batch size must be 2 or greater.
 * The input data must be 16-byte aligned.


