# persistent-rnn


## Limitations in the current release

 * No support for GRU or LSTM layer types
 * The maximum layer size is determined by the selected GPU (1088 for NVIDIA TitanX)
 * The layer size must be a multiple of 4.
 * The mini-batch size must be 2 or greater.


