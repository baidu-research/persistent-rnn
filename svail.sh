# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PRNN_INSTALL_PATH=$SCRIPTPATH/build_local
export CUDA_PATH=/tools/cuda_7.5.18

export CUDA_INC_PATH=$CUDA_PATH/include
export CUDA_LIB_PATH=$CUDA_PATH/lib64
export CUDA_BIN_PATH=$CUDA_PATH/bin

export PATH=$PRNN_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda/bin:/usr/local/slurm/bin:/usr/local/slurm/sbin
export LD_LIBRARY_PATH=/usr/local/ffmpeg_build/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:$CUDA_PATH/lib64:$PRNN_INSTALL_PATH/lib

export PRNN_KNOB_FILE=$SCRIPTPATH/config/persistent-rnn.config
export CUDA_ARCH=sm_50




