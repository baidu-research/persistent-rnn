# Get the current directory
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PRNN_INSTALL_PATH=$SCRIPTPATH/build_local

export PATH=$PRNN_INSTALL_PATH/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/opt/local/lib:$PRNN_INSTALL_PATH/lib:/usr/local/lib
export DYLD_LIBRARY_PATH=$PRNN_INSTALL_PATH/lib:/usr/local/cuda/lib

export PRNN_KNOB_FILE=$SCRIPTPATH/config/persistent-rnn.config

export CXX=clang++
export CC=clang
export CPP=clang


