#pragma once

// Majel Includes
#include <majel/recurrent/detail/recurrent_op_types.h>

// Standard Library Includes
#include <map>
#include <memory>
#include <string>

namespace prnn {

enum RecurrentLayerDirection {
    RECURRENT_FORWARD,
    RECURRENT_REVERSE
};

class RecurrentOpsConfig {
public:
    RecurrentOpsConfig(size_t layer_size, size_t mini_batch_, bool allow_persistent_kernels_,
        double skip_connection_scale = 0.0) :

        layer_size(layer_size),
        mini_batch_size(mini_batch_),
        allow_persistent_kernels(allow_persistent_kernels_),
        skip_connection_scale(skip_connection_scale)
    {}

public:
    size_t layer_size;
    size_t mini_batch_size;
    bool   allow_persistent_kernels;
    double skip_connection_scale;

};

/*! \brief A handle to store state associated with recurrent ops */
class RecurrentOpsHandle {
public:
    explicit RecurrentOpsHandle(RecurrentOpsConfig config_);

public:
    DDim get_weight_dimensions() const;

public:
    bool are_persistent_kernels_allowed(int device) const;
    bool are_sass_kernels_allowed(int device) const;

public:
    RecurrentOpsConfig config;
};

}


