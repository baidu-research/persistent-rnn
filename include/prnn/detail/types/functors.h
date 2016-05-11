
#pragma once

// Persistent RNN Includes
#include <prnn/detail/types/float16.h>

namespace prnn {
namespace detail {
namespace types {

struct relu {
    template<typename T>
    CUDA_DECORATOR T operator()(const T& x) const {
        return maximum()(T(0.0), minimum()(T(20.0), x));
    }

    CUDA_DECORATOR float16 operator()(const float16& x) const {
        float tmp = x.to_float();
        return float16(relu()(tmp));
    }
};

struct tanh {
    template<typename T>
    CUDA_DECORATOR T operator()(const T& x) const {
        return std::tanh(x);
    }

    CUDA_DECORATOR float operator()(const int& x) const {
        return std::tanh(static_cast<float>(x));
    }

    CUDA_DECORATOR float16 operator()(const float16& x) const {
        return float16(std::tanh(x.to_float()));
    }
};

struct mult_drelu {
    template<typename T, typename U>
    CUDA_DECORATOR auto operator()(const T& x, const U& y) const -> decltype(x*y) {
        return (((x > 0.0) && (x < 20.0)) * y);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& x, const T& y) const {
        return (((x > T(0.0)) && (x < T(20.0))) * y);
    }

    CUDA_DECORATOR float16 operator()(const float16& x, const float16 &y) const {
        return float16(mult_drelu()(x.to_float(), y.to_float()));
    }

     CUDA_DECORATOR float operator()(const float16& x, const float& y) const {
        return mult_drelu()(x.to_float(), y);
    }

     CUDA_DECORATOR float operator()(const float& x, const float16& y) const {
        return mult_drelu()(x, y.to_float());
    }

};

struct mult_dtanh {
    template<typename T, typename U>
    CUDA_DECORATOR auto operator()(const T& x, const U& y) const -> decltype(x*y) {
        return ((1 - x * x) * y);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& x, const T& y) const {
        return T((T(1) - x * x) * y);
    }

    CUDA_DECORATOR float16 operator()(const float16& x, const float16& y) const {
        return float16(mult_dtanh()(x.to_float(), y.to_float()));
    }

    CUDA_DECORATOR float operator()(const float16& x, const float& y) const {
        return mult_dtanh()(x.to_float(), y);
    }

    CUDA_DECORATOR float operator()(const float& x, const float16& y) const {
        return mult_dtanh()(x, y.to_float());
    }

};

typedef std::tuple<relu, tanh> AllRecurrentForwardOps;
typedef std::tuple<mult_drelu, mult_dtanh> AllRecurrentBackwardOps;

}
}
}

