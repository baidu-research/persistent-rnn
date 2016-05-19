#pragma once

#include <prnn/detail/types/float16.h>

namespace prnn {
namespace types {

template <class BaseType, size_t FracDigits>
class fixed_point
{
    const static BaseType factor = ((BaseType)1) << FracDigits;

    BaseType data;

public:
    CUDA_DECORATOR fixed_point(BaseType d) {
        data = d;
    }

    CUDA_DECORATOR fixed_point(double d) {
        *this = d;
    }

    CUDA_DECORATOR fixed_point(float f) {
        *this = f;
    }

    CUDA_DECORATOR fixed_point(float16 f) {
        *this = f.to_float();
    }

    CUDA_DECORATOR fixed_point& operator=(double d) {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    CUDA_DECORATOR fixed_point& operator=(float d) {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    CUDA_DECORATOR const BaseType& raw_data() const {
        return data;
    }

    CUDA_DECORATOR BaseType& raw_data() {
        return data;
    }

    template <typename T>
    CUDA_DECORATOR T extract() const {
        return ((T)data) / factor;
    }
};

}
}


