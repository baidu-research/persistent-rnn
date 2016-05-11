#pragma once

namespace prnn {
namespace detail {
namespace types {

template <class BaseType, size_t FracDigits>
class fixed_point
{
    const static BaseType factor = ((BaseType)1) << FracDigits;

    BaseType data;

public:
    __host__ __device__ fixed_point(BaseType d)
    {
        data = d;
    }

    __host__ __device__ fixed_point(double d)
    {
        *this = d;
    }

    __host__ __device__ fixed_point(float f)
    {
        *this = f;
    }

    __host__ __device__ fixed_point& operator=(double d)
    {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    __host__ __device__ fixed_point& operator=(float d)
    {
        data = static_cast<BaseType>(d*factor);
        return *this;
    }

    __host__ __device__ const BaseType& raw_data() const
    {
        return data;
    }

    __host__ __device__ BaseType& raw_data()
    {
        return data;
    }

    template <typename T>
    __host__ __device__ T extract() const {
        return ((T)data) / factor;
    }
};

}
}
}


