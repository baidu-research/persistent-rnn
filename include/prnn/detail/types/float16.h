#pragma once

#include <prnn/detail/parallel/cuda.h>

#include <cstdint>
#include <ostream>
#include <istream>

#ifndef __APPLE__
    #include <immintrin.h>
#else
    #ifdef __F16C__
        #undef __F16C__
    #endif
#endif

//Conversion routine adapted from
//http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion

namespace prnn {
namespace types {

/** 16-bit floating point class.  Conversions are done with intrinsics on GPU and
    slowly with emulation on CPU.  Operations are supported by casting to floats.
*/

class float16 {
public:
    uint16_t val_;

    CUDA_DECORATOR
    inline
    float16() {};

    CUDA_DECORATOR
    inline float16(float value) {
        #ifdef __CUDA_ARCH__
            val_ = __float2half_rn(value);
        #elif defined(__F16C__)
            //the 0 sets the rounding mode
            //0 == round to nearest even
            val_ = _cvtss_sh(value, 0);
        #else
            Bits v, s;
            v.f = value;
            uint32_t sign = v.si & sigN;
            v.si ^= sign;
            sign >>= shiftSign; // logical shift
            s.si = mulN;
            s.si = s.f * v.f; // correct subnormals
            v.si ^= (s.si ^ v.si) & -(minN > v.si);
            v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
            v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
            v.ui >>= shift; // logical shift
            v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
            v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
            val_ = v.ui | sign;
        #endif
    }

    CUDA_DECORATOR
    inline
    float to_float() const {
        #ifdef __CUDA_ARCH__
            return __half2float(val_);
        #elif defined(__F16C__)
            return _cvtsh_ss(val_);
        #else
            Bits v;
            v.ui = val_;
            int32_t sign = v.si & sigC;
            v.si ^= sign;
            sign <<= shiftSign;
            v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
            v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
            Bits s;
            s.si = mulC;
            s.f *= v.si;
            int32_t mask = -(norC > v.si);
            v.si <<= shift;
            v.si ^= (s.si ^ v.si) & mask;
            v.si |= sign;
            return v.f;
        #endif
    }

    CUDA_DECORATOR
    inline
    float16 operator-() const {
        float16 tmp(-(this->to_float()));
        return tmp;
    }

    CUDA_DECORATOR
    inline
    bool operator==(float16 rhs) const {
        return this->to_float() == rhs.to_float();
    }

    CUDA_DECORATOR
    inline
    bool operator<(float16 rhs) const {
        return this->to_float() < rhs.to_float();
    }

    CUDA_DECORATOR
    inline
    bool operator<=(float16 rhs) const {
        return this->to_float() <= rhs.to_float();
    }

    CUDA_DECORATOR
    inline
    bool operator>(float16 rhs) const {
        return this->to_float() > rhs.to_float();
    }

    CUDA_DECORATOR
    inline
    bool operator>=(float16 rhs) const {
        return this->to_float() >= rhs.to_float();
    }

    CUDA_DECORATOR
    inline
    bool operator==(float rhs) const {
        return this->to_float() == rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<(float rhs) const {
        return this->to_float() < rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<=(float rhs) const {
        return this->to_float() <= rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>(float rhs) const {
        return this->to_float() > rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>=(float rhs) const {
        return this->to_float() >= rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator==(double rhs) const {
        return this->to_float() == rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<(double rhs) const {
        return this->to_float() < rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<=(double rhs) const {
        return this->to_float() <= rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>(double rhs) const {
        return this->to_float() > rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>=(double rhs) const {
        return this->to_float() >= rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator==(int rhs) const {
        return this->to_float() == rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<(int rhs) const {
        return this->to_float() < rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator<=(int rhs) const {
        return this->to_float() <= rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>(int rhs) const {
        return this->to_float() > rhs;
    }

    CUDA_DECORATOR
    inline
    bool operator>=(int rhs) const {
        return this->to_float() >= rhs;
    }

    CUDA_DECORATOR
    inline
    float16& operator+=(const float16& rhs) {
        this->val_ = float16(this->to_float() + rhs.to_float()).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float operator+(float rhs) {
        return this->to_float() + rhs;
    }

    CUDA_DECORATOR
    inline
    float operator+(const float &rhs) {
        return this->to_float() + rhs;
    }

    CUDA_DECORATOR
    inline
    float16& operator-=(const float16 &rhs) {
        this->val_ = float16(this->to_float() - rhs.to_float()).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float16 operator-(float16 rhs) {
        return float16(this->to_float() - rhs.to_float());
    }

    CUDA_DECORATOR
    inline
    float operator-(float rhs) {
        return this->to_float() - rhs;
    }

    CUDA_DECORATOR
    inline
    float16& operator*=(float16 rhs) {
        this->val_ = float16(this->to_float() * rhs.to_float()).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float16 operator*(const float16 &rhs) {
        return float16(this->to_float() * rhs.to_float());
    }

    CUDA_DECORATOR
    inline
    float operator*(const float &rhs) {
        return this->to_float() * rhs;
    }

    CUDA_DECORATOR
    inline
    float16& operator/=(const float16 &rhs) {
        this->val_ = float16(this->to_float() / rhs.to_float()).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float16 operator/(float16 rhs) {
        return float16(this->to_float() / rhs.to_float());
    }

    CUDA_DECORATOR
    inline
    float operator/(float rhs) {
        return this->to_float() / rhs;
    }

    CUDA_DECORATOR
    inline
    float16& operator=(const float16& rhs) {
        this->val_ = rhs.val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float16& operator=(const float& rhs) {
        this->val_ = float16(rhs).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    float16& operator=(const double& rhs) {
        this->val_ = float16(rhs).val_;
        return *this;
    }

    CUDA_DECORATOR
    inline
    operator float() const {
        return to_float();
    }

    inline
    friend std::ostream& operator<<(std::ostream& os, const float16& val) {
        os << val.to_float();
        return os;
    }

    inline
    friend std::istream& operator>>(std::istream& os, float16& val) {
        float tmp;
        os >> tmp;

        val = float16(tmp);
        return os;
    }

private:

    union Bits {
        float f;
        int32_t si;
        uint32_t ui;
    };

    static const int shift = 13;
    static const int shiftSign = 16;

    static const int32_t infN = 0x7F800000;
    static const int32_t maxN = 0x477FE000; //max flt16 as flt32
    static const int32_t minN = 0x38800000; //min flt16 normal as flt32
    static const int32_t sigN = 0x80000000; //sign bit

    static constexpr int32_t infC = infN >> shift;
    static constexpr int32_t nanN = (infC + 1) << shift; //minimum flt16 nan as float32
    static constexpr int32_t maxC = maxN >> shift;
    static constexpr int32_t minC = minN >> shift;
    static constexpr int32_t sigC = sigN >> shiftSign;

    static const int32_t mulN = 0x52000000; //(1 << 23) / minN
    static const int32_t mulC = 0x33800000; //minN / (1 << (23 - shift))
    static const int32_t subC = 0x003FF; //max flt32 subnormal downshifted
    static const int32_t norC = 0x00400; //min flt32 normal downshifted

    static constexpr int32_t maxD = infC - maxC - 1;
    static constexpr int32_t minD = minC - subC - 1;
};

CUDA_DECORATOR
inline
float16 operator*(const float16& lhs, const float16& rhs) {
    return float16(lhs.to_float() * rhs.to_float());
}

CUDA_DECORATOR
inline
bool operator==(const float& lhs, const float16& rhs) {
    return rhs.to_float() == lhs;
}

CUDA_DECORATOR
inline
bool operator!=(const float& lhs, const float16& rhs) {
    return rhs.to_float() != lhs;
}

CUDA_DECORATOR
inline
bool operator<(const float& lhs, const float16& rhs) {
    return lhs < rhs.to_float();
}


CUDA_DECORATOR
inline
bool operator>(const float& lhs, const float16& rhs) {
    return lhs > rhs.to_float();
}


CUDA_DECORATOR
inline
bool operator<=(const float& lhs, const float16& rhs) {
    return lhs <= rhs.to_float();
}


CUDA_DECORATOR
inline
bool operator>=(const float& lhs, const float16& rhs) {
    return lhs >= rhs.to_float();
}


CUDA_DECORATOR
inline
double operator-(const double& lhs, const float16& rhs) {
    return lhs - static_cast<double>(rhs.to_float());
}

CUDA_DECORATOR
inline
double operator+(const double& lhs, const float16& rhs) {
    return lhs + static_cast<double>(rhs.to_float());
}

CUDA_DECORATOR
inline
double operator*(const double& lhs, const float16& rhs) {
    return lhs * static_cast<double>(rhs.to_float());
}

CUDA_DECORATOR
inline
double operator/(const double& lhs, const float16& rhs) {
    return lhs / static_cast<double>(rhs.to_float());
}


CUDA_DECORATOR
inline
float operator-(const float& lhs, const float16& rhs) {
    return lhs - rhs.to_float();
}

CUDA_DECORATOR
inline
float operator+(const float& lhs, const float16& rhs) {
    return lhs + rhs.to_float();
}

CUDA_DECORATOR
inline
float16 operator+(const float16& lhs, const float16& rhs) {
    return float16(lhs.to_float() + rhs.to_float());
}

CUDA_DECORATOR
inline
float16 operator-(const float16& lhs, const float16& rhs) {
    return float16(lhs.to_float() - rhs.to_float());
}

CUDA_DECORATOR
inline
float16 operator/(const float16& lhs, const float16& rhs) {
    return float16(lhs.to_float() / rhs.to_float());
}

CUDA_DECORATOR
inline
float operator*(const float& lhs, const float16& rhs) {
    return lhs * rhs.to_float();
}

CUDA_DECORATOR
inline
float operator/(const float& lhs, const float16& rhs) {
    return lhs / rhs.to_float();
}

CUDA_DECORATOR
inline
float operator-(const int& lhs, const float16& rhs) {
    return lhs - rhs.to_float();
}

CUDA_DECORATOR
inline
float operator+(const int& lhs, const float16& rhs) {
    return lhs + rhs.to_float();
}

CUDA_DECORATOR
inline
float operator/(const int& lhs, const float16& rhs) {
    return lhs / rhs.to_float();
}
CUDA_DECORATOR
inline
float operator*(const int& lhs, const float16& rhs) {
    return lhs * rhs.to_float();
}

CUDA_DECORATOR
inline
float operator*(const bool& lhs, const float16& rhs) {
    return lhs * rhs.to_float();
}

}
}

