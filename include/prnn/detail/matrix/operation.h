
#pragma once

// Persistent RNN Includes
#include <prnn/detail/types/float16.h>

#include <prnn/detail/parallel/cuda.h>
#include <prnn/detail/parallel/scalar_operations.h>

#include <prnn/detail/matrix/dimension_transformations.h>

// Standard Library Includes
#include <cmath>
#include <algorithm>

namespace prnn {
namespace matrix {

/*! \brief A class for specifying basic matrix operations. */
class Operation
{
public:
    enum Type
    {
        Add,
        Subtract,
        Multiply,
        MultiplyAccumulate,
        Divide,
        Log,
        Exp,
        Pow,
        Abs,
        Sqrt,
        Sigmoid,
        SigmoidDerivative,
        RectifiedLinear,
        RectifiedLinearDerivative,
        Tanh,
        TanhDerivative,
        KLDivergence,
        KLDivergenceDerivative,
        Negate,
        Maximum,
        Minimum,
        Equal,
        LessThan,
        NotEqual,
        LessThanOrEqual,
        GreaterThanOrEqual,
        Fill,
        Square,
        SquareAndScale,
        Inverse,
        CopyRight
    };

public:
    CUDA_DECORATOR Operation(Type t) : _type(t) {}

public:
    CUDA_DECORATOR bool operator==(const Operation&) const;

private:
    Type _type;

};

class Add : public Operation
{
public:
    CUDA_DECORATOR Add() : Operation(Operation::Add), _value(0.0)
    {

    }

    CUDA_DECORATOR Add(double d) : Operation(Operation::Add), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return l + r;
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value) + r;
    }

    CUDA_DECORATOR types::float16 operator()(const types::float16& r) const
    {
        return types::float16(_value + r.to_float());
    }

private:
    double _value;

};

class Subtract : public Operation
{
public:
    CUDA_DECORATOR Subtract() : Operation(Operation::Subtract), _value(0.0)
    {

    }

    CUDA_DECORATOR Subtract(double d) : Operation(Operation::Subtract), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return l - r;
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r - T(_value);
    }

private:
    double _value;

};

class Log : public Operation
{
public:
    CUDA_DECORATOR Log() : Operation(Operation::Log)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return log(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return logf(l);
    }
};

class Exp : public Operation
{
public:
    CUDA_DECORATOR Exp() : Operation(Operation::Exp)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return exp(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return expf(l);
    }
};

class Pow : public Operation
{
public:
    CUDA_DECORATOR Pow() : Operation(Operation::Pow), _value(0.0)
    {

    }

    CUDA_DECORATOR Pow(double v) : Operation(Operation::Pow), _value(v)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return pow(l, T(_value));
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return powf(l, static_cast<float>(_value));
    }

public:
    double _value;
};

class Abs : public Operation
{
public:
    CUDA_DECORATOR Abs() : Operation(Operation::Abs)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return std::abs(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return fabs(l);
    }
};

class Sqrt : public Operation
{
public:
    CUDA_DECORATOR Sqrt() : Operation(Operation::Sqrt)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return sqrt(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return sqrtf(l);
    }
};

class Multiply : public Operation
{
public:
    CUDA_DECORATOR Multiply() : Operation(Operation::Multiply), _value(0.0)
    {

    }

    CUDA_DECORATOR Multiply(double d) : Operation(Operation::Multiply), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return l * r;
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value) * r;
    }

private:
    double _value;

};

class MultiplyAccumulate : public Operation
{
public:
    CUDA_DECORATOR MultiplyAccumulate() : Operation(Operation::MultiplyAccumulate), _value(0.0)
    {

    }

    CUDA_DECORATOR MultiplyAccumulate(double d)
    : Operation(Operation::MultiplyAccumulate), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(_value) * l + r;
    }

private:
    double _value;

};

class Divide : public Operation
{
public:
    CUDA_DECORATOR Divide() : Operation(Operation::Divide), _value(0.0)
    {

    }

    CUDA_DECORATOR Divide(double d) : Operation(Operation::Divide), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return l / r;
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r / T(_value);
    }

private:
    double _value;

};

class Sigmoid : public Operation
{
public:
    CUDA_DECORATOR Sigmoid() : Operation(Operation::Sigmoid)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        if(l < -50.0) return -50.0;
        if(l >  50.0) return  50.0;

        return 1.0 / (1.0 + exp(-l));
    }

    CUDA_DECORATOR types::float16 operator()(const types::float16& l) const
    {
        if(l < -50.0) return -50.0;
        if(l >  50.0) return  50.0;

        return types::float16(1.0 / (1.0 + exp(-l.to_float())));
    }

};

class SigmoidDerivative : public Operation
{
public:
    CUDA_DECORATOR SigmoidDerivative() : Operation(Operation::SigmoidDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return l * (T(1.0) - l);
    }

};

class RectifiedLinear : public Operation
{
public:
    CUDA_DECORATOR RectifiedLinear() : Operation(Operation::RectifiedLinear)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return parallel::min(parallel::max(l, T(0.0)), T(20.0));
    }

};

class RectifiedLinearDerivative : public Operation
{
public:
    CUDA_DECORATOR RectifiedLinearDerivative() : Operation(Operation::RectifiedLinearDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& x, const T& y) const
    {
        return (((x > T(0.0)) && (x < T(20.0))) * y);
    }

};

class Negate : public Operation
{
public:
    CUDA_DECORATOR Negate() : Operation(Operation::Negate)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return -l;
    }

};

class Maximum : public Operation
{
public:
    CUDA_DECORATOR Maximum() : Operation(Operation::Maximum), _value(0.0)
    {

    }

    CUDA_DECORATOR Maximum(double d) : Operation(Operation::Maximum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return parallel::max(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return parallel::max(T(_value), r);
    }

private:
    double _value;

};

class Minimum : public Operation
{
public:
    CUDA_DECORATOR Minimum() : Operation(Operation::Minimum), _value(0.0)
    {

    }

    CUDA_DECORATOR Minimum(double d) : Operation(Operation::Minimum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return parallel::min(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return parallel::min(T(_value), r);
    }

private:
    double _value;

};

class Equal : public Operation
{
public:
    CUDA_DECORATOR Equal() : Operation(Operation::Equal), _value(0.0)
    {

    }

    CUDA_DECORATOR Equal(double d) : Operation(Operation::Equal), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l == r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value) == r;
    }

private:
    double _value;

};

class LessThan : public Operation
{
public:
    CUDA_DECORATOR LessThan() : Operation(Operation::LessThan), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThan(double d) : Operation(Operation::LessThan), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l < r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r < T(_value);
    }

private:
    double _value;

};

class NotEqual : public Operation
{
public:
    CUDA_DECORATOR NotEqual() : Operation(Operation::NotEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR NotEqual(double d) : Operation(Operation::NotEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l != r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value) != r;
    }

private:
    double _value;

};

class GreaterThanOrEqual : public Operation
{
public:
    CUDA_DECORATOR GreaterThanOrEqual() : Operation(Operation::GreaterThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR GreaterThanOrEqual(double d) : Operation(Operation::GreaterThanOrEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l >= r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r >= T(_value);
    }

private:
    double _value;

};

class LessThanOrEqual : public Operation
{
public:
    CUDA_DECORATOR LessThanOrEqual() : Operation(Operation::LessThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThanOrEqual(double d) : Operation(Operation::LessThanOrEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l <= r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r <= T(_value);
    }

private:
    double _value;

};

class Fill : public Operation
{
public:
    CUDA_DECORATOR Fill(double d = 0.0) : Operation(Operation::Fill), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return _value;
    }

private:
    double _value;

};

class Square : public Operation
{
public:
    CUDA_DECORATOR Square() : Operation(Operation::Square)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r;
    }

};

class SquareAndScale : public Operation
{
public:
    CUDA_DECORATOR SquareAndScale(double d = 1.0) : Operation(Operation::SquareAndScale), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r * T(_value);
    }

private:
    double _value;

};

class Inverse : public Operation
{
public:
    CUDA_DECORATOR Inverse() : Operation(Operation::Inverse)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(1.0) / r;
    }

};

class CopyRight : public Operation
{
public:
    CUDA_DECORATOR CopyRight() : Operation(Operation::CopyRight)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return r;
    }

};

class Tanh : public Operation
{
public:
    CUDA_DECORATOR Tanh() : Operation(Operation::Tanh)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& x) const
    {
        return std::tanh(x);
    }
};

class TanhDerivative : public Operation
{
public:
    CUDA_DECORATOR TanhDerivative() : Operation(Operation::TanhDerivative)
    {

    }

public:
    template<typename T, typename U>
    CUDA_DECORATOR auto operator()(const T& x, const U& y) const -> decltype(x*y)
    {
        return ((1 - x * x) * y);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& x, const T& y) const {
        return T((T(1) - x * x) * y);
    }

};

typedef std::tuple<RectifiedLinear, Tanh> AllRecurrentForwardOps;
typedef std::tuple<RectifiedLinearDerivative, TanhDerivative> AllRecurrentBackwardOps;

typedef std::tuple<Add, Subtract, Multiply, MultiplyAccumulate, Divide, Log, Exp, Pow, Abs, Sqrt,
                   RectifiedLinear, RectifiedLinearDerivative, Sigmoid, SigmoidDerivative, Negate,
                   Maximum, Minimum, Equal, LessThan, NotEqual, Fill, Square, SquareAndScale,
                   Inverse> AllOperations;

typedef std::tuple<Add, Subtract, Multiply, MultiplyAccumulate, Divide, Maximum, Minimum,
                   Equal, LessThan, NotEqual, CopyRight, RectifiedLinearDerivative,
                   TanhDerivative> AllBinaryOperations;

typedef std::tuple<Add, Subtract, Multiply, Divide, Log, Exp, Pow, Abs, Sqrt, RectifiedLinear,
                   Sigmoid, Negate, Maximum,
                   Minimum, Equal, LessThan, NotEqual, Fill, Square, SquareAndScale, Inverse
                   > AllUnaryOperations;

}
}

