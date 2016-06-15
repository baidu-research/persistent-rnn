
#pragma once

// Persistent RNN Includes
#include <prnn/detail/types/float16.h>

#include <prnn/detail/parallel/cuda.h>

// Standard Library Includes
#include <tuple>
#include <memory>
#include <string>

namespace prnn
{
namespace matrix
{

class Precision
{
public:
    enum Type
    {
        NoType,
        Half,
        Single,
        Double,
    };

public:
    Precision();
    explicit Precision(Type t);

public:
    Type type() const;

public:
    CUDA_DECORATOR size_t size() const
    {
        switch(_type)
        {
        case Half: return sizeof(float)/2;
        case Single: return sizeof(float);
        case Double: return sizeof(double);
        default:
            return 0;
        }
    }

public:
    std::string toString() const;

public:
    static std::unique_ptr<Precision> fromString(const std::string& name);

public:
    bool operator==(const Precision&) const;
    bool operator!=(const Precision&) const;

public:
    static Precision getDefaultPrecision();

private:
    Type _type;

};

class HalfPrecision : public Precision
{
public:
    typedef types::float16 type;

public:
    HalfPrecision();

};

class SinglePrecision : public Precision
{
public:
    typedef float type;

public:
    SinglePrecision();

};

class DoublePrecision : public Precision
{
public:
    typedef double type;

public:
    DoublePrecision();

};

typedef std::tuple<SinglePrecision, DoublePrecision> AllPrecisions;
typedef std::tuple<SinglePrecision> RecurrentPrecisions;

}
}


