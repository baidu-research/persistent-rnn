
// Persistent RNN Includes
#include <prnn/detail/matrix/float_reference.h>

#include <prnn/detail/util/metaprogramming.h>

#include <prnn/detail/parallel/synchronization.h>

// Standard Library Includes
#include <cassert>

namespace prnn
{
namespace matrix
{
namespace detail
{

template<typename T>
void set(void* data, double value)
{
    parallel::synchronize();

    *static_cast<typename T::type*>(data) = value;
}

template<typename PossiblePrecisionType>
void setOverPrecisions(const Precision& precision,
    const std::tuple<PossiblePrecisionType>& precisions, void* data, double value)
{
    assert(precision == PossiblePrecisionType());

    set<PossiblePrecisionType>(data, value);
}

template<typename PossiblePrecisions>
void setOverPrecisions(const Precision& precision,
    const PossiblePrecisions& precisions, void* data, double value)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        set<PossiblePrecisionType>(data, value);
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        setOverPrecisions(precision, RemainingPrecisions(), data, value);
    }
}

static void set(const Precision& precision, void* data, double value)
{
    setOverPrecisions(precision, AllPrecisions(), data, value);
}

template<typename T>
double get(const void* data)
{
    parallel::synchronize();
    return *static_cast<const typename T::type*>(data);
}

template<typename PossiblePrecisionType>
double getOverPrecisions(const Precision& precision,
    const std::tuple<PossiblePrecisionType>& precisions, const void* data)
{
    assert(precision == PossiblePrecisionType());

    return get<PossiblePrecisionType>(data);
}

template<typename PossiblePrecisions>
double getOverPrecisions(const Precision& precision,
    const PossiblePrecisions& precisions, const void* data)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        return get<PossiblePrecisionType>(data);
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return getOverPrecisions(precision, RemainingPrecisions(), data);
    }
}

static double get(const Precision& precision, const void* data)
{
    return getOverPrecisions(precision, AllPrecisions(), data);
}

}

FloatReference::FloatReference(const Precision& p, void* d)
: _precision(p), _data(d)
{

}

FloatReference& FloatReference::operator=(double value)
{
    detail::set(_precision, _data, value);

    return *this;
}

FloatReference& FloatReference::operator+=(double value)
{
    detail::set(_precision, _data, detail::get(_precision, _data) + value);

    return *this;
}

FloatReference& FloatReference::operator-=(double value)
{
    detail::set(_precision, _data, detail::get(_precision, _data) - value);

    return *this;
}

FloatReference::operator double() const
{
    return detail::get(_precision, _data);
}

void* FloatReference::address()
{
    return _data;
}

const void* FloatReference::address() const
{
    return _data;
}

ConstFloatReference::ConstFloatReference(const Precision& p, const void* d)
: _precision(p), _data(d)
{

}

ConstFloatReference::operator double() const
{
    return detail::get(_precision, _data);
}

const void* ConstFloatReference::address() const
{
    return _data;
}

}
}





