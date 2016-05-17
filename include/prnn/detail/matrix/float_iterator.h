
#pragma once

// Persistent RNN Includes
#include <prnn/detail/matrix/float_reference.h>
#include <prnn/detail/matrix/dimension.h>

namespace prnn
{
namespace matrix
{

class ConstFloatIterator;

class FloatIterator
{
public:
    FloatIterator();
    FloatIterator(const Precision& p, const Dimension& size,
        const Dimension& stride, const Dimension& offset, void* d);

public:
    FloatIterator(const FloatIterator& );

public:
    FloatIterator& operator=(const FloatIterator&);

public:
    FloatReference operator*();

public:
    FloatIterator& operator++();

public:
    bool operator==(const FloatIterator&) const;
    bool operator==(const ConstFloatIterator&) const;

public:
    bool operator!=(const FloatIterator&) const;
    bool operator!=(const ConstFloatIterator&) const;

private:
    void*     _data;
    Dimension _stride;
    Dimension _size;
    Dimension _offset;
    Precision _precision;

private:
    friend class ConstFloatIterator;

};

class ConstFloatIterator
{
public:
    ConstFloatIterator();
    ConstFloatIterator(const Precision& p, const Dimension& s,
        const Dimension& stride, const Dimension& offset, const void* d);

public:
    ConstFloatIterator(const FloatIterator& );
    ConstFloatIterator(const ConstFloatIterator& );

public:
    ConstFloatIterator& operator=(const FloatIterator&);
    ConstFloatIterator& operator=(const ConstFloatIterator&);

public:
    ConstFloatReference operator*();

public:
    ConstFloatIterator& operator++();

public:
    bool operator==(const FloatIterator&) const;
    bool operator==(const ConstFloatIterator&) const;

public:
    bool operator!=(const FloatIterator&) const;
    bool operator!=(const ConstFloatIterator&) const;

private:
    const void* _data;
    Dimension   _stride;
    Dimension   _size;
    Dimension   _offset;
    Precision   _precision;

private:
    friend class FloatIterator;

};

}
}



