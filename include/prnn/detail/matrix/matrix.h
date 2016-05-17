/*  \file   Matrix.h
    \brief  The header file for the Matrix class.
*/

#pragma once

// Persistent RNN Includes
#include <prnn/detail/matrix/float_reference.h>
#include <prnn/detail/matrix/float_iterator.h>
#include <prnn/detail/matrix/dimension.h>
#include <prnn/detail/matrix/precision.h>

// Standard Library Includes
#include <string>
#include <cstddef>
#include <memory>

// Forward Declarations
namespace prnn { namespace matrix { class Allocation; } }

namespace prnn
{

namespace matrix
{

/*! \brief An interface to operations on a general purpose array. */
class Matrix
{
public:
    Matrix();
    Matrix(std::initializer_list<size_t>);
    explicit Matrix(const Dimension& size);
    Matrix(const Dimension& size, const Precision& precision);
    Matrix(const Dimension& size, const Dimension& stride);
    Matrix(const Dimension& size, const Dimension& stride, const Precision& precision);
    Matrix(const Dimension& size, const Dimension& stride, const Precision& precision,
        const std::shared_ptr<Allocation>& allocation);
    Matrix(const Dimension& size, const Dimension& stride, const Precision& precision,
        const std::shared_ptr<Allocation>& allocation, void* start);

public:
    ~Matrix();

public:
    const Dimension& size()   const;
    const Dimension& stride() const;

public:
    const Precision& precision() const;

public:
    size_t elements() const;
    bool empty() const;

public:
    FloatIterator begin();
    FloatIterator end();

    ConstFloatIterator begin() const;
    ConstFloatIterator end()   const;

public:
    std::shared_ptr<Allocation> allocation();

public:
          void* data();
    const void* data() const;

public:
    bool isContiguous() const;
    bool isLeadingDimensionContiguous() const;

public:
    std::string toString() const;
    std::string debugString() const;
    std::string shapeString() const;

public:
    template<typename... Args>
    FloatReference operator()(Args... args)
    {
        return (*this)[Dimension(args...)];
    }

    template<typename... Args>
    ConstFloatReference operator()(Args... args) const
    {
        return (*this)[Dimension(args...)];
    }

public:
    FloatReference      operator[](const Dimension& d);
    ConstFloatReference operator[](const Dimension& d) const;

public:
    bool operator==(const Matrix& m) const;
    bool operator!=(const Matrix& m) const;

public:
    template<typename... Sizes>
    Matrix(Sizes... sizes)
    : Matrix(Dimension(sizes...))
    {

    }

private:
    std::shared_ptr<Allocation> _allocation;

private:
    void* _data_begin;

private:
    Dimension _size;
    Dimension _stride;

private:
    Precision _precision;

};

}

}


