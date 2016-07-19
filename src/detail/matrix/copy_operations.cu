
// Persistent RNN Includes
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/parallel/multi_bulk_synchronous_parallel.h>

#include <prnn/detail/util/metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace prnn
{
namespace matrix
{

namespace detail
{

template<typename NativeResultType, typename NativeInputType>
class CopyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            rawResult[i] = rawInput[i];
        }
    }

public:
          NativeResultType* rawResult;
    const NativeInputType*  rawInput;

public:
    size_t elements;

};

template<typename NativeResultType, typename NativeInputType>
class NoncontiguousCopyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension      = linearToDimension(i, resultView.size());
            auto inputDimension = linearToDimension(i,  inputView.size());

            resultView(dimension) = inputView(inputDimension);
        }
    }

public:
    MatrixView<NativeResultType>     resultView;
    ConstMatrixView<NativeInputType> inputView;

public:
    size_t elements;
};

template<typename ResultPrecision, typename InputPrecision>
void copy(DynamicView& result, const ConstDynamicView& input)
{
    typedef typename ResultPrecision::type NativeResultType;
    typedef typename InputPrecision::type  NativeInputType;

    size_t elements = result.elements();

    if(result.isContiguous() && input.isContiguous())
    {
        auto rawResult = result.data<NativeResultType>();
        auto rawInput  = input.data<const NativeInputType>();

        auto lambda = CopyLambda<NativeResultType, NativeInputType>
            {rawResult, rawInput, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeResultType>     resultView(result);
        ConstMatrixView<NativeInputType> inputView(input);

        auto lambda = NoncontiguousCopyLambda<NativeResultType, NativeInputType>
            {resultView, inputView, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template<typename ResultPrecision, typename T>
void copyOverInputPrecisions(DynamicView& result, const ConstDynamicView& input,
    const std::tuple<T>& inputPrecisions)
{
    typedef T PossibleInputPrecision;

    assert(PossibleInputPrecision() == input.precision());

    copy<ResultPrecision, PossibleInputPrecision>(result, input);
}

template<typename ResultPrecision, typename InputPrecisions>
void copyOverInputPrecisions(DynamicView& result, const ConstDynamicView& input,
    const InputPrecisions& inputPrecisions)
{
    typedef typename std::tuple_element<0, InputPrecisions>::type PossibleInputPrecision;

    if(input.precision() == PossibleInputPrecision())
    {
        copy<ResultPrecision, PossibleInputPrecision>(result, input);
    }
    else
    {
        typedef typename util::RemoveFirstType<InputPrecisions>::type RemainingPrecisions;

        copyOverInputPrecisions<ResultPrecision>(result, input, RemainingPrecisions());
    }

}

template<typename T, typename InputPrecisions>
void copyOverPrecisions(DynamicView& result, const std::tuple<T>& resultPrecisions,
    const ConstDynamicView& input, const InputPrecisions& inputPrecisions)
{
    typedef T PossibleResultPrecision;

    assert(PossibleResultPrecision() == result.precision());

    copyOverInputPrecisions<PossibleResultPrecision>(result, input, inputPrecisions);
}

template<typename ResultPrecisions, typename InputPrecisions>
void copyOverPrecisions(DynamicView& result, const ResultPrecisions& resultPrecisions,
    const ConstDynamicView& input, const InputPrecisions& inputPrecisions)
{
    typedef typename std::tuple_element<0, ResultPrecisions>::type PossibleResultPrecision;

    if(result.precision() == PossibleResultPrecision())
    {
        copyOverInputPrecisions<PossibleResultPrecision>(result, input, inputPrecisions);
    }
    else
    {
        typedef typename util::RemoveFirstType<ResultPrecisions>::type RemainingPrecisions;

        copyOverPrecisions(result, RemainingPrecisions(), input, inputPrecisions);
    }
}

void copyOverPrecisions(const DynamicView& result, const ConstDynamicView& input)
{
    DynamicView resultView(result);

    copyOverPrecisions(resultView, AllPrecisions(), input, AllPrecisions());
}

}

void copy(const DynamicView& result, const ConstDynamicView& input)
{
    detail::copyOverPrecisions(result, input);
}

void copy(const Matrix& result, const Matrix& input)
{
    Matrix temp(result);

    detail::copyOverPrecisions(DynamicView(temp), ConstDynamicView(input));
}

Matrix copy(const Matrix& input)
{
    Matrix result(input.size(), input.precision());

    copy(result, input);

    return result;
}

Matrix copy(const Matrix& input, const Precision& precision)
{
    Matrix result(input.size(), precision);

    copy(result, input);

    return result;
}

}
}



