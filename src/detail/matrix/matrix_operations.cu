
// Persistent RNN Includes
#include <prnn/detail/matrix/matrix_operations.h>
#include <prnn/detail/matrix/matrix_transforms.h>
#include <prnn/detail/matrix/copy_operations.h>
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/operation.h>

#include <prnn/detail/parallel/multi_bulk_synchronous_parallel.h>
#include <prnn/detail/parallel/shared_memory_allocator.h>

#include <prnn/detail/util/metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace prnn
{
namespace matrix
{

namespace detail
{

template<typename NativeType, typename OperationType>
class BinaryApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            resultBase[i] = nativeOperation(leftBase[i], rightBase[i]);
        }
    }

public:
          NativeType* resultBase;
    const NativeType* leftBase;
    const NativeType* rightBase;

public:
    OperationType nativeOperation;

public:
    size_t elements;

};

template<typename NativeType, typename OperationType>
class BinaryNoncontiguousApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto fullDimension = linearToDimension(i, resultView.size());

            resultView(fullDimension) = nativeOperation(leftView(fullDimension),
                rightView(fullDimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> leftView;
    ConstMatrixView<NativeType> rightView;

public:
    OperationType nativeOperation;

public:
    size_t elements;

};

template<typename OperationType, typename T>
void applyOverPrecisions(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    if(result.isContiguous() && left.isContiguous() && right.isContiguous())
    {
        auto resultBase = result.data<NativeType>();
        auto leftBase   = left.data<NativeType>();
        auto rightBase  = right.data<NativeType>();

        auto lambda = BinaryApplyLambda<NativeType, OperationType>
            {resultBase, leftBase, rightBase, nativeOperation, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> leftView(left);
        ConstMatrixView<NativeType> rightView(right);

        auto lambda = BinaryNoncontiguousApplyLambda<NativeType, OperationType>
            {resultView, leftView, rightView, nativeOperation, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right, const Operation& op,
    const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        applyOverPrecisions<OperationType>(result, left, right, op, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        applyOverPrecisions<OperationType>(result, left, right,
            op, precision, RemainingPrecisions());
    }
}


template<typename T>
void applyOverOperations(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right, const Operation& op,
    const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, left, right,
        op, precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right,
    const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        applyOverOperations(result, left, right, op, precision,
            std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        applyOverOperations(result, left, right, op, precision, RemainingOperations());
    }
}

void applyOverOperations(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right,
    const Operation& op, const Precision& precision)
{
    applyOverOperations<AllBinaryOperations>(result, left, right, op, precision,
        AllBinaryOperations());
}

}

void apply(const DynamicView& result, const ConstDynamicView& left,
    const ConstDynamicView& right, const Operation& op)
{
    auto precision = left.precision();

    assert(left.size() == right.size());
    assert(result.size() == right.size());
    assert(left.precision() == right.precision());
    assert(result.precision() == right.precision());

    detail::applyOverOperations(result, left, right, op, precision);
}

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op)
{
    apply(DynamicView(result), ConstDynamicView(left), ConstDynamicView(right), op);
}

Matrix apply(const Matrix& left, const Matrix& right, const Operation& op)
{
    assert(left.size()      == right.size());
    assert(left.precision() == right.precision());

    Matrix temp(left.size(), left.precision());

    apply(temp, left, right, op);

    return temp;
}

namespace detail
{

template<typename NativeType, typename OperationType>
class UnaryApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            resultBase[i] = nativeOperation(inputBase[i]);
        }
    }

public:
          NativeType* resultBase;
    const NativeType* inputBase;

public:
    size_t elements;

public:
    OperationType nativeOperation;
};

template<typename NativeType, typename OperationType>
class UnaryNoncontiguousApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension = linearToDimension(i, resultView.size());

            resultView(dimension) = nativeOperation(inputView(dimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    OperationType nativeOperation;
};

template<typename OperationType, typename T>
void applyOverPrecisions(const DynamicView& result, const ConstDynamicView& input,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    if(input.isContiguous() && result.isContiguous())
    {
        auto resultBase = result.data<NativeType>();
        auto inputBase  = input.data<NativeType>();

        auto lambda = UnaryApplyLambda<NativeType, OperationType>
            {resultBase, inputBase, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> inputView(input);

        auto lambda = UnaryNoncontiguousApplyLambda<NativeType, OperationType>
            {resultView, inputView, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(const DynamicView& result, const ConstDynamicView& input,
    const Operation& op, const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        applyOverPrecisions<OperationType>(result, input, op, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        applyOverPrecisions<OperationType>(result, input, op, precision, RemainingPrecisions());
    }
}


template<typename T>
void applyOverOperations(const DynamicView& result, const ConstDynamicView& input,
    const Operation& op, const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, input, op,
        precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(const DynamicView& result, const ConstDynamicView& input,
    const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        applyOverOperations(result, input, op, precision, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        applyOverOperations(result, input, op, precision, RemainingOperations());
    }
}

void applyOverOperations(const DynamicView& result, const ConstDynamicView& input,
    const Operation& op, const Precision& precision)
{
    applyOverOperations<AllUnaryOperations>(result, input, op, precision, AllUnaryOperations());
}

}

void apply(const DynamicView& result, const ConstDynamicView& input, const Operation& op)
{
    detail::applyOverOperations(result, input, op, input.precision());
}

void apply(Matrix& result, const Matrix& input, const Operation& op)
{
    apply(DynamicView(result), ConstDynamicView(input), op);
}

Matrix apply(const Matrix& input, const Operation& op)
{
    Matrix result(input.size(), input.precision());

    apply(result, input, op);

    return result;
}

namespace detail
{

static const size_t tileSize = 8;

template<typename NativeType, typename ActualOperation>
class ReduceAllDimensionsStepLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            NativeType value = rawInput[i];

            for(size_t inputElement = i + elements; inputElement < inputElements;
                inputElement += elements)
            {
                value = nativeOperation(value, rawInput[inputElement]);
            }

            rawResult[i] = value;
        }
    }

public:
          NativeType* rawResult;
    const NativeType* rawInput;

public:
    size_t elements;
    size_t inputElements;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename ActualPrecision>
void reduceAllDimensionsStep(Matrix& result, const Matrix& input,
    const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    typedef typename ActualPrecision::type NativeType;

    NativeType*       rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawInput  = static_cast<const NativeType*>(input.data());

    auto lambda = ReduceAllDimensionsStepLambda<NativeType, ActualOperation>{rawResult, rawInput,
        result.elements(), input.elements(), nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template <typename ActualOperation, typename ActualPrecision>
void reduceAllDimensions(Matrix& result, const Matrix& input,
    const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    if(input.elements() < tileSize)
    {
        return reduceAllDimensionsStep(result, input, nativeOperation, p);
    }

    Matrix temporary({(input.elements() + tileSize - 1) / tileSize}, p);

    reduceAllDimensionsStep(temporary, input, nativeOperation, p);

    reduceAllDimensions(result, temporary, nativeOperation, p);
}

template<typename NativeType, typename ActualOperation>
class ReduceFirstDimensionLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, 2);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        size_t rows = inputElements / columns;

        for(size_t column = relativeGroup.id(); column < columns; column += relativeGroup.size())
        {
            size_t row         = innerGroup.id();
            size_t position    = column * rows + row;
            size_t endPosition = (column + 1) * rows;

            if(innerGroup.size() <= rows)
            {
                NativeType value = rawInput[position];

                for(position += innerGroup.size(); position < endPosition;
                    position += innerGroup.size())
                {
                    value = nativeOperation(value, rawInput[position]);
                }

                value = reduce(innerGroup, value, nativeOperation);

                if(innerGroup.id() == 0)
                {
                    rawResult[column] = value;
                }
            }
            else if(innerGroup.id() == 0)
            {
                NativeType value = rawInput[position];
                ++position;

                for(size_t row = 1; row < rows; ++row, ++position)
                {
                    value = nativeOperation(value, rawInput[position]);
                }

                rawResult[column] = value;
            }
        }
    }

public:
          NativeType* rawResult;
    const NativeType* rawInput;

public:
    size_t columns;
    size_t inputElements;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename ActualPrecision>
void reduceFirstDimension(Matrix& result, const Matrix& input,
    const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    typedef typename ActualPrecision::type NativeType;

    NativeType*       rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawInput  = static_cast<const NativeType*>(input.data());

    auto lambda = ReduceFirstDimensionLambda<NativeType, ActualOperation>{rawResult, rawInput,
        result.elements(), input.elements(), nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename NativeType, typename ActualOperation>
class ReduceSecondDimensionLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        auto warp        = partitionThreadGroupAtLevel(threadGroup, 1);
        auto cta         = partitionThreadGroupAtLevel(threadGroup, 2);
        auto ctaInKernel = getRelativeGroup(cta, threadGroup);
        auto warpInCta   = getRelativeGroup(warp, cta);

        auto* memory = parallel::SharedMemoryAllocator<NativeType,
            parallel::GroupLevelSize<2>::size()>::allocate();

        size_t start = ctaInKernel.id() * warp.size();

        for(size_t startingRow = start; startingRow < rows;
            startingRow += warp.size() * ctaInKernel.size())
        {
            size_t row = startingRow + warp.id();

            size_t startingColumnPosition = rows * warpInCta.id();
            size_t columnStep = rows * warpInCta.size();
            size_t startingPosition = startingColumnPosition + row;
            size_t position = startingPosition;

            NativeType value = 0;

            if(row < rows && position < inputElements)
            {
                value = rawInput[position];

                for(position += columnStep; position < inputElements; position += columnStep)
                {
                    value = nativeOperation(value, rawInput[position]);
                }
            }

            memory[cta.id()] = value;
            barrier(cta);

            if(warpInCta.id() == 0 && row < rows)
            {
                for(size_t sharedPosition = warp.id() + warp.size(), warpCounter = 1,
                    position = row + rows;
                    warpCounter < warpInCta.size() && position < inputElements;
                    ++warpCounter, position += rows, sharedPosition += warp.size())
                {
                    value = nativeOperation(value, memory[sharedPosition]);
                }

                if (startingPosition < inputElements)
                {
                    rawResult[row] = value;
                }
            }

            barrier(cta);
        }
    }

public:
          NativeType* rawResult;
    const NativeType* rawInput;

public:
    size_t rows;
    size_t inputElements;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename ActualPrecision>
void reduceSecondDimension(Matrix& result, const Matrix& input,
    const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    typedef typename ActualPrecision::type NativeType;

    NativeType*       rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawInput  = static_cast<const NativeType*>(input.data());

    auto lambda = ReduceSecondDimensionLambda<NativeType, ActualOperation>{rawResult, rawInput,
        result.elements(), input.elements(), nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename NativeType, typename ActualOperation>
class GenericReduceLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, 2);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        for(size_t i = relativeGroup.id(); i < elements; i += relativeGroup.size())
        {
            auto resultIndex = linearToDimension(i, resultView.size());

            // find the start of the input slice
            auto inputBegin = selectNamedDimensions(dimensions, resultIndex,
                zeros(inputView.size()));

            // find the end of the input slice
            auto inputEnd = selectNamedDimensions(dimensions,
                resultIndex + ones(resultView.size()), inputView.size());

            auto inputSlice = slice(inputView, inputBegin, inputEnd);

            // find the total size of the slice
            size_t sliceSize = inputSlice.elements();

            // iterate over i linearly from 0 to size
            NativeType resultValue;

            if(innerGroup.size() <= sliceSize)
            {
                auto inputIndex = linearToDimension(innerGroup.id(), inputSlice.size());

                resultValue = inputSlice(inputIndex);

                for(size_t inputLinearIndex = innerGroup.size() + innerGroup.id();
                    inputLinearIndex < sliceSize; inputLinearIndex += innerGroup.size())
                {
                    // get index for i in the input's space
                    auto inputIndex = linearToDimension(inputLinearIndex, inputSlice.size());

                    // apply operator to resultValue, input[index]
                    resultValue = nativeOperation(resultValue, inputSlice(inputIndex));
                }

                resultValue = reduce(innerGroup, resultValue, nativeOperation);
            }
            else if(innerGroup.id() == 0)
            {
                auto inputIndex = linearToDimension(0, inputSlice.size());

                resultValue = inputSlice(inputIndex);

                for(size_t inputLinearIndex = 1; inputLinearIndex < sliceSize; ++inputLinearIndex)
                {
                    // get index for i in the input's space
                    auto inputIndex = linearToDimension(inputLinearIndex, inputSlice.size());

                    // apply operator to resultValue, input[index]
                    resultValue = nativeOperation(resultValue, inputSlice(inputIndex));
                }
            }

            // save the result
            if(innerGroup.id() == 0)
            {
                resultView(resultIndex) = resultValue;
            }
        }

    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    ActualOperation nativeOperation;

public:
    Dimension dimensions;

};

template <typename ActualOperation, typename ActualPrecision>
void reduce(Matrix& result, const Matrix& input, const Dimension& unsortedDimensions,
    const Operation& op, const std::tuple<ActualPrecision>& p)
{
    typedef typename ActualPrecision::type NativeType;

    Dimension dimensions = unsortedDimensions;

    std::sort(dimensions.begin(), dimensions.end());

    assert(ActualPrecision()  == result.precision());
    assert(result.precision() == input.precision());
    assert(result.size()      == removeDimensions(input.size(), dimensions));

    size_t elements = result.elements();

    auto nativeOperation = static_cast<const ActualOperation&>(op);

    // try to simplify the operation to a 2d reduction
    auto reshapedInput  = input;
    auto reshapedResult = result;

    if(elements > 1 && input.isContiguous() && result.isContiguous() &&
        isContiguous(dimensions) &&
        (dimensions.front() == 0 || dimensions.back() == (input.size().size() - 1)))
    {
        size_t reducedElements = selectDimensions(input.size(), dimensions).product();

        auto newInputSize = dimensions.front() == 0 ?
            Dimension(reducedElements, elements) : Dimension(elements, reducedElements);

        reshapedInput  = reshape(input,  newInputSize);
        reshapedResult = reshape(result, {result.elements()} );

        dimensions = dimensions.front() == 0 ? Dimension(0) : Dimension(1);
    }

    // special case reduce down to a single element, and 2d reductions
    if(elements == 1 && reshapedInput.isContiguous() && reshapedResult.isContiguous())
    {
        reduceAllDimensions(reshapedResult, reshapedInput, nativeOperation, ActualPrecision());
    }
    else if(reshapedInput.size().size() == 2 && dimensions.size() == 1 && dimensions[0] == 0
        && reshapedInput.isContiguous() && reshapedResult.isContiguous())
    {
        reduceFirstDimension(reshapedResult, reshapedInput, nativeOperation, ActualPrecision());
    }
    else if(reshapedInput.size().size() == 2 && dimensions.size() == 1 && dimensions[0] == 1
        && reshapedInput.isContiguous() && reshapedResult.isContiguous())
    {
        reduceSecondDimension(reshapedResult, reshapedInput, nativeOperation, ActualPrecision());
    }
    else
    {
        // handle other types of reductions (note that this is much slower)
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> inputView(input);

        auto lambda = GenericReduceLambda<NativeType, ActualOperation>{resultView,
            inputView, elements, nativeOperation, dimensions};

        parallel::multiBulkSynchronousParallel(lambda);
    }

}

template <typename ActualOperation, typename PossiblePrecisions>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(result.precision() == PossiblePrecisionType())
    {
        reduce<ActualOperation>(result, input, dimensions, op, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        reduce<ActualOperation>(result, input, dimensions, op, RemainingPrecisions());
    }
}

template <typename PossibleOperation>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op,
    const std::tuple<PossibleOperation>& p)
{
    assert(PossibleOperation() == op);
    reduce<PossibleOperation>(result, input, dimensions, op, AllPrecisions());
}

template <typename PossibleOperations>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op,
    const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;
    if(op == PossibleOperationType())
    {
        reduce(result, input, dimensions, op, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        reduce(result, input, dimensions, op, RemainingOperations());
    }

}

}

void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op)
{
    detail::reduce(result, input, dimensions, op, AllBinaryOperations());
}

Matrix reduce(const Matrix& input, const Dimension& dimensions, const Operation& op)
{
    Matrix result(removeDimensions(input.size(), dimensions), input.precision());

    reduce(result, input, dimensions, op);

    return result;
}

namespace detail
{

template <typename NativeType, typename ActualOperation>
class BroadcastFirstOfTwoDimensionsLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        if(rows > columns)
        {
            for(size_t row = threadGroup.id(); row < rows; row += threadGroup.size())
            {
                size_t offset = row;

                for(size_t column = 0; column < columns; ++column, offset += rows)
                {
                    rawResult[offset] = nativeOperation(rawLeft[offset], rawRight[row]);
                }
            }
        }
        else
        {
            for(size_t column = threadGroup.id(); column < columns; column += threadGroup.size())
            {
                size_t offset = column * rows;

                for(size_t row = 0; row < rows; ++row, ++offset)
                {
                    rawResult[offset] = nativeOperation(rawLeft[offset], rawRight[row]);
                }
            }
        }
    }

public:
    NativeType*       rawResult;
    const NativeType* rawLeft;
    const NativeType* rawRight;

public:
    size_t rows;
    size_t columns;

public:
    ActualOperation nativeOperation;

};

template <typename NativeType, typename ActualOperation>
class BroadcastFirstOfTwoDimensionsWarpStrideLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, 1);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        int globalOffset = relativeGroup.id() * rows;

        for(int column = relativeGroup.id(); column < columns;
            column += relativeGroup.size(), globalOffset += rows * relativeGroup.size())
        {
            int offset = globalOffset + innerGroup.id();

            for(int row = innerGroup.id(); row < rows; row += innerGroup.size(),
                offset += innerGroup.size())
            {
                //std::printf("thread (%d, %d) handling (%d, %d)\n", threadGroup.id(), threadGroup.size(), row, column);
                rawResult[offset] = nativeOperation(rawLeft[offset], rawRight[row]);
            }
        }
    }

public:
    NativeType*       rawResult;
    const NativeType* rawLeft;
    const NativeType* rawRight;

public:
    int rows;
    int columns;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename NativeType>
void broadcastFirstOfTwoDimensions(Matrix& result, const Matrix& left, const Matrix& right,
    const ActualOperation& op)
{
          NativeType* rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawLeft   = static_cast<const NativeType*>(left.data());
    const NativeType* rawRight  = static_cast<const NativeType*>(right.data());

    size_t rows    = left.size()[0];
    size_t columns = left.size()[1];

    if(rows < columns && rows >= 32 &&
        (rows <= std::numeric_limits<int>::max() && columns <= std::numeric_limits<int>::max()))
    {
        auto lambda = BroadcastFirstOfTwoDimensionsWarpStrideLambda<NativeType, ActualOperation>{
            rawResult, rawLeft, rawRight, static_cast<int>(rows), static_cast<int>(columns), op};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        auto lambda = BroadcastFirstOfTwoDimensionsLambda<NativeType, ActualOperation>{rawResult,
            rawLeft, rawRight, rows, columns, op};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template <typename NativeType, typename ActualOperation>
class BroadcastSecondOfTwoDimensionsLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        if(rows > columns)
        {
            for(size_t row = threadGroup.id(); row < rows; row += threadGroup.size())
            {
                size_t offset = row;

                for(size_t column = 0; column < columns; ++column, offset += rows)
                {
                    rawResult[offset] = nativeOperation(rawLeft[offset], rawRight[column]);
                }
            }
        }
        else
        {
            for(size_t column = threadGroup.id(); column < columns; column += threadGroup.size())
            {
                size_t offset = column * rows;

                for(size_t row = 0; row < rows; ++row, ++offset)
                {
                    rawResult[offset] = nativeOperation(rawLeft[offset], rawRight[column]);
                }
            }
        }
    }

public:
    NativeType*       rawResult;
    const NativeType* rawLeft;
    const NativeType* rawRight;

public:
    size_t rows;
    size_t columns;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename NativeType>
void broadcastSecondOfTwoDimensions(Matrix& result, const Matrix& left, const Matrix& right,
    const ActualOperation& op)
{
          NativeType* rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawLeft   = static_cast<const NativeType*>(left.data());
    const NativeType* rawRight  = static_cast<const NativeType*>(right.data());

    auto lambda = BroadcastSecondOfTwoDimensionsLambda<NativeType, ActualOperation>{rawResult,
        rawLeft, rawRight, left.size()[0], left.size()[1], op};

    parallel::multiBulkSynchronousParallel(lambda);
}

static Dimension fillOutDimension(const Dimension& d, const Dimension& leftSize,
    const Dimension& rightSize)
{
    if(d.size() != 0)
    {
        return d;
    }

    Dimension retVal;
    for(auto i = leftSize.begin(), j = rightSize.begin(); i != leftSize.end(); ++i)
    {
        if((j != rightSize.end()) && (*i == *j))
        {
            ++j;
            continue;
        }

        retVal.push_back(std::distance(leftSize.begin(), i));
    }

    return retVal;
}

static Dimension invert(const Dimension& original, const Dimension& removed)
{
    Dimension result;

    for(size_t i = 0; i < original.size(); ++i)
    {
        if(!isContained(removed, i))
        {
            result.push_back(i);
        }
    }

    return result;
}

template <typename NativeType, typename ActualOperation>
class BroadcastLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto fullDimension    = linearToDimension(i, resultView.size());
            auto reducedDimension = selectDimensions(fullDimension, dimension);

            resultView(fullDimension) = nativeOperation(leftView(fullDimension),
                rightView(reducedDimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> leftView;
    ConstMatrixView<NativeType> rightView;

public:
    size_t elements;

public:
    ActualOperation nativeOperation;

public:
    Dimension dimension;

};

static bool isNotInnerDimension(const Dimension& size, const Dimension& dimension)
{
    assert(dimension.front() < size.size());

    bool leftOkay = true;

    // everything preceding it is one
    for(size_t i = 0; i < dimension.front(); ++i)
    {
        if(size[i] != 1)
        {
            leftOkay = false;
            break;
        }
    }

    if(leftOkay)
    {
        return true;
    }

    // everything after it is one
    for(size_t i = dimension.front() + 1; i < size.size(); ++i)
    {
        if(size[i] != 1)
        {
            return false;
        }
    }

    return true;
}

template <typename ActualOperation, typename ActualPrecision>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op, const std::tuple<ActualPrecision>& p)
{
    auto dimension = invert(left.size(), fillOutDimension(d, left.size(), right.size()));
    typedef typename ActualPrecision::type NativeType;

    assert(ActualPrecision()  == result.precision());
    assert(result.precision() == left.precision());
    assert(result.precision() == right.precision());
    assert(result.size()      == left.size());

    auto nativeOperation = static_cast<const ActualOperation&>(op);

    // try to simplify the operation to a 2d broadcast
    auto reshapedLeft   = left;
    auto reshapedRight  = right;
    auto reshapedResult = result;

    if(left.isContiguous() && result.isContiguous() && right.isContiguous() &&
        dimension.size() == 1 && isNotInnerDimension(result.size(), dimension))
    {
        Dimension leftDimension({1, 1});
        bool isLeft = true;

        for(size_t i = 0; i < dimension.front(); ++i)
        {
            if(left.size()[i] != 1)
            {
                isLeft = false;
                break;
            }
        }

        if(isLeft)
        {
            for(size_t i = 0; i <= dimension.front(); ++i)
            {
                leftDimension[0] *= left.size()[i];
            }

            leftDimension[1] = left.size().product() / leftDimension[0];

            reshapedRight = reshape(right, leftDimension[0]);

            dimension.front() = 0;
        }
        else
        {
            for(size_t i = dimension.front(); i < left.size().size(); ++i)
            {
                leftDimension[1] *= left.size()[i];
            }

            leftDimension[0] = left.size().product() / leftDimension[1];

            reshapedRight = reshape(right, leftDimension[1]);

            dimension.front() = 1;
        }

        reshapedLeft   = reshape(left,   leftDimension);
        reshapedResult = reshape(result, leftDimension);

        if(isLeft)
        {
            broadcastFirstOfTwoDimensions<ActualOperation, NativeType>(reshapedResult,
                reshapedLeft, reshapedRight, nativeOperation);
        }
        else
        {
            broadcastSecondOfTwoDimensions<ActualOperation, NativeType>(reshapedResult,
                reshapedLeft, reshapedRight, nativeOperation);
        }
    }
    else
    {
        size_t elements = reshapedResult.elements();

        MatrixView<NativeType>      resultView(reshapedResult);
        ConstMatrixView<NativeType> leftView(reshapedLeft);
        ConstMatrixView<NativeType> rightView(reshapedRight);

        auto lambda = BroadcastLambda<NativeType, ActualOperation>{resultView, leftView,
            rightView, elements, nativeOperation, dimension};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template <typename ActualOperation, typename PossiblePrecisions>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(result.precision() == PossiblePrecisionType())
    {
        broadcast<ActualOperation>(result, left, right, d, op, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        broadcast<ActualOperation>(result, left, right, d, op, RemainingPrecisions());
    }
}

template <typename PossibleOperation>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op, const std::tuple<PossibleOperation>& p)
{
    assert(PossibleOperation() == op);
    broadcast<PossibleOperation>(result, left, right, d, op, AllPrecisions());
}

template <typename PossibleOperations>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d,
    const Operation& op, const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;
    if(op == PossibleOperationType())
    {
        broadcast(result, left, right, d, op, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        broadcast(result, left, right, d, op, RemainingOperations());
    }

}

}

void broadcast(Matrix& result, const Matrix& left, const Matrix& right,
    const Dimension& d, const Operation& op)
{
    detail::broadcast(result, left, right, d, op, AllBinaryOperations());
}

Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op)
{
    Matrix retVal(left.size(), left.precision());
    broadcast(retVal, left, right, d, op);
    return retVal;
}

void zeros(const DynamicView& result)
{
	apply(result, result, Fill(0.0));
}

void zeros(Matrix& result)
{
    zeros(DynamicView(result));
}

Matrix zeros(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    zeros(result);

    return result;
}

void ones(Matrix& result)
{
	apply(result, result, Fill(1.0));
}

Matrix ones(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    ones(result);

    return result;
}

}
}






