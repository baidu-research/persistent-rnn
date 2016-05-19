
// Persistent RNN Includes
#include <prnn/detail/matrix/blas_operations.h>
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_view.h>
#include <prnn/detail/matrix/atlas_library.h>
#include <prnn/detail/matrix/cublas_library.h>

#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/logger.h>

namespace prnn
{
namespace matrix
{

void gemm(Matrix& result, const Matrix& left, const Matrix& right)
{
    gemm(result, 0.0, left, false, 1.0, right, false);
}

Matrix gemm(const Matrix& left, const Matrix& right)
{
    return gemm(left, false, 1.0, right, false);
}

void gemm(Matrix& result, const Matrix& left, bool transposeLeft, const Matrix& right,
    bool transposeRight)
{
    gemm(result, 0.0, left, transposeLeft, 1.0, right, transposeRight);
}

Matrix gemm(const Matrix& left, bool transposeLeft, const Matrix& right, bool transposeRight)
{
    return gemm(left, transposeLeft, 1.0, right, transposeRight);
}

void gemm(Matrix& result, double beta,
    const Matrix& left,  bool transposeLeft, double alpha,
    const Matrix& right, bool transposeRight)
{
    assert(left.isLeadingDimensionContiguous());
    assert(right.isLeadingDimensionContiguous());
    assert(result.isLeadingDimensionContiguous());

    gemm(DynamicView(result), beta, ConstDynamicView(left), transposeLeft, alpha,
        ConstDynamicView(right), transposeRight);
}

void gemm(const DynamicView& result, double beta,
    const ConstDynamicView& left,  bool transposeLeft, double alpha,
    const ConstDynamicView& right, bool transposeRight)
{
    assert(left.size().size() == right.size().size());
    assert(left.size().size() == result.size().size());
    assert(left.size().size() == 2);

    assert(left.precision() == right.precision());
    assert(left.precision() == result.precision());

    size_t m = transposeLeft  ? left.size()[1]  : left.size()[0];
    size_t n = transposeRight ? right.size()[0] : right.size()[1];
    size_t k = transposeLeft  ? left.size()[0]  : left.size()[1];

    assert(k == (transposeRight ? right.size()[1] : right.size()[0]));

    size_t lda = left.stride()[1];
    size_t ldb = right.stride()[1];
    size_t ldc = result.stride()[1];

    if(CublasLibrary::loaded())
    {
        parallel::setNotSynchronized();

        if(left.precision() == SinglePrecision())
        {
            float alphaCopy = alpha;
            float betaCopy  = beta;
            CublasLibrary::cublasSgemm(transposeLeft ?
                CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                transposeRight ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                m, n, k, &alphaCopy,
                left.data<float>(),   lda,
                right.data<float>(),  ldb, &betaCopy,
                result.data<float>(), ldc);
        }
        else if(left.precision() == DoublePrecision())
        {
            CublasLibrary::cublasDgemm(transposeLeft ?
                CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                transposeRight ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                m, n, k, &alpha,
                left.data<double>(),   lda,
                right.data<double>(),  ldb, &beta,
                result.data<double>(), ldc);
        }
        else
        {
            throw std::runtime_error("Precision not implemented.");
        }
    }
    else if(AtlasLibrary::loaded())
    {
        if(left.precision() == SinglePrecision())
        {
            AtlasLibrary::sgemm(AtlasLibrary::CblasColMajor,
                transposeLeft  ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                transposeRight ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                m, n, k, alpha,
                left.data<float>(),   lda,
                right.data<float>(),  ldb, beta,
                result.data<float>(), ldc);
        }
        else if(left.precision() == DoublePrecision())
        {
            AtlasLibrary::dgemm(AtlasLibrary::CblasColMajor,
                transposeLeft  ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                transposeRight ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                m, n, k, alpha,
                left.data<double>(),   lda,
                right.data<double>(),  ldb, beta,
                result.data<double>(), ldc);
        }
        else
        {
            throw std::runtime_error("Precision not implemented.");
        }
    }
    else
    {
        throw std::runtime_error("Fallback GEMM not implemented.");
    }
}

Matrix gemm(const Matrix& left, bool transposeLeft, double alpha,
    const Matrix& right, bool transposeRight)
{
    size_t rows    = transposeLeft  ? left.size()[1]  : left.size()[0];
    size_t columns = transposeRight ? right.size()[0] : right.size()[1];

    Matrix result({rows, columns}, left.precision());

    gemm(result, 0.0, left, transposeLeft, alpha, right, transposeRight);

    return result;
}

}
}



