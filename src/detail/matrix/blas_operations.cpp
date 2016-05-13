
// Persistent RNN Includes
#include <prnn/detail/matrix/blas_operations.h>
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/atlas_library.h>
#include <prnn/detail/matrix/cublas_library.h>

#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/debug.h>

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
    assert(left.size().size() == right.size().size());
    assert(left.size().size() == result.size().size());
    assert(left.size().size() == 2);

    assert(left.precision() == right.precision());
    assert(left.precision() == result.precision());

    assert(left.isLeadingDimensionContiguous());
    assert(right.isLeadingDimensionContiguous());
    assert(result.isLeadingDimensionContiguous());

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

            CublasLibrary::cublasSgemm(transposeLeft ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                transposeRight ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                m, n, k, &alphaCopy,
                static_cast<const float*>(left.data()),   lda,
                static_cast<const float*>(right.data()),  ldb, &betaCopy,
                static_cast<      float*>(result.data()), ldc);
        }
        else if(left.precision() == DoublePrecision())
        {
            CublasLibrary::cublasDgemm(transposeLeft ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                transposeRight ? CublasLibrary::CUBLAS_OP_T : CublasLibrary::CUBLAS_OP_N,
                m, n, k, &alpha,
                static_cast<const double*>(left.data()),   lda,
                static_cast<const double*>(right.data()),  ldb, &beta,
                static_cast<      double*>(result.data()), ldc);
        }
        else
        {
            assertM(false, "Precision not implemented.");
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
                static_cast<const float*>(left.data()),   lda,
                static_cast<const float*>(right.data()),  ldb, beta,
                static_cast<      float*>(result.data()), ldc);
        }
        else if(left.precision() == DoublePrecision())
        {
            AtlasLibrary::dgemm(AtlasLibrary::CblasColMajor,
                transposeLeft  ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                transposeRight ? AtlasLibrary::CblasTrans : AtlasLibrary::CblasNoTrans,
                m, n, k, alpha,
                static_cast<const double*>(left.data()),   lda,
                static_cast<const double*>(right.data()),  ldb, beta,
                static_cast<      double*>(result.data()), ldc);
        }
        else
        {
            assertM(false, "Precision not implemented.");
        }
    }
    else
    {
        assertM(false, "Fallback GEMM not implemented.");
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



