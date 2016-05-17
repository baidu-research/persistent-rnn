/*  \file   CublasLibrary.cpp
    \brief  The source file for the CublasLibrary class.
*/

// Persistent RNN Includes
#include <prnn/detail/matrix/cublas_library.h>

#include <prnn/detail/parallel/cuda.h>
#include <prnn/detail/parallel/assert.h>

#include <prnn/detail/util/casts.h>
#include <prnn/detail/util/logger.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace prnn
{

namespace matrix
{

void CublasLibrary::load()
{
    _interface.load();
}

bool CublasLibrary::loaded()
{
    load();

    return _interface.loaded();
}

void CublasLibrary::cublasSgeam(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, const float *alpha, const float *A,
    int lda, const float *beta, const float *B, int ldb,
    float *C, int ldc)
{
    _check();

    util::log("CublasLibrary") << " CUBLAS SGEAM: ("
        "handle: " << _interface.handle <<  ", "
        "transa: " << transa <<  ", "
        "transb: " << transb <<  ", "

        "m: " << m <<  ", "
        "n: " << n <<  ", "

        "alpha: " << alpha <<  " (" << *alpha << "), "
        "A: " << A <<  ", "
        "lda: " << lda <<  ", "

        "beta: " << beta <<  " (" << *beta << "), "
        "B: " << B <<  ", "
        "ldb: " << ldb <<  ", "

        "C: " << C <<  ", "
        "ldc: " << ldc << ")\n";

    cublasStatus_t status = (*_interface.cublasSgeam)(_interface.handle,
        transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda SGEAM failed: " +
            cublasGetErrorString(status));
    }

}

void CublasLibrary::cublasSgemm(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float *A, int lda,
    const float *B, int ldb, const float* beta, float *C, int ldc)
{
    _check();

    util::log("CublasLibrary") << " CUBLAS SGEMM: ("
        "handle: " << _interface.handle <<  ", "
        "transa: " << transa <<  ", "
        "transb: " << transb <<  ", "

        "m: " << m <<  ", "
        "n: " << n <<  ", "
        "k: " << k <<  ", "

        "alpha: " << alpha <<  " (" << *alpha << "), "
        "A: " << A <<  ", "
        "lda: " << lda <<  ", "

        "B: " << B <<  ", "
        "ldb: " << ldb <<  ", "
        "beta: " << beta <<  " (" << *beta << "), "

        "C: " << C <<  ", "
        "ldc: " << ldc << ")\n";

    cublasStatus_t status = (*_interface.cublasSgemm_v2)(_interface.handle,
        transa, transb, m, n, k, alpha, A, lda, B, ldb,
        beta, C, ldc);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda SGEMM failed: " +
            cublasGetErrorString(status));
    }
}

void CublasLibrary::cublasDgemm(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc)
{
    _check();

    util::log("CublasLibrary") << " CUBLAS DGEMM: ("
        "handle: " << _interface.handle <<  ", "
        "transa: " << transa <<  ", "
        "transb: " << transb <<  ", "

        "m: " << m <<  ", "
        "n: " << n <<  ", "
        "k: " << k <<  ", "

        "alpha: " << alpha <<  " (" << *alpha << "), "
        "A: " << A <<  ", "
        "lda: " << lda <<  ", "

        "B: " << B <<  ", "
        "ldb: " << ldb <<  ", "
        "beta: " << beta <<  " (" << *beta << "), "

        "C: " << C <<  ", "
        "ldc: " << ldc << ")\n";

    cublasStatus_t status = (*_interface.cublasDgemm_v2)(_interface.handle,
        transa, transb, m, n, k, alpha, A, lda, B, ldb,
        beta, C, ldc);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda DGEMM failed: " +
            cublasGetErrorString(status));
    }
}

void CublasLibrary::cublasSgemmBatched(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float *A, int lda,
    const float *B, int ldb, const float* beta, float *C, int ldc,
    int batch)
{
    _check();

    util::log("CublasLibrary") << " CUBLAS SGEMM BATCH: ("
        "handle: " << _interface.handle <<  ", "
        "transa: " << transa <<  ", "
        "transb: " << transb <<  ", "

        "m: " << m <<  ", "
        "n: " << n <<  ", "
        "k: " << k <<  ", "

        "alpha: " << alpha <<  " (" << *alpha << "), "
        "A: " << A <<  ", "
        "lda: " << lda <<  ", "

        "B: " << B <<  ", "
        "ldb: " << ldb <<  ", "
        "beta: " << beta <<  " (" << *beta << "), "

        "C: " << C <<  ", "
        "ldc: " << ldc << ", "
        "batch: " << batch << ")\n";

    cublasStatus_t status = (*_interface.cublasSgemmBatched)(_interface.handle,
        transa, transb, m, n, k, alpha, A, lda, B, ldb,
        beta, C, ldc, batch);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda SGEMM failed: " +
            cublasGetErrorString(status));
    }
}

std::string CublasLibrary::cublasGetErrorString(cublasStatus_t error)
{
    switch(error)
    {
    case CUBLAS_STATUS_SUCCESS:
    {
        return "success";
        break;
    }
    case CUBLAS_STATUS_NOT_INITIALIZED:
    {
        return "not initialized";
        break;
    }
    case CUBLAS_STATUS_ALLOC_FAILED:
    {
        return "allocation failed";
        break;
    }
    case CUBLAS_STATUS_INVALID_VALUE:
    {
        return "invalid value";
        break;
    }
    case CUBLAS_STATUS_ARCH_MISMATCH:
    {
        return "arch mismatch";
        break;
    }
    case CUBLAS_STATUS_MAPPING_ERROR:
    {
        return "mapping error";
        break;
    }
    case CUBLAS_STATUS_EXECUTION_FAILED:
    {
        return "execution failed";
        break;
    }
    case CUBLAS_STATUS_INTERNAL_ERROR:
    {
        return "internal error";
        break;
    }
    }

    return "Unknown error.";
}

void CublasLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call CUBLAS function when "
            "the library is not loaded. Loading library failed, consider "
            "installing CUBLAS.");
    }
}

CublasLibrary::Interface::Interface()
: handle(nullptr), _library(nullptr), _failed(false)
{

}

CublasLibrary::Interface::~Interface()
{
    unload();
}

static void checkFunction(void* pointer, const std::string& name)
{
    if(pointer == nullptr)
    {
        throw std::runtime_error("Failed to load function '" + name +
            "' from dynamic library.");
    }
}

void CublasLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    const char* libraryName = "libcublas.dylib";
    #else
    const char* libraryName = "libcublas.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CublasLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("CublasLibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {

        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(cublasSgeam);
        DynLink(cublasSgemm_v2);
        DynLink(cublasDgemm_v2);
        DynLink(cublasSgemmBatched);

        DynLink(cublasCreate_v2);
        DynLink(cublasDestroy_v2);

        #undef DynLink

        util::log("CublasLibrary") << " Loaded library '" << libraryName
            << "' successfully, creating handle...\n";

        _createHandle();

        util::log("CublasLibrary") << "  success....\n";
    }
    catch(...)
    {
        util::log("CublasLibrary") << "  failed....\n";
        unload();
        throw;
    }
}

bool CublasLibrary::Interface::loaded() const
{
    return _library != nullptr;
}

void CublasLibrary::Interface::unload()
{
    if(!loaded()) return;

    //_destroyHandle();

    dlclose(_library);
    _library = nullptr;
}

void CublasLibrary::Interface::_createHandle()
{
    cublasStatus_t status = (*cublasCreate_v2)(&handle);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create cublas handle with error '"
            + cublasGetErrorString(status) + "'.");
    }
}

void CublasLibrary::Interface::_destroyHandle()
{
    cublasStatus_t status = (*cublasDestroy_v2)(handle);

    assert(status == CUBLAS_STATUS_SUCCESS);
}

CublasLibrary::Interface CublasLibrary::_interface;

}

}


