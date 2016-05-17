/*  \file   cublas_library.h
    \brief  The header file for the CublasLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstring>
#include <string>

namespace prnn
{

namespace matrix
{

class CublasLibrary
{
public:
    static const int CUBLAS_FILL_MODE_LOWER = 0;
    static const int CUBLAS_FILL_MODE_UPPER = 1;

    static const int CUBLAS_DIAG_NON_UNIT = 0;
    static const int CUBLAS_DIAG_UNIT     = 1;

    static const int CUBLAS_SIDE_LEFT  = 0;
    static const int CUBLAS_SIDE_RIGHT = 1;

    static const int CUDA_SUCCESS = 0;

    //static const int CUBLAS_OP_N = 0;
    //static const int CUBLAS_OP_T = 1;
    //static const int CUBLAS_OP_C = 2;

    static const int CUBLAS_POINTER_MODE_HOST   = 0;
    static const int CUBLAS_POINTER_MODE_DEVICE = 1;

    typedef enum{
        CUBLAS_STATUS_SUCCESS          = 0,
        CUBLAS_STATUS_NOT_INITIALIZED  = 1,
        CUBLAS_STATUS_ALLOC_FAILED     = 3,
        CUBLAS_STATUS_INVALID_VALUE    = 7,
        CUBLAS_STATUS_ARCH_MISMATCH    = 8,
        CUBLAS_STATUS_MAPPING_ERROR    = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR   = 14
    } cublasStatus_t;

    typedef enum {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2
    } cublasOperation_t;

public:
    static void load();
    static bool loaded();

public:
    static void    cublasSgeam(cublasOperation_t transa,
        cublasOperation_t transb, int m, int n,
        const float *alpha, const float *A, int lda,
        const float *beta, const float *B, int ldb, float *C,
        int ldc);

    static void cublasSgemm(cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const float* alpha, const float* A, int lda,
        const float* B, int ldb, const float* beta, float* C,
        int ldc);

    static void cublasDgemm(cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const double* alpha, const double* A, int lda,
        const double* B, int ldb, const double* beta, double* C,
        int ldc);

    static void cublasSgemmBatched(cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k,
        const float* alpha, const float *A, int lda,
        const float *B, int ldb, const float* beta, float *C,
        int ldc, int batches);

public:
    static std::string cublasGetErrorString(cublasStatus_t error);

private:
    static void _check();

private:
    class Interface
    {
    public:
                struct cublasContext;
        typedef struct cublasContext* cublasHandle_t;

    public:
        cublasStatus_t (*cublasSgeam)(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, const float *alpha, const float *A,
                          int lda, const float *beta, const float *B, int ldb,
                          float *C, int ldc);

        cublasStatus_t (*cublasSgemm_v2) (cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const float* alpha, const float* A, int lda,
            const float* B, int ldb, const float* beta, float* C, int ldc);

        cublasStatus_t (*cublasDgemm_v2) (cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const double* alpha, const double* A, int lda,
            const double* B, int ldb, const double* beta, double* C, int ldc);

        cublasStatus_t (*cublasSgemmBatched) (cublasHandle_t handle,
            cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const float* alpha, const float* A, int lda,
            const float* B, int ldb, const float* beta, float* C, int ldc, int batch);

    public:
        cublasStatus_t (*cublasCreate_v2)  (cublasHandle_t* handle);
        cublasStatus_t (*cublasDestroy_v2) (cublasHandle_t  handle);

    public:
        cublasHandle_t handle;

    public:
        /*! \brief The constructor zeros out all of the pointers */
        Interface();

        /*! \brief The destructor closes dlls */
        ~Interface();
        /*! \brief Load the library */
        void load();
        /*! \brief Has the library been loaded? */
        bool loaded() const;
        /*! \brief unloads the library */
        void unload();

    private:
        void _createHandle();
        void _destroyHandle();

    private:
        void* _library;
        bool  _failed;

    };

private:
    static Interface _interface;

};

}

}


