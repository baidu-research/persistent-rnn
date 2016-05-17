/*    \file   AtlasLibrary.h
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the AtlasLibrary class.
*/

#pragma once

// Forward Declarations

namespace prnn
{

namespace matrix
{

class AtlasLibrary
{
public:
    static const int CblasRowMajor = 101;
    static const int CblasColMajor = 102;

    static const int CblasNoTrans   = 111;
    static const int CblasTrans     = 112;
    static const int CblasConjTrans = 113;
    static const int AtlasConj      = 114;

    static const int CblasUpper = 121;
    static const int CblasLower = 122;

    static const int CblasNonUnit = 131;
    static const int CblasUnit    = 132;

    static const int CblasLeft  = 141;
    static const int CblasRight = 142;

public:
    static void load();
    static bool loaded();

public:
    static void sgemm(const int Order, const int TransA,
         const int TransB, const int M, const int N,
         const int K, const float alpha, const float *A,
         const int lda, const float *B, const int ldb,
         const float beta, float *C, const int ldc);

    static void dgemm(const int Order, const int TransA,
         const int TransB, const int M, const int N,
         const int K, const double alpha, const double* A,
         const int lda, const double* B, const int ldb,
         const double beta, double* C, const int ldc);

private:
    static void _check();

private:
    class Interface
    {
    public:
        void (*cblas_sgemm)(const int Order, const int TransA,
             const int TransB, const int M, const int N,
             const int K, const float alpha, const float *A,
             const int lda, const float *B, const int ldb,
             const float beta, float *C, const int ldc);

        void (*cblas_dgemm)(const int Order, const int TransA,
             const int TransB, const int M, const int N,
             const int K, const double alpha, const double *A,
             const int lda, const double *B, const int ldb,
             const double beta, double *C, const int ldc);

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
        void* _library;
        bool  _failed;
    };

private:
    static Interface _interface;

};

}

}


