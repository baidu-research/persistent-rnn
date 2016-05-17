/*  \file  curand_library.h
    \brief The header file for the CurandLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstring>
#include <string>

namespace prnn
{

namespace matrix
{

class CurandLibrary
{
public:
    enum curandStatus {
        CURAND_STATUS_SUCCESS = 0, ///< No errors
        CURAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
        CURAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not initialized
        CURAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
        CURAND_STATUS_TYPE_ERROR = 103, ///< Generator is wrong type
        CURAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
        CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Length requested is not a multple of dimension
        CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision required by MRG32k3a
        CURAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
        CURAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
        CURAND_STATUS_INITIALIZATION_FAILED = 203, ///< Initialization of CUDA failed
        CURAND_STATUS_ARCH_MISMATCH = 204, ///< Architecture mismatch, GPU does not support requested feature
        CURAND_STATUS_INTERNAL_ERROR = 999 ///< Internal library error
    };

    enum curandRngType {
        CURAND_RNG_TEST = 0,
        CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
        CURAND_RNG_PSEUDO_XORWOW = 101, ///< XORWOW pseudorandom generator
        CURAND_RNG_PSEUDO_MRG32K3A = 121, ///< MRG32k3a pseudorandom generator
        CURAND_RNG_PSEUDO_MTGP32 = 141, ///< Mersenne Twister pseudorandom generator
        CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161, ///< Default pseudorandom generator
        CURAND_RNG_QUASI_DEFAULT = 200, ///< Default quasirandom generator
        CURAND_RNG_QUASI_SOBOL32 = 201, ///< Sobol32 quasirandom generator
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  ///< Scrambled Sobol32 quasirandom generator
        CURAND_RNG_QUASI_SOBOL64 = 203, ///< Sobol64 quasirandom generator
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  ///< Scrambled Sobol64 quasirandom generator
    };

    typedef curandRngType curandRngType_t;
    typedef curandStatus  curandStatus_t;

    struct curandGenerator_st;
    typedef struct curandGenerator_st* curandGenerator_t;

public:
    static void load();
    static bool loaded();

public:
    static void curandCreateGenerator(curandGenerator_t *generator,
        curandRngType_t rng_type);
    static void curandDestroyGenerator(
        curandGenerator_t generator);

    static void curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
        unsigned long long seed);

    static void curandGenerateUniform(curandGenerator_t generator,
        float *outputPtr, size_t num);
    static void curandGenerateNormal(curandGenerator_t generator,
        float *outputPtr, size_t num, float mean, float stddev);

    static void curandGenerateUniformDouble(curandGenerator_t generator,
        double *outputPtr, size_t num);
    static void curandGenerateNormalDouble(curandGenerator_t generator,
        double *outputPtr, size_t num, double mean, double stddev);

public:
    static std::string curandGetErrorString(curandStatus error);

private:
    static void _check();

private:
    class Interface
    {
    public:
        curandStatus_t (*curandCreateGenerator)(curandGenerator_t *generator,
            curandRngType_t rng_type);
        curandStatus_t (*curandDestroyGenerator)(curandGenerator_t generator);

        curandStatus_t (*curandSetPseudoRandomGeneratorSeed)(curandGenerator_t generator,
            unsigned long long seet);

        curandStatus_t (*curandGenerateUniform)(curandGenerator_t generator,
            float *outputPtr, size_t num);
        curandStatus_t (*curandGenerateNormal)(curandGenerator_t generator,
            float *outputPtr, size_t num, float mean, float stddev);

        curandStatus_t (*curandGenerateUniformDouble)(curandGenerator_t generator,
            double *outputPtr, size_t num);
        curandStatus_t (*curandGenerateNormalDouble)(curandGenerator_t generator,
            double *outputPtr, size_t num, double mean, double stddev);

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



