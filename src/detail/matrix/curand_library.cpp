/*  \file   CurandLibrary.cpp
    \brief  The source file for the CurandLibrary class.
*/

// Persistent RNN Includes
#include <prnn/detail/matrix/curand_library.h>

#include <prnn/detail/parallel/cuda.h>

#include <prnn/detail/util/casts.h>
#include <prnn/detail/util/logger.h>

// System-Specific Includes
#include <dlfcn.h>
#include <stdexcept>

namespace prnn
{

namespace matrix
{

void CurandLibrary::load()
{
    _interface.load();
}

bool CurandLibrary::loaded()
{
    load();

    return _interface.loaded();
}

void CurandLibrary::curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
    _check();

    curandStatus_t status = (*_interface.curandCreateGenerator)(generator, rng_type);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda destroy generator failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandDestroyGenerator(curandGenerator_t generator)
{
    _check();

    curandStatus_t status = (*_interface.curandDestroyGenerator)(generator);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda destroy generator failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
    _check();

    curandStatus_t status = (*_interface.curandSetPseudoRandomGeneratorSeed)(generator, seed);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda seed generator failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num)
{
    _check();

    curandStatus_t status = (*_interface.curandGenerateUniform)(generator, outputPtr, num);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda generate uniform failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandGenerateNormal(curandGenerator_t generator, float* outputPtr, size_t num, float mean, float stddev)
{
    _check();

    curandStatus_t status = (*_interface.curandGenerateNormal)(generator, outputPtr, num, mean, stddev);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda generate normal failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandGenerateUniformDouble(curandGenerator_t generator, double* outputPtr, size_t num)
{
    _check();

    curandStatus_t status = (*_interface.curandGenerateUniformDouble)(generator, outputPtr, num);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda generate uniform double failed: " +
            curandGetErrorString(status));
    }
}

void CurandLibrary::curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr, size_t num, double mean, double stddev)
{
    _check();

    curandStatus_t status = (*_interface.curandGenerateNormalDouble)(generator, outputPtr, num, mean, stddev);

    if(status != CURAND_STATUS_SUCCESS)
    {
        throw std::runtime_error("Cuda generate normal double failed: " +
            curandGetErrorString(status));
    }
}


std::string CurandLibrary::curandGetErrorString(curandStatus error)
{
    switch(error)
    {
    case CURAND_STATUS_SUCCESS:
    {
        return "No errors";
        break;
    }
    case CURAND_STATUS_VERSION_MISMATCH:
    {
        return "Header file and linked library version do not match";
        break;
    }
    case CURAND_STATUS_NOT_INITIALIZED:
    {
        return "Generator not initialized";
        break;
    }
    case CURAND_STATUS_ALLOCATION_FAILED:
    {
        return "Memory allocation failed";
        break;
    }
    case CURAND_STATUS_TYPE_ERROR:
    {
        return "Generator is wrong type";
        break;
    }
    case CURAND_STATUS_OUT_OF_RANGE:
    {
        return "Argument out of range";
        break;
    }
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    {
        return "Length requtested is not a multiple of dimension";
        break;
    }
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    {
        return "GPU does not have double precision required MRG32k3a";
        break;
    }
    case CURAND_STATUS_LAUNCH_FAILURE:
    {
        return "Kernel launch failure";
        break;
    }
    case CURAND_STATUS_PREEXISTING_FAILURE:
    {
        return "Preexisting failure on library entry";
        break;
    }
    case CURAND_STATUS_INITIALIZATION_FAILED:
    {
        return "Initialization of cuda failed";
        break;
    }
    case CURAND_STATUS_ARCH_MISMATCH:
    {
        return "Architecture mismatch, GPU does not support requested feature";
        break;
    }
    case CURAND_STATUS_INTERNAL_ERROR:
    {
        return "Internal library error";
        break;
    }
    }

    return "Unknown error.";
}

void CurandLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call CURAND function when "
            "the library is not loaded. Loading library failed, consider "
            "installing CURAND.");
    }
}

static void checkFunction(void* pointer, const std::string& name)
{
    if(pointer == nullptr)
    {
        throw std::runtime_error("Failed to load function '" + name +
            "' from dynamic library.");
    }
}

CurandLibrary::Interface::Interface()
: _library(nullptr), _failed(false)
{

}

CurandLibrary::Interface::~Interface()
{
    unload();
}

void CurandLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    //const char* libraryName = "libcurand-optimized.dylib";
    const char* libraryName = "libcurand.dylib";
    #else
    const char* libraryName = "libcurand.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CurandLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("CurandLibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {
        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(curandCreateGenerator);
        DynLink(curandDestroyGenerator);

        DynLink(curandSetPseudoRandomGeneratorSeed);

        DynLink(curandGenerateUniform);
        DynLink(curandGenerateNormal);

        DynLink(curandGenerateUniformDouble);
        DynLink(curandGenerateNormalDouble);

        #undef DynLink

        util::log("CurandLibrary") << " Loaded library '" << libraryName
            << "' successfully\n";
    }
    catch(...)
    {
        unload();
        throw;
    }

}

bool CurandLibrary::Interface::loaded() const
{
    return _library != nullptr;
}

void CurandLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

CurandLibrary::Interface CurandLibrary::_interface;

}

}

