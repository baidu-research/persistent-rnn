/*  \file   AtlasLibrary.cpp
    \brief  The source file for the AtlasLibrary class.
*/

// Persistent RNN Includes
#include <prnn/detail/matrix/atlas_library.h>

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

void AtlasLibrary::load()
{
    _interface.load();
}

bool AtlasLibrary::loaded()
{
    load();

    return _interface.loaded();
}

void AtlasLibrary::sgemm(const int Order, const int TransA,
         const int TransB, const int M, const int N,
         const int K, const float alpha, const float *A,
         const int lda, const float *B, const int ldb,
         const float beta, float *C, const int ldc)
{
    _check();

    (*_interface.cblas_sgemm)(Order, TransA, TransB, M, N, K, alpha, A, lda, B,
        ldb, beta, C, ldc);
}

void AtlasLibrary::dgemm(const int Order, const int TransA,
         const int TransB, const int M, const int N,
         const int K, const double alpha, const double* A,
         const int lda, const double* B, const int ldb,
         const double beta, double* C, const int ldc)
{
    _check();

    (*_interface.cblas_dgemm)(Order, TransA, TransB, M, N, K, alpha, A, lda, B,
        ldb, beta, C, ldc);
}

void AtlasLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call ATLAS function when "
            "the library is not loaded. Loading library failed, consider "
            "installing ATLAS.");
    }
}

AtlasLibrary::Interface::Interface()
: _library(nullptr), _failed(false)
{

}

AtlasLibrary::Interface::~Interface()
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

void AtlasLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;

    #ifdef __APPLE__
    const char* libraryName = "libcblas.dylib";
    #else
    const char* libraryName = "libcblas.so.3";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("AtlasLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("AtlasLibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    #define DynLink( function ) util::bit_cast(function, \
        dlsym(_library, #function)); checkFunction((void*)function, #function)

    DynLink(cblas_sgemm);
    DynLink(cblas_dgemm);

    #undef DynLink

    util::log("AtlasLibrary") << " Loaded library '" << libraryName
        << "' successfully\n";

}

bool AtlasLibrary::Interface::loaded() const
{
    return _library != nullptr;
}

void AtlasLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

AtlasLibrary::Interface AtlasLibrary::_interface;

}

}


