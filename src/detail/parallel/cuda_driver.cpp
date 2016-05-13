/*! \file   CudaDriver.cpp
    \brief  The source file for the CudaDriver class.
*/

// Persistent RNN Includes
#include <prnn/detail/parallel/cuda_driver.h>
#include <prnn/detail/parallel/cuda.h>

#include <prnn/detail/util/casts.h>
#include <prnn/detail/util/logger.h>

// System Specific Includes
#include <dlfcn.h>
#include <stdexcept>

namespace prnn
{

namespace parallel
{

void CudaDriver::load()
{
    _interface.load();
}

bool CudaDriver::loaded()
{
    load();

    return _interface.loaded();
}

void CudaDriver::cuInit(unsigned int f)
{
    _check();

    _checkResult((*_interface.cuInit)(f));
}

void CudaDriver::cuDriverGetVersion(int* v)
{
    _check();

    _checkResult((*_interface.cuDriverGetVersion)(v));
}

void CudaDriver::cuDeviceGet(CUdevice* d, int o)
{
    _check();

    _checkResult((*_interface.cuDeviceGet)(d, o));
}

void CudaDriver::cuDeviceGetCount(int* c)
{
    _check();

    _checkResult((*_interface.cuDeviceGetCount)(c));
}

void CudaDriver::cuDeviceGetName(char* n, int l, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuDeviceGetName)(n, l, d));
}

void CudaDriver::cuDeviceComputeCapability(int* m, int* minor, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuDeviceComputeCapability)(m, minor, d));
}

void CudaDriver::cuDeviceTotalMem(size_t* b, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuDeviceTotalMem_v2)(b, d));
}

void CudaDriver::cuDeviceGetProperties(CUdevprop* p, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuDeviceGetProperties)(p, d));
}

void CudaDriver::cuDeviceGetAttribute(int* p, CUdevice_attribute a, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuDeviceGetAttribute)(p, a, d));
}

void CudaDriver::cuCtxPushCurrent(CUcontext c)
{
    _check();

    _checkResult((*_interface.cuCtxPushCurrent_v2)(c));
}

void CudaDriver::cuCtxGetCurrent(CUcontext* c)
{
    _check();

    _checkResult((*_interface.cuCtxGetCurrent)(c));
}

void CudaDriver::cuCtxCreate(CUcontext* c, unsigned int f, CUdevice d)
{
    _check();

    _checkResult((*_interface.cuCtxCreate_v2)(c, f, d));
}

void CudaDriver::cuCtxGetApiVersion(CUcontext c, unsigned int* v)
{
    _check();

    _checkResult((*_interface.cuCtxGetApiVersion)(c, v));
}

void CudaDriver::cuCtxDestroy(CUcontext c)
{
    _check();

    _checkResult((*_interface.cuCtxDestroy_v2)(c));
}

void CudaDriver::cuCtxSynchronize(void)
{
    _check();

    _checkResult((*_interface.cuCtxSynchronize)());
}

void CudaDriver::cuModuleLoadDataEx(CUmodule* m,
    const void* i, unsigned int n,
    CUjit_option* o, void** v)
{
    _check();

    _checkResult((*_interface.cuModuleLoadDataEx)(m, i, n, o, v));
}

void CudaDriver::cuModuleUnload(CUmodule h)
{
    _check();

    _checkResult((*_interface.cuModuleUnload)(h));
}

void CudaDriver::cuModuleGetFunction(CUfunction* f, CUmodule m, const char* n)
{
    _check();

    _checkResult((*_interface.cuModuleGetFunction)(f, m, n));
}

void CudaDriver::cuModuleGetGlobal(CUdeviceptr* p,
        size_t* b, CUmodule m, const char* n)
{
    _check();

    _checkResult((*_interface.cuModuleGetGlobal_v2)(p, b, m, n));
}

void CudaDriver::cuMemGetInfo(size_t* free, size_t* total)
{
    _check();

    _checkResult((*_interface.cuMemGetInfo_v2)(free, total));
}

void CudaDriver::cuMemAlloc(CUdeviceptr* p, unsigned int b)
{
    _check();

    _checkResult((*_interface.cuMemAlloc_v2)(p, b));
}

void CudaDriver::cuMemFree(CUdeviceptr p)
{
    _check();

    _checkResult((*_interface.cuMemFree_v2)(p));
}

void CudaDriver::cuMemGetAddressRange(CUdeviceptr* p, size_t* d, CUdeviceptr dp)
{
    _check();

    _checkResult((*_interface.cuMemGetAddressRange_v2)(p, d, dp));
}

void CudaDriver::cuMemAllocHost(void** p, unsigned int b)
{
    _check();

    _checkResult((*_interface.cuMemAllocHost_v2)(p, b));
}

void CudaDriver::cuMemFreeHost(void* p)
{
    _check();

    _checkResult((*_interface.cuMemFreeHost)(p));
}

void CudaDriver::cuMemHostAlloc(void** p, unsigned long long b, unsigned int f)
{
    _check();

    _checkResult((*_interface.cuMemHostAlloc)(p, b, f));
}

void CudaDriver::cuMemHostRegister(void* p, unsigned long long b, unsigned int f)
{
    _check();

    _checkResult((*_interface.cuMemHostRegister)(p, b, f));
}

void CudaDriver::cuMemHostUnregister(void* p)
{
    _check();

    _checkResult((*_interface.cuMemHostUnregister)(p));
}

void CudaDriver::cuMemHostGetDevicePointer(CUdeviceptr* d,
    void* p, unsigned int f)
{
    _check();

    _checkResult((*_interface.cuMemHostGetDevicePointer_v2)(d, p, f));
}

void CudaDriver::cuMemHostGetFlags(unsigned int* f, void* p)
{
    _check();

    _checkResult((*_interface.cuMemHostGetFlags)(f, p));
}


void CudaDriver::cuMemcpyHtoD(CUdeviceptr d,
    const void* s, unsigned int b)
{
    _check();

    _checkResult((*_interface.cuMemcpyHtoD_v2)(d, s, b));
}

void CudaDriver::cuMemcpyDtoH(void* d, CUdeviceptr s,
    unsigned int b)
{
    _check();

    _checkResult((*_interface.cuMemcpyDtoH_v2)(d, s, b));
}

void CudaDriver::cuFuncSetBlockShape(CUfunction h, int x,
    int y, int z)
{
    _check();

    _checkResult((*_interface.cuFuncSetBlockShape)(h, x, y, z));
}

void CudaDriver::cuFuncSetSharedSize(CUfunction h, unsigned int b)
{
    _check();

    _checkResult((*_interface.cuFuncSetSharedSize)(h, b));
}

void CudaDriver::cuParamSetSize(CUfunction h, unsigned int n)
{
    _check();

    _checkResult((*_interface.cuParamSetSize)(h, n));
}

void CudaDriver::cuParamSetv(CUfunction f, int o, void* p, unsigned int b)
{
    _check();

    _checkResult((*_interface.cuParamSetv)(f, o, p, b));
}

void CudaDriver::cuLaunchGrid (CUfunction f, int w, int h)
{
    _check();

    _checkResult((*_interface.cuLaunchGrid)(f, w, h));
}

void CudaDriver::cuEventCreate(CUevent* e, unsigned int f)
{
    _check();

    _checkResult((*_interface.cuEventCreate)(e, f));
}

void CudaDriver::cuEventRecord(CUevent e, CUstream s)
{
    _check();

    _checkResult((*_interface.cuEventRecord)(e, s));
}

void CudaDriver::cuEventQuery(CUevent e)
{
    _check();

    _checkResult((*_interface.cuEventQuery)(e));
}

void CudaDriver::cuEventSynchronize(CUevent e)
{
    _check();

    _checkResult((*_interface.cuEventSynchronize)(e));
}

void CudaDriver::cuEventDestroy(CUevent e)
{
    _check();

    _checkResult((*_interface.cuEventDestroy_v2)(e));
}

void CudaDriver::cuEventElapsedTime(float* t,
        CUevent s, CUevent e)
{
    _check();

    _checkResult((*_interface.cuEventElapsedTime)(t, s, e));
}

std::string CudaDriver::toString(CUresult result)
{
    switch(result)
    {
        case CUDA_SUCCESS: return "CUDA DRIVER - no errors";
        case CUDA_ERROR_INVALID_VALUE: return "CUDA DRIVER - invalid value";
        case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA DRIVER - out of memory";
        case CUDA_ERROR_NOT_INITIALIZED: return "CUDA DRIVER - driver not initialized";
        case CUDA_ERROR_DEINITIALIZED: return "CUDA DRIVER - deinitialized";
        case CUDA_ERROR_NO_DEVICE: return "CUDA DRIVER - no device";
        case CUDA_ERROR_INVALID_DEVICE: return "CUDA DRIVER - invalid device";
        case CUDA_ERROR_INVALID_IMAGE: return "CUDA DRIVER - invalid kernel image";
        case CUDA_ERROR_INVALID_CONTEXT: return "CUDA DRIVER - invalid context";
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "CUDA DRIVER - context already current";
        case CUDA_ERROR_MAP_FAILED: return "CUDA DRIVER - map failed";
        case CUDA_ERROR_UNMAP_FAILED: return "CUDA DRIVER - unmap failed";
        case CUDA_ERROR_ARRAY_IS_MAPPED: return "CUDA DRIVER - array is mapped";
        case CUDA_ERROR_ALREADY_MAPPED: return "CUDA DRIVER - already mapped";
        case CUDA_ERROR_NO_BINARY_FOR_GPU: return "CUDA DRIVER - no gpu binary";
        case CUDA_ERROR_ALREADY_ACQUIRED: return "CUDA DRIVER - already aquired";
        case CUDA_ERROR_NOT_MAPPED: return "CUDA DRIVER - not mapped";
        case CUDA_ERROR_INVALID_SOURCE: return "CUDA DRIVER - invalid source";
        case CUDA_ERROR_FILE_NOT_FOUND: return "CUDA DRIVER - file not found";
        case CUDA_ERROR_INVALID_HANDLE: return "CUDA DRIVER - invalid handle";
        case CUDA_ERROR_NOT_FOUND: return "CUDA DRIVER - not found";
        case CUDA_ERROR_NOT_READY: return "CUDA DRIVER - not ready";
        case CUDA_ERROR_LAUNCH_FAILED: return "CUDA DRIVER - launch failed";
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA DRIVER - out of resources";
        case CUDA_ERROR_LAUNCH_TIMEOUT: return "CUDA DRIVER - launch timeout";
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "CUDA DRIVER - incompatible texturing";
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA DRIVER - not mapped as pointer";
        case CUDA_ERROR_UNKNOWN: return "CUDA DRIVER - unknown error";
        default: break;
    }
    return "invalid_error";
}

bool CudaDriver::doesFunctionExist(CUmodule m, const char* n)
{
    _check();

    CUfunction f;

    CUresult result = (*_interface.cuModuleGetFunction)(&f, m, n);

    if(result == CUDA_ERROR_NOT_FOUND)
    {
        return false;
    }

    _checkResult(result);

    return true;
}

void CudaDriver::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call libcuda function when "
            "the library is not loaded. Loading library failed, consider "
            "installing libcuda or putting it on your library search path.");
    }
}

static std::string errorCodeToString(CUresult r)
{
    std::stringstream stream;

    stream << "(" << r << ") : " << CudaDriver::toString(r);

    // TODO: add more info

    return stream.str();
}

void CudaDriver::_checkResult(CUresult r)
{
    if(r != CUDA_SUCCESS)
    {
        throw std::runtime_error("libcuda API call returned error code: " +
            errorCodeToString(r));
    }
}

CudaDriver::Interface CudaDriver::_interface;

CudaDriver::Interface::Interface()
: _library(nullptr)
{

}

CudaDriver::Interface::~Interface()
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

void CudaDriver::Interface::load()
{
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    const char* libraryName = "libcuda.dylib";
    #else
    const char* libraryName = "libcuda.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CudaDriver") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("CudaDriver") << " loading library '" << libraryName << "' failed\n";
        return;
    }

    #define DynLink( function ) util::bit_cast(function, dlsym(_library, #function)); checkFunction((void*)function, #function)

    DynLink(cuInit);
    DynLink(cuDriverGetVersion);
    DynLink(cuDeviceGet);
    DynLink(cuDeviceGetCount);
    DynLink(cuDeviceGetName);
    DynLink(cuDeviceComputeCapability);
    DynLink(cuDeviceTotalMem_v2);
    DynLink(cuDeviceGetProperties);
    DynLink(cuDeviceGetAttribute);
    DynLink(cuCtxCreate_v2);
    DynLink(cuCtxPushCurrent_v2);
    DynLink(cuCtxGetCurrent);
    DynLink(cuCtxDestroy_v2);
    DynLink(cuCtxGetApiVersion);
    DynLink(cuCtxSynchronize);

    DynLink(cuModuleLoadDataEx);
    DynLink(cuModuleUnload);
    DynLink(cuModuleGetFunction);
    DynLink(cuModuleGetGlobal_v2);
    DynLink(cuFuncSetBlockShape);
    DynLink(cuFuncSetSharedSize);

    DynLink(cuMemGetInfo_v2);
    DynLink(cuMemAlloc_v2);
    DynLink(cuMemFree_v2);
    DynLink(cuMemGetAddressRange_v2);
    DynLink(cuMemAllocHost_v2);
    DynLink(cuMemFreeHost);

    DynLink(cuMemHostAlloc);
    DynLink(cuMemHostRegister);
    DynLink(cuMemHostUnregister);
    DynLink(cuMemHostGetDevicePointer_v2);
    DynLink(cuMemHostGetFlags);

    DynLink(cuMemcpyHtoD_v2);
    DynLink(cuMemcpyDtoH_v2);

    DynLink(cuParamSetv);
    DynLink(cuParamSetSize);

    DynLink(cuLaunchGrid);
    DynLink(cuEventCreate);
    DynLink(cuEventRecord);
    DynLink(cuEventQuery);
    DynLink(cuEventSynchronize);
    DynLink(cuEventDestroy_v2);
    DynLink(cuEventElapsedTime);

    #undef DynLink

    util::log("CudaDriver") << " success\n";

    _createContext();
}

bool CudaDriver::Interface::loaded() const
{
    return _library != nullptr;
}

void CudaDriver::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

void CudaDriver::Interface::_createContext()
{
    try
    {
        cuInit(0);

        cuCtxGetCurrent(&context);

        if(context == nullptr)
        {
            cuCtxCreate(&context, CU_CTX_MAP_HOST, 0);
            util::log("CudaDriver") << " created new context\n";
        }
        else
        {
            unsigned int version  = 0;
            cuCtxGetApiVersion(context, &version);

            util::log("CudaDriver") << " extracted existing context with version " << version << "\n";
        }
    }
    catch(...)
    {
        unload();

        throw;
    }
}

}

}


