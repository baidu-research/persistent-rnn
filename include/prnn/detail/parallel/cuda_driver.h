/*! \file   CudaDriver.h
    \author Gregory Diamos <solusstultus@gmail.com>
    \date   Wednesday November 13, 2013
    \brief  The header file for the CudaDriver class.
*/

#pragma once

// Persistent RNN Includes
#include <prnn/detail/parallel/cuda_driver_types.h>

// Standard Library Includes
#include <cstring>
#include <string>

namespace prnn
{

namespace parallel
{

class CudaDriver
{
public:
    static void load();
    static bool loaded();

public:
    /*********************************
    ** Initialization
    *********************************/
    static void cuInit(unsigned int Flags);

    /*********************************
    ** Driver Version Query
    *********************************/
    static void cuDriverGetVersion(int* driverVersion);

    /************************************
    **
    **    Device management
    **
    ***********************************/

    static void cuDeviceGet(CUdevice* device, int ordinal);
    static void cuDeviceGetCount(int* count);
    static void cuDeviceGetName(char* name, int len, CUdevice dev);
    static void cuDeviceComputeCapability(int* major, int* minor,
        CUdevice dev);
    static void cuDeviceTotalMem(size_t* bytes, CUdevice dev);
    static void cuDeviceGetProperties(CUdevprop* prop,
        CUdevice dev);
    static void cuDeviceGetAttribute(int* pi,
        CUdevice_attribute attrib, CUdevice dev);

    /************************************
    **
    **    Context management
    **
    ***********************************/

    static void cuCtxGetCurrent(CUcontext* c);
    static void cuCtxPushCurrent(CUcontext c);
    static void cuCtxCreate(CUcontext* c, unsigned int flags, CUdevice dev);
    static void cuCtxGetApiVersion(CUcontext ctx, unsigned int* version);
    static void cuCtxDestroy(CUcontext ctx);
    static void cuCtxSynchronize(void);

    /************************************
    **
    **    Module management
    **
    ***********************************/

    static void cuModuleLoadDataEx(CUmodule* module,
        const void* image, unsigned int numOptions,
        CUjit_option* options, void** optionValues);
    static void cuModuleUnload(CUmodule hmod);
    static void cuModuleGetFunction(CUfunction* hfunc,
        CUmodule hmod, const char* name);
    static void cuModuleGetGlobal(CUdeviceptr* dptr,
        size_t* bytes, CUmodule hmod, const char* name);

    /************************************
    **
    **    Memory management
    **
    ***********************************/

    static void cuMemGetInfo(size_t* free,
        size_t* total);

    static void cuMemAlloc(CUdeviceptr* dptr,
        unsigned int bytesize);
    static void cuMemFree(CUdeviceptr dptr);
    static void cuMemGetAddressRange(CUdeviceptr* pbase,
        size_t* psize, CUdeviceptr dptr);

    static void cuMemAllocHost(void** pp, unsigned int bytesize);
    static void cuMemFreeHost(void* p);

    static void cuMemHostAlloc(void** pp,
        unsigned long long bytesize, unsigned int Flags);
    static void cuMemHostRegister(void* pp,
        unsigned long long bytesize, unsigned int Flags);
    static void cuMemHostUnregister(void* pp);

    static void cuMemHostGetDevicePointer(CUdeviceptr* pdptr,
        void* p, unsigned int Flags);
    static void cuMemHostGetFlags(unsigned int* pFlags, void* p);

    /************************************
    **
    **    Synchronous Memcpy
    **
    ** Intra-device memcpy's done with these functions may execute
    **    in parallel with the CPU,
    ** but if host memory is involved, they wait until the copy is
    **    done before returning.
    **
    ***********************************/

    // 1D functions
    // system <-> device memory
    static void cuMemcpyHtoD (CUdeviceptr dstDevice,
        const void* srcHost, unsigned int ByteCount);
    static void cuMemcpyDtoH (void* dstHost, CUdeviceptr srcDevice,
        unsigned int ByteCount);


    /************************************
    **
    **    Function management
    **
    ***********************************/

    static void cuFuncSetBlockShape (CUfunction hfunc, int x,
        int y, int z);
    static void cuFuncSetSharedSize (CUfunction hfunc,
        unsigned int bytes);


    /************************************
    **
    **    Parameter management
    **
    ***********************************/

    static void cuParamSetSize(CUfunction hfunc,
        unsigned int numbytes);
    static void cuParamSetv(CUfunction hfunc, int offset,
        void*  ptr, unsigned int numbytes);

    /************************************
    **
    **    Launch functions
    **
    ***********************************/
    static void cuLaunchGrid (CUfunction f, int grid_width,
        int grid_height);

    /************************************
    **
    **    Events
    **
    ***********************************/
    static void cuEventCreate(CUevent* phEvent,
        unsigned int Flags);
    static void cuEventRecord(CUevent hEvent, CUstream hStream);
    static void cuEventQuery(CUevent hEvent);
    static void cuEventSynchronize(CUevent hEvent);
    static void cuEventDestroy(CUevent hEvent);
    static void cuEventElapsedTime(float* pMilliseconds,
        CUevent hStart, CUevent hEnd);

    /************************************
    **
    **    Error Reporting
    **
    ***********************************/
    static std::string toString(CUresult result);

    /************************************
    **
    **    New
    **
    ***********************************/
    static bool doesFunctionExist(CUmodule hmod, const char* name);

private:
    static void _check();
    static void _checkResult(CUresult);

private:
    class Interface
    {
    public:
        CUresult (*cuInit)(unsigned int Flags);
        CUresult (*cuDriverGetVersion)(int* driverVersion);
        CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
        CUresult (*cuDeviceGetCount)(int* count);
        CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
        CUresult (*cuDeviceComputeCapability)(int* major,
            int* minor, CUdevice dev);
        CUresult (*cuDeviceTotalMem_v2)(size_t* bytes,
            CUdevice dev);
        CUresult (*cuDeviceGetProperties)(CUdevprop* prop,
            CUdevice dev);
        CUresult (*cuDeviceGetAttribute)(int* pi,
            CUdevice_attribute attrib, CUdevice dev);
        CUresult (*cuCtxCreate_v2)(CUcontext* pctx,
            unsigned int flags, CUdevice dev);
        CUresult (*cuCtxGetCurrent)(CUcontext* pctx);
        CUresult (*cuCtxPushCurrent_v2)(CUcontext pctx);
        CUresult (*cuCtxGetApiVersion)(CUcontext ctx, unsigned int* version);
        CUresult (*cuCtxSynchronize)(void);
        CUresult (*cuCtxDestroy_v2)(CUcontext ctx);

        CUresult (*cuModuleLoadDataEx)(CUmodule* module,
            const void* image, unsigned int numOptions,
            CUjit_option* options, void** optionValues);
        CUresult (*cuModuleUnload)(CUmodule hmod);
        CUresult (*cuModuleGetFunction)(CUfunction* hfunc,
            CUmodule hmod, const char* name);
        CUresult (*cuModuleGetGlobal_v2)(CUdeviceptr* dptr,
            size_t* bytes, CUmodule hmod, const char* name);
        CUresult (*cuFuncSetBlockShape)(CUfunction hfunc, int x,
            int y, int z);
        CUresult (*cuFuncSetSharedSize)(CUfunction hfunc,
            unsigned int bytes);

        CUresult (*cuMemGetInfo_v2)(size_t* free,
            size_t* total);
        CUresult (*cuMemAlloc_v2)(CUdeviceptr* dptr,
            unsigned int bytesize);
        CUresult (*cuMemFree_v2)(CUdeviceptr dptr);
        CUresult (*cuMemGetAddressRange_v2)(CUdeviceptr* pbase,
            size_t* psize, CUdeviceptr dptr);
        CUresult (*cuMemAllocHost_v2)(void** pp,
            unsigned int bytesize);
        CUresult (*cuMemFreeHost)(void* p);

        CUresult (*cuMemHostAlloc)(void** pp,
            unsigned long long bytesize, unsigned int Flags);
        CUresult (*cuMemHostRegister)(void* pp,
            unsigned long long bytesize, unsigned int Flags);
        CUresult (*cuMemHostUnregister)(void* pp);
        CUresult (*cuMemHostGetDevicePointer_v2)(CUdeviceptr* pdptr,
            void* p, unsigned int Flags);
        CUresult (*cuMemHostGetFlags)(unsigned int* pFlags, void* p);

        CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr dstDevice,
            const void* srcHost, unsigned int ByteCount);
        CUresult (*cuMemcpyDtoH_v2)(void* dstHost,
            CUdeviceptr srcDevice, unsigned int ByteCount);

        CUresult (*cuParamSetSize)(CUfunction hfunc,
            unsigned int numbytes);
        CUresult (*cuParamSetv)(CUfunction hfunc, int offset,
            void*  ptr, unsigned int numbytes);

        CUresult (*cuLaunchGrid)(CUfunction f, int grid_width,
            int grid_height);
        CUresult (*cuEventCreate)(CUevent* phEvent,
            unsigned int Flags);
        CUresult (*cuEventRecord)(CUevent hEvent,
            CUstream hStream);
        CUresult (*cuEventQuery)(CUevent hEvent);
        CUresult (*cuEventSynchronize)(CUevent hEvent);
        CUresult (*cuEventDestroy_v2)(CUevent hEvent);
        CUresult (*cuEventElapsedTime)(float* pMilliseconds,
            CUevent hStart, CUevent hEnd);

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

    public:
        CUcontext context;

    private:
        void _createContext();

    private:
        void* _library;

    };

private:
    static Interface _interface;

};

}

}


