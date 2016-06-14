/*  \file   CudaRuntimeLibrary.h
    \brief  The header file for the CudaRuntimeLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstring>
#include <string>

namespace prnn
{

namespace parallel
{

class CudaRuntimeLibrary
{
public:
    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost          =   0,
        cudaMemcpyHostToDevice        =   1,
        cudaMemcpyDeviceToHost        =   2,
        cudaMemcpyDeviceToDevice      =   3,
        cudaMemcpyDefault             =   4
    };

    enum cudaResult
    {
        cudaSuccess = 0
    };

    enum cudaDeviceAttr
    {
        cudaDevAttrMultiProcessorCount = 16,
        cudaDevAttrComputeCapabilityMajor = 75,
        cudaDevAttrComputeCapabilityMinor = 76
    };

    enum cudaHostAllocFlag
    {
        cudaHostAllocMappedFlag = 2
    };

    enum cudaLimit
    {
        cudaLimitStackSize = 0x00,
        cudaLimitPrintfFifoSize = 0x01,
        cudaLimitMallocHeapSize = 0x02,
        cudaLimitDevRuntimeSyncDepth = 0x03,
        cudaLimitDevRuntimePendingLaunchCount = 0x04
    };

public:
    static void load();
    static bool loaded();

public:
    static void cudaSetDevice(int device);
    static void cudaDeviceSynchronize();

    static void* cudaMalloc(size_t bytes);
    static void* cudaMallocManaged(size_t bytes);
    static void* cudaHostAlloc(size_t bytes);
    static void cudaFree(void* ptr);
    static void cudaFreeHost(void* ptr);

    static void cudaMemcpy(void* dest, const void* src,
        size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault);
    static void cudaMemcpyAsync(void* dest, const void* src,
        size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault, void* stream = nullptr);

    static void cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
    static void cudaDeviceSetLimit(cudaLimit limit, size_t value);

public:
    static std::string cudaGetErrorString(int error);

private:
    static void _check();

private:
    class Interface
    {
    public:
        int (*cudaSetDevice)(int ptr);
        int (*cudaDeviceSynchronize)();

        int (*cudaMalloc)(void** ptr, size_t bytes);
        int (*cudaMallocManaged)(void** ptr, size_t bytes, unsigned int flags);
        int (*cudaHostAlloc)(void** ptr, size_t bytes, unsigned int flags);
        int (*cudaFree)  (void*  ptr);
        int (*cudaFreeHost)  (void*  ptr);
        int (*cudaMemcpy)(void*  dest, const void* src, size_t bytes,
            cudaMemcpyKind kind);
        int (*cudaMemcpyAsync)(void*  dest, const void* src, size_t bytes,
            cudaMemcpyKind kind, void* stream);

        int (*cudaDeviceGetAttribute)(int* value, cudaDeviceAttr attr, int device);
        int (*cudaDeviceSetLimit)(cudaLimit limit, size_t value);

    public:
        const char* (*cudaGetErrorString)(int error);

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

