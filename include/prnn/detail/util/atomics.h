
#pragma once

#include <cstdint>

template<typename T, typename U>
class UnionCaster {
public:
    union {
        T output;
        U input;
    };
};

template <typename T, typename U>
__device__ inline T bit_cast(const U& v) {
    UnionCaster<T, U> caster;

    caster.input = v;

    return caster.output;
}

template <int bytes>
class ByteWidthType {
};

template <>
class ByteWidthType<4> {
public:
    typedef int type;
};

template <>
class ByteWidthType<8> {
public:
    typedef unsigned long long int type;
};

template <typename T>
class CASTypeConverter {
public:
    typedef typename ByteWidthType<sizeof(T)>::type type;

};

template <typename T>
__device__ inline T atomic_cas_relaxed(T& value, const T& old_value, const T& new_value) {
    typedef typename CASTypeConverter<T>::type Type;

    return atomicCAS(reinterpret_cast<Type*>(&value), bit_cast<Type>(old_value),
        bit_cast<Type>(new_value));
}

template <typename T>
class AtomicAdder
{
public:
    __device__ static T increment(T& address, const T& increment) {

        T updated_value = 0;
        T old_value;

        do {
            old_value = atomic_load_relaxed(address);

            T new_value = old_value + increment;

            updated_value = atomic_cas_relaxed(address, old_value, new_value);
        }
        while(updated_value != old_value);

        return updated_value;
    }

    __device__ static T increment(T& address, const T& value, int predicate) {
        if (predicate) {
            return increment(address, value);
        }
    }
};

template <>
class AtomicAdder<int>
{
public:
    __device__ static int increment(int& address, const int& increment) {
        return atomicAdd(&address, increment);
    }

    __device__ static int increment(int& address, const int& increment, int predicate) {
        if (predicate) {
            return atomicAdd(&address, increment);
        }

        return 0;
    }
};

template <>
class AtomicAdder<float>
{
public:
    __device__ static float increment(float& address, const float& increment) {
        float result;
        asm("atom.global.add.f32 %0, [%1], %2;" : "=f"(result) : "l"(&address),
            "f"(increment) : "memory");

        return result;
    }

    __device__ static float increment(float& address, const float& increment, int predicate) {
        float result = 0.0f;
        asm("{\n"
            //"    .reg .u64 address;"
            //"    cvta.global.u64 address, %0;\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %3, 0;\n\t"
            "   @p0 atom.global.add.f32 %0, [%1], %2;\n"
            "}" : "=f"(result) : "l"(&address), "f"(increment), "r"(predicate) : "memory");

        return result;
    }
};

template <typename T>
class AtomicReducer
{
public:
    __device__ static void increment(T& address, const T& increment) {
        AtomicAdder<T>::increment(address, increment);
    }

    __device__ static void increment(T& address, const T& increment, int condition) {
        AtomicAdder<T>::increment(address, increment, condition);
    }
};

template <>
class AtomicReducer<float>
{
public:
    __device__ static void increment(float& address, const float& increment) {
        asm("{\n"
            //"    .reg .u64 address;"
            //"    cvta.global.u64 address, %0;\n"
            "    red.global.add.f32 [%0], %1;\n"
            "}" :: "l"(&address), "f"(increment));// : "memory");
    }

    __device__ static void increment(float& address, const float& increment, int predicate) {
        asm("{\n"
            //"    .reg .u64 address;"
            //"    cvta.global.u64 address, %0;\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %2, 0;\n\t"
            "   @p0 red.global.add.f32 [%0], %1;\n"
            "}" :: "l"(&address), "f"(increment), "r"(predicate));// : "memory");
    }
};

template <>
class AtomicReducer<int32_t>
{
public:
    __device__ static void increment(int32_t& address, const int32_t& increment) {
        asm("{\n"
            //"    .reg .u64 address;"
            //"    cvta.global.u64 address, %0;\n"
            "    red.global.add.s32 [%0], %1;\n"
            //"    st.global.s32 [%0], %1;\n"
            "}" :: "l"(&address), "r"(increment));// : "memory");
    }

    __device__ static void increment(int32_t& address, const int32_t& increment, int predicate) {
        asm("{\n"
            //"    .reg .u64 address;"
            //"    cvta.global.u64 address, %0;\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %2, 0;\n\t"
            "   @p0 red.global.add.s32 [%0], %1;\n"
            //"    st.global.s32 [%0], %1;\n"
            "}" :: "l"(&address), "r"(increment), "r"(predicate));// : "memory");
    }
};

template <typename T>
__device__ inline T atomic_increment_relaxed(T& value, const T& increment) {
    return AtomicAdder<T>::increment(value, increment);
}

template <typename T>
__device__ inline void atomic_increment_reduce_relaxed(T& value, const T& increment) {
    return AtomicReducer<T>::increment(value, increment);
}

template <typename T>
__device__ inline void predicated_atomic_increment_reduce_relaxed(T& value, const T& increment,
    int predicate) {
    return AtomicReducer<T>::increment(value, increment, predicate);
}

__device__ inline void atomic_fence_release() {
    // Note: we should actually flush the caches here, otherwise this only applies to stores
    //       that bypass the cache.

    __threadfence();
}

__device__ inline void atomic_fence_acquire() {
    // Note: we should actually invalidate the caches here, otherwise this only loads to stores
    //       that bypass the cache.

    __threadfence();
}

template <typename T>
__device__ inline T atomic_increment_release(T& value, const T& increment) {
    T result = atomicAdd(&value, increment);

    atomic_fence_release();

    return result;
}

template <int size>
class GetAlignedType
{
public:


};

template <>
class GetAlignedType<1>
{
public:
    typedef int8_t type;
};

template <>
class GetAlignedType<2>
{
public:
    typedef int16_t type;
};

template <>
class GetAlignedType<4>
{
public:
    typedef float type;
};

template <>
class GetAlignedType<8>
{
public:
    typedef double type;
};

template <>
class GetAlignedType<12>
{
public:
    typedef float3 type;
};

template <>
class GetAlignedType<16>
{
public:
    typedef float4 type;
};

template <int size>
class L2SizedCacheAccessor
{
public:
    typedef typename GetAlignedType<size>::type type;

public:
    __device__ static void load(type& result, const type& value, int predicate) {
        if (predicate) {
            result = (*(volatile const type*)&value);
        }
    }

    __device__ static void store(type* result, const type& value, int predicate) {
        if (predicate) {
            (*(volatile type*)result) = (*(volatile const type*)&value);
        }
    }
};

template <>
class L2SizedCacheAccessor<8>
{
public:
    typedef float2 type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %3, 0;\n\t"
            "   @p0 ld.global.cg.v2.f32 {%0, %1}, [%2];\n\t"
            "}\n" : "=f"(result.x), "=f"(result.y)
                  : "l"(&value), "r"(predicate)
                  : "memory");
    }

    __device__ static void store(type* result, const type& value, int predicate) {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %0, 0;\n\t"
            "   @p0 st.global.cg.v2.f32 [%1], {%2, %3};\n\t"
            "}\n" :: "r"(predicate), "l"(result), "f"(value.x), "f"(value.y) : "memory");
    }
};

template <>
class L2SizedCacheAccessor<12>
{
public:
    typedef float3 type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %4, 0;\n\t"
            "   @p0 ld.global.cg.f32 %0, [%3 + 0x0];\n\t"
            "   @p0 ld.global.cg.f32 %1, [%3 + 0x4];\n\t"
            "   @p0 ld.global.cg.f32 %2, [%3 + 0x8];\n\t"
            "}\n" : "=f"(result.x), "=f"(result.y), "=f"(result.z)
                  : "l"(&value), "r"(predicate)
                  : "memory");
    }

    __device__ static void store(type* result, const type& value, int predicate) {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %0, 0;\n\t"
            "   @p0 st.global.cg.f32 [%1 + 0x0], %2;\n\t"
            "   @p0 st.global.cg.f32 [%1 + 0x4], %3;\n\t"
            "   @p0 st.global.cg.f32 [%1 + 0x8], %4;\n\t"
            "}\n" :: "r"(predicate), "l"(result), "f"(value.x), "f"(value.y),
                     "f"(value.z) : "memory");
    }
};

template <>
class L2SizedCacheAccessor<16>
{
public:
    typedef float4 type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %5, 0;\n\t"
            "   @p0 ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"
            "}\n" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
                  : "l"(&value), "r"(predicate)
                  : "memory");
    }

    __device__ static void store(type* result, const type& value, int predicate) {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %0, 0;\n\t"
            "   @p0 st.global.cg.v4.f32 [%1], {%2, %3, %4, %5};\n\t"
            "}\n" :: "r"(predicate), "l"(result), "f"(value.x), "f"(value.y),
                     "f"(value.z), "f"(value.w) : "memory");
    }
};

template <typename T>
class L2CacheAccessor
{
public:
    __device__ static T load(T& value)
    {
        return (*(volatile T*)&value);
    }

    __device__ static void load(T& result, const T& value, int predicate)
    {
        typedef L2SizedCacheAccessor<sizeof(T)> Accessor;

        typedef typename Accessor::type type;

        Accessor::load(reinterpret_cast<type&>(result),
            reinterpret_cast<const type&>(value), predicate);
    }

    __device__ static void store(T* value, const T& data)
    {
        (*(volatile T*)value) = data;
    }

    __device__ static void store(T* value, const T& data, int predicate)
    {
        typedef L2SizedCacheAccessor<sizeof(T)> Accessor;

        typedef typename Accessor::type type;

        Accessor::store(reinterpret_cast<type*>(value),
            reinterpret_cast<const type&>(data), predicate);
    }
};

template <>
class L2CacheAccessor<float>
{
public:
    __device__ static float load(const float& value)
    {
        float result;

        asm("ld.global.cg.f32 %0, [%1];" : "=f"(result) : "l"(&value));

        return result;
    }

    __device__ static void load(float& result, const float& value, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %1, 0;\n\t"
            "   @p0 ld.global.cg.f32 %0, [%2];\n\t"
            "}\n" : "=f"(result) : "r"(predicate), "l"(&value) : "memory");
    }

    __device__ static void store(float* address, const float& data)
    {
        asm("st.global.cg.f32 [%0], %1;\n" :: "l"(address),  "f"(data) : "memory");
    }

    __device__ static void store(float* address, const float& data, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %0, 0;\n\t"
            "   @p0 st.global.cg.f32 [%1], %2;\n\t"
            "}\n" :: "r"(predicate), "l"(address), "f"(data) : "memory");
    }

};

template <int size>
class SharedMemorySizedCacheAccessor
{
public:
    typedef typename GetAlignedType<size>::type type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        if (predicate) {
            result = value;
        }
    }
};

/*
template <>
class SharedMemorySizedCacheAccessor<8>
{
public:
    typedef float2 type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        asm("{\n"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %3, 0;\n\t"
            "   @p0 ld.shared.v2.f32 {%0, %1}, [%2];\n\t"
            "}\n" : "=f"(result.x), "=f"(result.y) : "l"(&value), "r"(predicate): "memory");
    }
};

template <>
class SharedMemorySizedCacheAccessor<16>
{
public:
    typedef float4 type;

public:
    __device__ static void load(type& result, const type& value, int predicate)
    {
        asm("{\n\t"
            "   .reg .pred p0;\n\t"
            "   setp.ne.u32 p0, %5, 0;\n\t"
            "   @p0 ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"
            "}\n" : "=f"(result.x), "=f"(result.y), "=f"(result.z),
            "=f"(result.w) : "l"(&value), "r"(predicate): "memory");
    }
};
*/

template <typename T>
class SharedMemoryAccessor
{
public:
    __device__ static void load(T& result, const T& value, int predicate)
    {
        typedef SharedMemorySizedCacheAccessor<sizeof(T)> Accessor;

        typedef typename Accessor::type type;

        Accessor::load(reinterpret_cast<type&>(result),
            reinterpret_cast<const type&>(value), predicate);
    }
};

template <typename T>
__device__ inline T atomic_load_relaxed(const T& value) {
    return L2CacheAccessor<T>::load(value);
}

template <typename T>
__device__ inline T atomic_load_relaxed(T& value) {
    return L2CacheAccessor<T>::load(value);
}

template <typename T>
__device__ inline void predicated_atomic_load_relaxed(T& result, const T& value, bool predicate) {
    if (predicate) {
        result = L2CacheAccessor<T>::load(value);
    }
}

template <typename T>
__device__ inline void predicated_atomic_global_load_relaxed(T& result, const T& value, bool predicate) {
    L2CacheAccessor<T>::load(result, value, predicate);
}

template <typename T>
__device__ inline void predicated_atomic_shared_load_relaxed(T& result, const T& value, bool predicate) {
    SharedMemoryAccessor<T>::load(result, value, predicate);
}

template <typename T>
__device__ inline void predicated_atomic_load_relaxed(T& result, T& value, bool predicate) {
    if (predicate) {
        result = L2CacheAccessor<T>::load(value);
    }
}

template <typename T>
__device__ inline void atomic_store_relaxed(T& value, const T& data) {
    L2CacheAccessor<T>::store(&value, data);
}

template <typename T>
__device__ inline void predicated_atomic_store_relaxed(T& value, const T& data, bool predicate) {
    if (predicate) {
        L2CacheAccessor<T>::store(&value, data);
    }
}

template <typename T>
__device__ inline void predicated_atomic_global_store_relaxed(T& value, const T& data, bool predicate) {
    L2CacheAccessor<T>::store(&value, data, predicate);
}




