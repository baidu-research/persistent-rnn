
#pragma once

#define STRING(x) #x

#if defined(__APPLE__) && defined(__CUDA_ARCH__) && !defined(NDEBUG)

    #ifdef assert
    #undef assert
    #endif

    #include <cstdio>
    #define assert(e) \
    do \
    { \
        if(!(e))\
        { \
            printf("%s:%d Assertion '%s' failed.\n", __FILE__, __LINE__, STRING(e)); \
            asm("trap;"); \
        } \
    } \
    while(0)

#else

    #include <cassert>

#endif


