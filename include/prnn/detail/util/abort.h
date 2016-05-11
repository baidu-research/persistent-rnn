
#pragma once

#include <cstdio>

__device__ inline void abort_gpu_kernel(const char* message) {
    printf("Aborted GPU Kernel: `%s` .\n", message);
    asm("trap;");
}


