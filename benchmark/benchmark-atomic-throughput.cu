
// Persistent RNN Includes
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_operations.h>

#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/timer.h>
#include <prnn/detail/util/argument_parser.h>

// Standard Library Includes
#include <iostream>

typedef float Type;

__global__ void performAtomicsKernel(Type* array, size_t size, size_t collisions)
{
    size_t id = threadIdx.x + (blockIdx.x / collisions) * blockDim.x;
    size_t gridSize = gridDim.x * blockDim.x;

    for(size_t i = 2*id; 2*i < size; i += 2*gridSize)
    {
        atomicAdd(array + 2*i, 1);
        atomicAdd(array + 2*i + 1, 1);
    }
}

void performAtomics(Type* array, size_t size, size_t collisions)
{
    size_t blocks  = 128;
    size_t threads = 128;

    performAtomicsKernel<<<blocks, threads>>>(array, size, collisions);
}

void benchmarkAtomics(size_t size, size_t iterations, size_t collisions)
{
    auto precision = prnn::matrix::SinglePrecision();

    auto data = prnn::matrix::zeros({size}, precision);

    // warm up
    performAtomics(reinterpret_cast<Type*>(data.data()), size, collisions);
    cudaDeviceSynchronize();

    prnn::util::Timer timer;

    timer.start();

    for(size_t i = 0; i < iterations; ++i)
    {
        performAtomics(reinterpret_cast<Type*>(data.data()), size, collisions);
    }

    cudaDeviceSynchronize();

    timer.stop();

    double totalBytes = 2 * iterations * size * sizeof(int);
    double gigaBytesPerSecond = totalBytes / (timer.seconds() * 1.0e9);

    std::cout << "32-bit Atomic Increment Throughput " << gigaBytesPerSecond << " GB/s\n";
}

int main(int argc, char** argv)
{
    prnn::util::ArgumentParser parser(argc, argv);

    size_t iterations = 20;
    size_t size       = (1 << 20);
    size_t collisions = 16*12;

    parser.parse("-i", "--iterations", iterations, iterations, "Iterations to run each atomic kernel.");
    parser.parse("-s", "--size",       size,       size,       "The size of the array to operate on.");
    parser.parse("-c", "--collisions", collisions, collisions, "The number of collisions.");

    parser.parse();

    benchmarkAtomics(size * collisions, iterations, collisions);

    return 0;
}



