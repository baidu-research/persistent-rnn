
// Persistent RNN Includes
#include <prnn/detail/matrix/matrix.h>
#include <prnn/detail/matrix/matrix_operations.h>

#include <prnn/detail/parallel/synchronization.h>

#include <prnn/detail/util/timer.h>
#include <prnn/detail/util/argument_parser.h>

// Standard Library Includes
#include <iostream>

__global__ void performAtomicsKernel(int* array, size_t size)
{
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t gridSize = gridDim.x * blockDim.x;

    for(size_t i = id; i < size; i += gridSize)
    {
        atomicAdd(array + i, 1);
    }
}

void performAtomics(int* array, size_t size)
{
    size_t blocks  = 128;
    size_t threads = 128;

    performAtomicsKernel<<<blocks, threads>>>(array, size);
}

void benchmarkAtomics(size_t size, size_t iterations)
{
    auto precision = prnn::matrix::SinglePrecision();

    auto data = prnn::matrix::zeros({size}, precision);

    // warm up
    performAtomics(reinterpret_cast<int*>(data.data()), size);
    prnn::parallel::synchronize();

    prnn::util::Timer timer;

    timer.start();

    for(size_t i = 0; i < size; ++i)
    {
        performAtomics(reinterpret_cast<int*>(data.data()), size);
    }

    prnn::parallel::synchronize();

    timer.stop();

    double totalBytes = iterations * size * sizeof(int);
    double gigaBytesPerSecond = totalBytes / (timer.seconds() * 1.0e9);

    std::cout << "32-bit Atomic Increment Throughput " << gigaBytesPerSecond << " GB/s\n";
}

int main(int argc, char** argv)
{
    prnn::util::ArgumentParser parser(argc, argv);

    prnn::matrix::Precision precision = prnn::matrix::SinglePrecision();

    size_t iterations = 20;
    size_t size       = (1 << 20);

    parser.parse("-i", "--iterations", iterations, iterations, "Iterations to run each atomic kernel.");
    parser.parse("-s", "--size",       size,       size,       "The size of the array to operate on.");

    parser.parse();

    benchmarkAtomics(size, iterations);

    return 0;
}



