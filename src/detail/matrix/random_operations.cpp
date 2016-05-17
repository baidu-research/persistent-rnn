
// Persistent RNN Includes
#include <prnn/detail/matrix/random_operations.h>
#include <prnn/detail/matrix/curand_library.h>
#include <prnn/detail/matrix/matrix.h>

#include <prnn/detail/parallel/synchronization.h>

// Standard Library Includes
#include <random>

namespace prnn
{
namespace matrix
{
namespace detail
{

class RandomGeneratorState
{
public:
    ~RandomGeneratorState()
    {

    }


public:
    std::default_random_engine& getCpuEngine()
    {
        _load();

        return *_cpuEngine;
    }

    CurandLibrary::curandGenerator_t getCudaEngine()
    {
        _load();

        return *_cudaEngine;
    }

private:
    void _load()
    {
        if(!_cpuEngine)
        {
            _cpuEngine.reset(new std::default_random_engine);
        }

        CurandLibrary::load();

        if(!_cudaEngine && CurandLibrary::loaded())
        {
            _cudaEngine.reset(new CurandLibrary::curandGenerator_t);

            CurandLibrary::curandCreateGenerator(_cudaEngine.get(),
                CurandLibrary::CURAND_RNG_PSEUDO_DEFAULT);
        }
    }

private:
    std::unique_ptr<std::default_random_engine>       _cpuEngine;
    std::unique_ptr<CurandLibrary::curandGenerator_t> _cudaEngine;

};

RandomGeneratorState randomGeneratorState;


}

void srand(size_t seed)
{
    if(CurandLibrary::loaded())
    {
        CurandLibrary::curandSetPseudoRandomGeneratorSeed(
            detail::randomGeneratorState.getCudaEngine(), seed);
    }
    else
    {
        detail::randomGeneratorState.getCpuEngine().seed(seed);
    }
}

void rand(Matrix& result)
{
    if(result.isContiguous() && CurandLibrary::loaded())
    {
        parallel::setNotSynchronized();

        if(result.precision() == SinglePrecision())
        {
            CurandLibrary::curandGenerateUniform(detail::randomGeneratorState.getCudaEngine(),
                static_cast<float*>(result.data()), result.elements());
        }
        else if(result.precision() == DoublePrecision())
        {
            CurandLibrary::curandGenerateUniformDouble(
                detail::randomGeneratorState.getCudaEngine(),
                static_cast<double*>(result.data()), result.elements());
        }
        else
        {
            throw std::runtime_error("Rand not implemented for this precision.");
        }
    }
    else
    {
        if(result.precision() == SinglePrecision())
        {
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::randomGeneratorState.getCpuEngine());
            }
        }
        else if(result.precision() == DoublePrecision())
        {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::randomGeneratorState.getCpuEngine());
            }
        }
        else
        {
            throw std::runtime_error("Rand not implemented for this precision.");
        }

    }
}

void randn(Matrix& result)
{
    if(result.isContiguous() && CurandLibrary::loaded())
    {
        parallel::setNotSynchronized();

        if(result.precision() == SinglePrecision())
        {
            CurandLibrary::curandGenerateNormal(detail::randomGeneratorState.getCudaEngine(),
                static_cast<float*>(result.data()), result.elements(), 0.0f, 1.0f);
        }
        else if(result.precision() == DoublePrecision())
        {
            CurandLibrary::curandGenerateNormalDouble(detail::randomGeneratorState.getCudaEngine(),
                static_cast<double*>(result.data()), result.elements(), 0.0, 1.0);
        }
        else
        {
            throw std::runtime_error("Rand not implemented for this precision.");
        }
    }
    else
    {
        if(result.precision() == SinglePrecision())
        {
            std::normal_distribution<float> distribution;

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::randomGeneratorState.getCpuEngine());
            }
        }
        else if(result.precision() == DoublePrecision())
        {
            std::normal_distribution<double> distribution;

            for(auto i = result.begin(); i != result.end(); ++i)
            {
                *i = distribution(detail::randomGeneratorState.getCpuEngine());
            }
        }
        else
        {
            throw std::runtime_error("Rand not implemented for this precision.");
        }

    }

}

Matrix rand(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    rand(result);

    return result;
}

Matrix randn(const Dimension& size, const Precision& precision)
{
    Matrix result(size, precision);

    randn(result);

    return result;
}

}
}


