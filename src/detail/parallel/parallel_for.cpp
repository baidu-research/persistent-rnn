
// Persistent RNN Includes
#include <prnn/detail/parallel/parallel_for.h>
#include <prnn/detail/parallel/concurrent_collectives.h>
#include <prnn/detail/parallel/synchronization.h>

// Standard Library Includes
#include <thread>
#include <list>

namespace prnn
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function)
{
    typedef std::list<std::thread> ThreadList;

    size_t threadCount = std::thread::hardware_concurrency();

    ThreadList threads;

    for(size_t i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(std::thread(function, ThreadGroup(threadCount, i)));
    }

    // barrier threads
    for(auto& thread : threads)
    {
        thread.join();
    }

    synchronize();
}

}
}






