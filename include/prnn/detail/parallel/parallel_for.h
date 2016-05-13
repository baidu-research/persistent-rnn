
#pragma once

// Standard Library Includes
#include <functional>

// Forward Declarations
namespace prnn { namespace parallel { class ThreadGroup; } }

namespace prnn
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function);

}
}





