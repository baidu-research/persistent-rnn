
#pragma once

// Standard Library Includes
#include <functional>

// Forward Declarations
namespace lucius { namespace parallel { class ThreadGroup; } }

namespace lucius
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function);

}
}





