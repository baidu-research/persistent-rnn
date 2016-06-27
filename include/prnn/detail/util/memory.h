
#pragma once

// Standard Library Includes
#include <memory>

namespace std
{

// TODO: remove this when c++14 support is widespread
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args)
{
    return std::unique_ptr<T>(new T( std::forward<Args>(args)... ));
}

}


