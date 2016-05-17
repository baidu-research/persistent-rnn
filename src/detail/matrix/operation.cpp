/*    \file   Operation.cpp
    \brief  The source file for the Operation classes.
*/

// Persistent RNN Includes
#include <prnn/detail/matrix/operation.h>

namespace prnn
{
namespace matrix
{

bool Operation::operator==(const Operation& o) const
{
    return o._type == _type;
}

}
}

