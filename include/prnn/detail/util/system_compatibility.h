/*! \file SystemCompatibility.h
    \brief The header file for system specific code.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace prnn
{

namespace util
{

/*! \brief Get the number of hardware threads */
unsigned int getHardwareThreadCount();
/*! \brief Get the full path to the named executable */
std::string getExecutablePath(const std::string& executableName);
/*! \brief The amount of free physical memory */
long long unsigned int getFreePhysicalMemory();
/*! \brief Get an estimate of the max clock speed */
long long unsigned int getMaxClockSpeed();
/*! \brief Get an estimate of the number of fmas per clock per core */
long long unsigned int getFMAsPerClockPerCore();
/*! \brief Get an estimate of the machine Floating Point Operations per second */
long long unsigned int getMachineFlops();
/*! \brief Has there been an OpenGL context bound to this process */
bool isAnOpenGLContextAvailable();
/*! \brief Is a string name mangled? */
bool isMangledCXXString(const std::string& string);
/*! \brief Demangle a string */
std::string demangleCXXString(const std::string& string);
/*! \brief Get the value of an environment variable */
std::string getEnvironmentVariable(const std::string& string);
/*! \brief Is an environment variable defined? */
bool isEnvironmentVariableDefined(const std::string& name);


}

}


