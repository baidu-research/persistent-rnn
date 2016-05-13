/*  \file   Knobs.h
    \brief  TThe header file for the Knob class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <sstream>

namespace prnn
{

namespace util
{

class KnobDatabase
{
public:
    template<typename T>
    static void setKnob(const std::string& name, const T& value)
    {
        std::stringstream stream;

        stream << value;

        setKnob(name, stream.str());
    }

public:
    static void addKnob(const std::string& name, const std::string& value);
    static void setKnob(const std::string& name, const std::string& value);

public:
    template<typename T>
    static T getKnobValue(const std::string& knobname);

    template<typename T>
    static T getKnobValue(const std::string& knobname, const T& defaultValue);

    static bool knobExists(const std::string& knobname);

    static std::string getKnobValue(const std::string& knobname,
        const std::string& defaultValue);

public:
    static std::string getKnobValueAsString(const std::string& knobname);
};

template<typename T>
T KnobDatabase::getKnobValue(const std::string& knobname)
{
    std::string string = getKnobValueAsString(knobname);
    std::stringstream stream(string);

    T value = 0;

    stream >> value;

    return value;
}

template<typename T>
T KnobDatabase::getKnobValue(const std::string& knobname, const T& defaultValue)
{
    if(!knobExists(knobname))
    {
        return defaultValue;
    }

    return getKnobValue<T>(knobname);
}

}

}






