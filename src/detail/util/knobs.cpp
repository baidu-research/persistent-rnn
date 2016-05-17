/*  \file   Knobs.cpp
    \brief  The source file for the Knob class.
*/

// Persistent RNN Includes
#include <prnn/detail/util/knobs.h>
#include <prnn/detail/util/knob_file.h>

#include <prnn/detail/util/system_compatibility.h>

// Standard Library Includes
#include <stdexcept>
#include <map>

namespace prnn
{

namespace util
{

class KnobDatabaseImplementation
{
public:
    typedef std::map<std::string, std::string> StringMap;

public:
    KnobDatabaseImplementation();

private:
    void loadKnobFiles();

public:
    StringMap knobs;
};

static KnobDatabaseImplementation database;

KnobDatabaseImplementation::KnobDatabaseImplementation()
{
    loadKnobFiles();
}

void KnobDatabaseImplementation::loadKnobFiles()
{
    // Check for an environment variable
    if(isEnvironmentVariableDefined("PRNN_KNOB_FILE"))
    {
        KnobFile knobFile(getEnvironmentVariable("PRNN_KNOB_FILE"));

        knobFile.loadKnobs();
    }
}

void KnobDatabase::addKnob(const std::string& name, const std::string& value)
{
    database.knobs[name] = value;
}

void KnobDatabase::setKnob(const std::string& name, const std::string& value)
{
    database.knobs[name] = value;
}

bool KnobDatabase::knobExists(const std::string& knobname)
{
    return database.knobs.count(knobname) != 0;
}

std::string KnobDatabase::getKnobValueAsString(const std::string& knobname)
{
    auto knob = database.knobs.find(knobname);

    if(knob == database.knobs.end())
    {
        throw std::runtime_error("Attempted to use uniniatilized knob '" +
            knobname + "'");
    }

    return knob->second;
}

std::string KnobDatabase::getKnobValue(const std::string& knobname,
    const std::string& defaultValue)
{
    if(!knobExists(knobname))
    {
        return defaultValue;
    }

    return getKnobValueAsString(knobname);
}

}

}






