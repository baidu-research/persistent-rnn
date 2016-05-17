/*! \file   KnobFile.cpp
    \brief  The source file for the KnobFile class.
*/

// Persistent RNN Includes
#include <prnn/detail/util/knob_file.h>
#include <prnn/detail/util/string.h>
#include <prnn/detail/util/knobs.h>


// Standard Library Includes
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace prnn
{

namespace util
{

KnobFile::KnobFile(const std::string& filename)
: _filename(filename)
{

}

static std::string getLine(std::istream& stream);
static void parseKnobFileLine(const std::string& line);

void KnobFile::loadKnobs() const
{
    std::ifstream stream(_filename);

    if(!stream.is_open())
    {
        throw std::runtime_error("Failed to open knob file '" +
            _filename + "' for reading.");
    }

    while(stream.good())
    {
        auto line = getLine(stream);

        parseKnobFileLine(line);
    }
}

std::string getLine(std::istream& stream)
{
    typedef std::vector<uint8_t> ByteVector;

    ByteVector bytes;

    while(stream.good())
    {
        uint8_t byte = stream.get();

        if(!stream.good()) break;

        if(byte == '\n') break;

        bytes.push_back(byte);
    }

    return std::string(bytes.begin(), bytes.end());
}

void parseKnobFileLine(const std::string& line)
{
    auto stripped = removeWhitespace(line);

    // discard empty lines
    if(stripped.empty()) return;

    // discard comments
    if(stripped[0] == '#') return;

    auto components = split(stripped, "=");

    // Knobs should be "key = value"
    // TODO: handle '=' in strings
    if(components.size() != 2)
    {
        throw std::runtime_error(
            "Invalid syntax in knob file line '" + line + "'");
    }

    KnobDatabase::addKnob(removeWhitespace(components[0]),
        removeWhitespace(components[1]));
}

}

}

