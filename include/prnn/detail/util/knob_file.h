/*! \file   KnobFile.h
    \date   Sunday January 26, 2013
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  The header file for the KnobFile class.
*/

#pragma once

// Standard Library Includes
#include <string>

namespace prnn
{

namespace util
{

class KnobFile
{
public:
    KnobFile(const std::string& filename);

public:
    void loadKnobs() const;

private:
    std::string _filename;

};

}

}



