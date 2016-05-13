
#pragma once

#include <string>
#include <vector>

namespace prnn
{
namespace util
{

typedef std::vector<std::string> StringVector;

inline StringVector split(const std::string& string, const std::string& delimiter);
inline std::string remove_whitespace(const std::string& string);
inline std::string strip(const std::string& string, const std::string& delimiter);
inline bool is_whitespace(char c);

}
}

#include "string.inl"


