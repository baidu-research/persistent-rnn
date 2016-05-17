
#pragma once

#include <string>
#include <vector>

namespace prnn
{
namespace util
{

typedef std::vector<std::string> StringVector;

inline StringVector split(const std::string& string, const std::string& delimiter)
{
    size_t begin = 0;
    size_t end   = 0;

    StringVector strings;

    while (end != std::string::npos) {
        end = string.find(delimiter, begin);

        if (end > begin) {

            std::string substring = string.substr(begin, end - begin);

            if(!substring.empty()) strings.push_back(substring);
        }

        begin = end + delimiter.size();
    }

    return strings;
}

inline bool isWhitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline std::string removeWhitespace(const std::string& string)
{
    std::string result;

    auto begin = string.begin();

    for( ; begin != string.end(); ++begin)
    {
        if(!isWhitespace(*begin)) break;
    }

    auto end = string.end();

    if(end != begin)
    {
        --end;

        for( ; end != begin; --end)
        {
            if(!isWhitespace(*end))
            {
                break;
            }
        }

        ++end;
    }

    return std::string(begin, end);
}

inline std::string strip(const std::string& string, const std::string& delimiter)
{
    std::string result;
    size_t begin = 0;
    size_t end = 0;

    while(end != std::string::npos)
    {
        end = string.find(delimiter, begin);
        result += string.substr(begin, end - begin);
        begin = end + delimiter.size();
    }

    return result;

}

}
}




