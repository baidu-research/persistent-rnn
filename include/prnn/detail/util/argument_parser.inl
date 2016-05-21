
#pragma once

// Persistent RNN Includes
#include <prnn/detail/util/argument_parser.h>
#include <prnn/detail/util/string.h>

// Standard Library Includes
#include <iostream>
#include <cassert>

namespace prnn
{
namespace util
{

inline ArgumentParser::ArgumentParser(int argc, char** argv) :
    argc_(argc), argv_(argv)
{
    set_description("No description provided.");
}

inline ArgumentParser::ArgumentParser(int argc, char** argv, const std::string& description) :
    argc_(argc), argv_(argv)
{
    set_description(description);
}

template< typename T >
static void set_value(T& value, const std::string& s)
{
    std::stringstream stream(s);
    stream >> value;
}

static inline std::string format(const std::string& input,
    const std::string& firstPrefix, const std::string& prefix,
    size_t width)
{
    std::string word;
    std::string result = firstPrefix;
    size_t currentIndex = firstPrefix.size();

    for (auto character : input)
    {

        if (isWhitespace(character))
        {

            if(currentIndex + word.size() > width)
            {

                currentIndex = prefix.size();
                result += "\n";
                result += prefix;
            }

            if(!word.empty())
            {

                result += word + " ";
                ++currentIndex;
                word.clear();
            }
        }
        else
        {

            word.push_back(character);
            ++currentIndex;
        }
    }

    if (currentIndex + word.size() > width)
    {
        result += "\n";
        result += prefix;
    }

    result += word + "\n";
    return result;
}

template< typename T >
void ArgumentParser::find_(const std::string& identifier, T& value) {
    std::string str;

    bool found = false;

    // Search over all arguments for a match
    for (int i = 0; i < argc_; i++) {
        str = argv_[i];

        if (str.size() > 0) {
            if (str[0] == '-') {

                str = str.substr(1);
                if (str.size() > 0) {
                    if (str[0] == '-') {
                        str = str.substr(1);
                    }
                }
            }
            else {
                continue;
            }
        }

        size_t pos = str.find(identifier);

        if (pos == 0) {

            if (str.size() == identifier.size()) {
                if (i < argc_ - 1) {
                    found = true;
                    str = argv_[i+1];
                    break;
                }
            }
            else {
                pos = identifier.size();
                if (pos < str.size()) {
                    if (str[pos] == '=') {
                        ++pos;
                    }
                }
                if (identifier == str) {
                    found = true;
                    str = str.substr(pos);
                    break;
                }
            }
        }
    }

    if (found) {
        set_value(value, str);
    }

}

template< class T, class V >
void ArgumentParser::parse(const std::string& _identifier, T& i,
    const V& starting, const std::string& message) {
    assert(_identifier.size() == 2);
    assert(_identifier[0] == '-');

    i = starting;
    find_(_identifier.substr(1), i);

    std::string identifier(' ' + _identifier);

    add_argument_message_(identifier, i, message, false);
}

template< class T, class V >
void ArgumentParser::parse(const std::string& identifier,
    const std::string& longIdentifier, T& i,
    const V& starting, const std::string& string) {

    i = starting;

    if(!identifier.empty()) {
        assert(identifier.size() == 2);
        assert(identifier[0] == '-');
        find_(identifier.substr(1), i);
    }

    assert(longIdentifier.size() > 2);
    assert(0 == longIdentifier.find( "--" ));

    find_(longIdentifier.substr(2), i);

    std::string prefix(' ' + identifier
        + '(' + longIdentifier + ')');

    add_argument_message_(prefix, i, string, false);
}

template< class T, class V >
T ArgumentParser::parse(const std::string& identifier,
    const std::string& longIdentifier,
    const V& starting, const std::string& message) {

    T value;

    parse(identifier, longIdentifier, value, starting, message);

    return value;
}

inline void ArgumentParser::set_description(const std::string& d)
{
    std::string desc = " Description: ";
    std::stringstream stream(d);
    int repetition = MESSAGE_OFFSET - (int)desc.size();
    std::string prefix(std::max(repetition, 0), ' ');
    std::string regularPrefix(MESSAGE_OFFSET, ' ');

    description_ = format(stream.str(), desc + prefix,
        regularPrefix, SCREEN_WIDTH) + "\n";
}

inline bool ArgumentParser::is_present_(const std::string& identifier)
{
    bool found = false;

    for (int i = 0; i < argc_; i++)
    {
        std::string str = argv_[i];
        size_t pos = str.find(identifier);
        if (pos == 0 && str.size() == identifier.size()) {
            found = true;
            break;
        }
    }

    return found;
}

static inline void set_value(std::string& value, const std::string& s) {
    value = s;
}

inline std::string ArgumentParser::help() const {

    assert(argc_ > 0);

    std::stringstream stream;

    stream << "\nProgram : " << argv_[0] << "\n\n";
    stream << description_;
    stream << "Arguments : \n\n";
    stream << arguments_.str();
    stream << "\n";

    return stream.str();
}

inline void ArgumentParser::parse(const std::string& _identifier, bool& b,
    bool starting, const std::string& string) {

    assert(_identifier.size() == 2);
    assert(_identifier[0] == '-');

    if (is_present_(_identifier)) {
        b = !starting;
    }
    else {
        b = starting;
    }

    std::string identifier(' ' + _identifier);

    add_argument_message_(identifier, b, string, true);
}

inline void ArgumentParser::parse(const std::string& _identifier,
    const std::string& _longIdentifier, bool& b, bool starting,
    const std::string& string) {

    bool inFirst = false;

    if(!_identifier.empty()) {
        assert(_identifier.size() == 2);
        assert(_identifier[0] == '-');
        inFirst = is_present_( _identifier );
    }

    if(inFirst || is_present_(_longIdentifier)) {
        b = !starting;
    }
    else {
        b = starting;
    }

    std::string identifier( ' ' + _identifier + '('
        + _longIdentifier + ')' );

    add_argument_message_(identifier, b, string, true);
}

template<typename T>
inline void ArgumentParser::add_argument_message_(const std::string& identifier,
    const T& value, const std::string& message, bool boolalpha) {

    int prefixSpacing = MESSAGE_OFFSET - (int)identifier.size();

    std::string prefix(std::max(prefixSpacing, 0 ), ' ');
    std::string regularPrefix(MESSAGE_OFFSET, ' ');

    std::stringstream secondStream(message + '\n');

    std::string result = format(secondStream.str(), prefix,
        regularPrefix, SCREEN_WIDTH);

    std::stringstream thirdStream;
    thirdStream << result << regularPrefix << "value = ";

    if (boolalpha) {
        thirdStream << std::boolalpha;
    }

    thirdStream << value << "\n";

    arguments_ << identifier << thirdStream.str() << "\n";
}

inline void ArgumentParser::parse() {

    bool print_help = false;

    parse( "-h", "--help", print_help, false, "Print this help message." );

    if(print_help) {
        std::cout << help();
        std::exit(0);
    }
}

}
}

