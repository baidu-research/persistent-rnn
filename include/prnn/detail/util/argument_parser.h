
#pragma once

// Standard Library Includes
#include <string>
#include <sstream>

namespace prnn
{
namespace util
{

/*! \brief A class that can be used to parse arguments from argv and argc. */
class ArgumentParser
{

public:
    /*! The constructor used to initialize an argument parser */
    inline ArgumentParser(int argc, char** argv);

    /*! The constructor used to initialize an argument parser
        with a description. */
    inline ArgumentParser(int argc, char** argv, const std::string& description);

    /*! \brief Set the description for the program. */
    inline void set_description(const std::string& d);


public:
    /*! \brief Parse a bool from the command line

        \param identifier a string used to match
            strings in argv.
        \param b A bool to set to true if the argument is found,
            false otherwise
        \param string The help message to print out when the
            help function is called

    */
    inline void parse(const std::string& identifier, bool& b, bool starting,
        const std::string& string);

    /*! \brief Parse a bool from the command line

        \param identifier a string used to match
            strings in argv.
        \param longIdentifier A long string used to match strings in
            argv.  Must begin with "--".
        \param b A bool to set to true if the argument is found,
            false otherwise
        \param string The help message to print out when the
            help function is called

    */
    inline void parse(const std::string& identifier,
        const std::string& longIdentifier, bool& b, bool starting,
        const std::string& string);

    /*! \brief Parse an argument from the command line

        \param identifier A string used to match strings in argv. Must
            be a '-' followed by a single character or blank
        \param i A refernce to an argument to set to the parsed value
        \param starting The value to assign to i if the identifier is
            not found in argv
        \param string The help message to print out when the help
            function is called

    */
    template< class T, class V >
    void parse(const std::string& identifier, T& i, const V& starting,
        const std::string& string);

    /*! \brief Parse a long argument from the command line

        \param identifier A string used to match strings in argv. Must
            be a '-' followed by a single character or blank
        \param longIdentifier A long string used to match strings in
            argv.  Must begin with "--".
        \param i A refernce to an argument to set to the parsed value
        \param starting The value to assign to i if the identifier is
            not found in argv
        \param string The help message to print out when the help
            function is called

    */
    template< class T, class V >
    void parse(const std::string& identifier,
        const std::string& longIdentifier, T& i, const V& starting,
        const std::string& string);

    /*! \brief Parse a long argument from the command line

        \param identifier A string used to match strings in argv. Must
            be a '-' followed by a single character or blank
        \param longIdentifier A long string used to match strings in
            argv.  Must begin with "--".
        \param starting The value to assign to i if the identifier is
            not found in argv
        \param string The help message to print out when the help
            function is called

        \return The parsed value.

    */
    template< class T, class V >
    T parse(const std::string& identifier,
        const std::string& longIdentifier, const V& starting,
        const std::string& string);

    /*! \brief Create a help message describing the program.
        \return A help message stored in a string.
    */
    inline std::string help() const;

    /*! \brief Signal that there will be no more rules added.  It
            is now safe to search for help messages.
    */
    inline void parse();

private:
    /*! A function for parsing an argument
        \param identifier a string used to
            match strings in argv.
        \param value The value to set if the identifier is found

    */
    template<class T>
    void find_( const std::string& identifier, T& value );

    /*! A function determining if an argument is present
        \param identifier a string used to
            match strings in argv.
        \return true if the identifier was found in argv false otherwise

    */
    inline bool is_present_(const std::string& identifier);

    /*! Add a description of a specified argument.

        \param identifier The name of the argument.
        \param value      The initial value of the argument.
        \param message    The help message for the argument.
        \param boolalpha  Treat the value as a bool.
    */
    template<typename T>
    void add_argument_message_(const std::string& identifier,
        const T& value, const std::string& message, bool boolalpha);

private:
    /*! number of strings contained in argv */
    int argc_;
    /*! pointer to character array of arguments */
    char** argv_;

    /*! A string to hold the descriptions of arguments */
    std::stringstream arguments_;

    /*! \brief A string to hold the description of the program. */
    std::string description_;

private:
    const unsigned int MESSAGE_OFFSET = 22;
    const unsigned int SCREEN_WIDTH   = 80;

};

}
}

#include "argument_parser.inl"


