
#pragma once

#include <cstdint>
#include <string>

namespace prnn {
namespace util {

class Timer
{
public:
    /*! \brief A type for seconds */
    typedef double Second;

    /*! \brief A type for representing clock ticks */
    typedef uint64_t Cycle;

public:
    /*! \brief The constructor initializes the private variables
        and makes sure that the Timer is not running.
    */
    inline Timer();

    /*! A function that is used to set beginning to the value of
        the hardware Timer
    */
    inline void start();

    /*! A function that is used to set ending to the value of
        the hardware Timer
    */
    inline void stop();

    /*! A function that is used to determine the number of clock cycles
        between the last time start was called and the last time that
        end was called.
        \return the difference between ending and beginning
    */
    inline Cycle cycles() const;

    /*! A function that is used to determine the number of seconds
        between the last time start was called and the last time that
        end was called.
        \return the difference between ending and beginning
    */
    inline Second seconds() const;

    /*! Get the absolute number of seconds elapsed since system
            start and the start time of the timer.
        \return the difference between system start and beginning
    */
    inline Second start_seconds() const;

    /*! \brief Get the absolute number of seconds elapsed since system
            start
        \return That time
    */
    inline Second absolute() const;

    /*! \brief Get a string representation of the current time */
    inline std::string to_string() const;

    /// Whether or not the timer is currently running
    inline bool is_running() const;

private:
    /*! An integer representing the value of the cycle counter when the
        last start() function was called
    */
    Cycle beginning_;

    /*! An integer representing the value of the cycle counter when the
        last stop() function was called
    */
    Cycle ending_;

    /*! A floating point number representing the value of the system
        clock when the last start() function was called
    */
    Second beginning_seconds_;

    /*! A floating point number representing the value of the system
        clock when the last stop() function was called
    */
    Second ending_seconds_;

    /*! Read a cycle counter using either assembly or an OS interface.
        \return a 64 bit value representing the current number of
        clock cycles since the last reset
    */
    static Cycle rdtsc();

    /*! \brief Is the Timer running? */
    bool running_;

};

}
}

#include "timer.inl"



