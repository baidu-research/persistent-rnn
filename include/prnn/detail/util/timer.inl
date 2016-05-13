#pragma once

#include "timer.h"
#include "timestamp.h"

#include <sstream>

namespace prnn {
namespace util {

inline Timer::Timer()
: beginning_(0), ending_(0), beginning_seconds_(0.0), ending_seconds_(0.0), running_(false)
{

}

inline void Timer::start() {

    beginning_ = support::timestamp();

    beginning_seconds_ = (beginning_ + 0.0) * 1.0e-9;

    running_ = true;
}

inline void Timer::stop() {

    ending_ = support::timestamp();

    ending_seconds_ = (ending_ + 0.0) * 1.0e-9;

    running_ = false;
}

inline Timer::Cycle Timer::cycles() const {

    if(running_) {
        return (support::timestamp() - beginning_);
    }
    else {
        return (ending_ - beginning_);
    }
}

inline Timer::Second Timer::start_seconds() const {
    return beginning_seconds_;
}

inline Timer::Second Timer::seconds() const {
    if(running_) {
        return (((support::timestamp() + 0.0) * 1.0e-9) - beginning_seconds_);
    }
    else {
        return ending_seconds_ - beginning_seconds_;
    }
}

inline Timer::Second Timer::absolute() const {

    if(running_) {
        return (((support::timestamp() + 0.0) * 1.0e-9));
    }
    else {
        return ending_seconds_;
    }
}

inline std::string Timer::to_string() const {

    std::stringstream stream;

    stream << seconds() << "s (" << cycles() << " ns)";

    return stream.str();
}


inline bool Timer::is_running() const {
    return running_;
}

}
}


