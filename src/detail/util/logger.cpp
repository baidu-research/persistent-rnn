
// Persistent RNN Includes
#include <prnn/detail/util/logger.h>

#include <prnn/detail/util/timer.h>
#include <prnn/detail/util/string.h>

// Standard Library Includes
#include <mutex>
#include <string>
#include <unordered_set>
#include <iostream>
#include <memory>

namespace prnn
{
namespace util
{
namespace detail
{

class LogDatabase
{
public:
    LogDatabase() : enable_all(false) { timer.start(); }

    ~LogDatabase() {}

public:
    typedef std::unordered_set<std::string> StringSet;

public:
    bool      enable_all;
    StringSet enabled_logs;
    Timer     timer;

public:
    bool is_enabled(const std::string& log_name) const
    {
        return enable_all || (enabled_logs.count(log_name) != 0);
    }
};

std::unique_ptr<LogDatabase> the_database;

LogDatabase& get_database()
{
    if(!the_database)
    {
        the_database.reset(new LogDatabase);
    }

    return *the_database;
}

std::string logger_time() {
    std::stringstream stream;

    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream.precision(6);

    stream << get_database().timer.seconds();

    return stream.str();
}

} // namespace detail

std::mutex log_writer;

log::log(const std::string& name) : is_enabled_(false) {
    if (detail::get_database().is_enabled(name)) {
        buffer_ << "(" << detail::logger_time() << "): " << name << ": ";
        is_enabled_ = true;
    }
}

log::~log() {
    if (is_enabled_) {
        std::lock_guard<std::mutex> guard(log_writer);
        std::cout << buffer_.str();
    }
}

log& log::operator<<(std::ostream& (*f)(std::ostream&)) {
    if (is_enabled_)
        buffer_ << f;
    return *this;
}

bool is_log_enabled(const std::string& name) {
    return detail::get_database().is_enabled(name);
}

void enable_specific_logs(const std::string& modules) {
    auto individual_modules = split(modules, ",");

    for (auto& module : individual_modules) {
        enable_log(module);
    }
}

void enable_all_logs() {
    detail::get_database().enable_all = true;
}

void disable_log(const std::string& name) {
    detail::get_database().enabled_logs.erase(name);
}

void enable_log(const std::string& name) {
    detail::get_database().enabled_logs.insert(name);
}

}
}

