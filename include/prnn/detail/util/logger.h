
#pragma once

#include <ostream>
#include <sstream>
#include <string>

namespace prnn {
namespace util {

/// Test if a specified log is enabled.
bool is_log_enabled(const std::string& name);

/// Enable all logs.
void enable_all_logs();

/// Enable the specified log.
void enable_log(const std::string& name);

/// Disable the specified log.
void disable_log(const std::string& name);

/// Enable a comma separated list of logs.
void enable_specific_logs(const std::string& modules);

/** \brief Create a logger for a specific log stream.
 *
 *  The logger accepts stream input and writes to std::cout. It is thread safe.
 *
 *  Example usage:
 *  \code
 *      log("Profiler") << "GEMM time = " << gemm_time << "sec" << std::endl;
 *  \endcode
 */
class log {
public:
    /// Construct log object for log stream `name`.
    explicit log(const std::string& name);

    /// Writes aggregated log to std::cout.
    ~log();

    /// Stream insertion for any type.
    template <typename T>
    log& operator<<(T const &value) {
        if (is_enabled_)
            buffer_ << value;
        return *this;
    }

    /// Stream insertion for std::endl.
    log& operator<<(std::ostream& (*f)(std::ostream&));

private:
    /// Buffer to aggregate log before writing to std::cout.
    std::ostringstream buffer_;

    /// Whether current log stream is enabled.
    bool is_enabled_;
};

}
}


