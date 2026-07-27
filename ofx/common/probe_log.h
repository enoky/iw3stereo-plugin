// Tiny append-only logger for the Phase 0 probe plugins.
//
// Resolve gives no console, so everything the probe learns goes to a file:
//   %LOCALAPPDATA%\iw3probe\probe.log
// Delete the file to start a fresh run.

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>

namespace probe
{
inline std::string logPath()
{
    const char* base = std::getenv("LOCALAPPDATA");
    std::string dir = base ? std::string(base) + "\\iw3probe" : std::string("C:\\iw3probe");
    std::string cmd = "mkdir \"" + dir + "\" >nul 2>&1";
    static bool made = false;
    if (!made)
    {
        std::system(cmd.c_str());
        made = true;
    }
    return dir + "\\probe.log";
}

inline void logf(const char* fmt, ...)
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static FILE* fp = nullptr;
    if (!fp)
    {
        fp = std::fopen(logPath().c_str(), "a");
        if (!fp)
        {
            return;
        }
    }

    // Timestamped so a run can be told apart from the previous one: the file is
    // append-only and stays locked while Resolve is up, so it cannot be cleared
    // between attempts.
    std::time_t now = std::time(nullptr);
    std::tm parts{};
    localtime_s(&parts, &now);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &parts);
    std::fprintf(fp, "%s ", stamp);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(fp, fmt, args);
    va_end(args);
    std::fputc('\n', fp);
    std::fflush(fp);
}
}  // namespace probe
