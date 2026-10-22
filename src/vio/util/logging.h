// SPDX-License-Identifier: MIT
/**
 * @file      logging.h
 * @brief     Console logging and option checks.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <utility>

namespace vio {

// Creates the console logger ("[I] message") once; every log call makes sure it exists.
inline void InitLogging() {
  static const bool initialized = [] {
    auto console = spdlog::get("vio");
    if (!console) {
      console = spdlog::stdout_color_mt("vio");
      console->set_pattern("%^[%L]%$ %v");
      console->set_level(spdlog::level::info);
    }
    spdlog::set_default_logger(console);
    return true;
  }();
  (void)initialized;
}

template <typename... Args>
void LogInfo(spdlog::format_string_t<Args...> format, Args&&... args) {
  InitLogging();
  spdlog::info(format, std::forward<Args>(args)...);
}

template <typename... Args>
void LogWarn(spdlog::format_string_t<Args...> format, Args&&... args) {
  InitLogging();
  spdlog::warn(format, std::forward<Args>(args)...);
}

template <typename... Args>
void LogError(spdlog::format_string_t<Args...> format, Args&&... args) {
  InitLogging();
  spdlog::error(format, std::forward<Args>(args)...);
}

}  // namespace vio

// Python-style formatting: LOG_INFO("{} features", count)
#define LOG_INFO(...) ::vio::LogInfo(__VA_ARGS__)
#define LOG_WARN(...) ::vio::LogWarn(__VA_ARGS__)
#define LOG_ERROR(...) ::vio::LogError(__VA_ARGS__)

// For XxxOptions::Check(): logs the violated condition and returns false
#define REQUIRE_OPTION(condition)                                                  \
  do {                                                                             \
    if (!(condition)) {                                                            \
      LOG_ERROR("Invalid option: {} ({}:{})", #condition, __FILE__, __LINE__);    \
      return false;                                                                \
    }                                                                              \
  } while (false)
