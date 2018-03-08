// Copyright (c) 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Contains utils for getting resource utilization

#ifndef LIBSPIRV_UTIL_TIMER_H_
#define LIBSPIRV_UTIL_TIMER_H_

#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)

#define SPIRV_TIMER_ENABLED
#define SPIRV_TIMER_DESCRIPTION(out) spvutils::TimerPrintDescription(out)
#define SPIRV_TIMER_SCROPED(out, tag) \
  spvutils::ScropedTimer timer##__LINE__(out, tag)

#if defined(SPIRV_LINUX)
#define SPIRV_MEMORY_MEASUREMENT_ENABLED
#endif  // defined(SPIRV_LINUX)

#include <sys/resource.h>

namespace spvutils {

void TimerPrintDescription(std::ostream* out);

class Timer {
 public:
  Timer(std::ostream* out) : report_stream_(out), usage_fail(kNotFailed) {}
  void Start();
  void Stop();
  void Report(const char* tag);
  void StopAndReport(const char* tag);

 private:
  std::ostream* report_stream_;

  enum { kGetrusageFail, kGettimeofdayFail, kNotFailed } usage_fail;

  rusage usage_before;
  rusage usage_after;
  timeval wall_before;
  timeval wall_after;
};

class ScropedTimer : Timer {
 public:
  ScropedTimer(std::ostream* out, const char* tag) : Timer(out), tag_(tag) {
    Start();
  }
  ~ScropedTimer() { StopAndReport(tag_); }

 private:
  const char* tag_;
};

}  // namespace spvutils

#elif  // defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) ||
       // defined(SPIRV_MAC) || defined(SPIRV_FREEBSD)

#define SPIRV_TIMER_DESCRIPTION(out)
#define SPIRV_TIMER_SCROPED(out, tag)

#endif  // defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) ||
        // defined(SPIRV_MAC) || defined(SPIRV_FREEBSD)

#endif  // LIBSPIRV_UTIL_TIMER_H_
