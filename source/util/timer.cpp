// Copyright (c) 2018 Google LLC.
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

#include "util/timer.h"

#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <sys/time.h>

namespace spvutils {

#if defined(SPIRV_TIMER_ENABLED)

namespace {
inline double TimeDifference(const timeval& before, const timeval& after) {
  return static_cast<double>(after.tv_sec - before.tv_sec) +
         static_cast<double>((after.tv_usec - before.tv_usec) / 10000) * .01;
}
}  // namespace

void TimerPrintDescription(std::ostream* out) {
  if (out) {
    *out << std::setw(30) << "PASS name" << std::setw(12) << "USR time"
         << std::setw(12) << "WALL time" << std::setw(12) << "SYS time"
#if defined(SPIRV_MEMORY_MEASUREMENT_ENABLED)
         << std::setw(12) << "RSS" << std::setw(12) << "Pagefault"
#endif  // defined(SPIRV_MEMORY_MEASUREMENT_ENABLED)
         << std::endl;
  }
}

void Timer::Start() {
  if (report_stream_) {
    if (getrusage(RUSAGE_SELF, &usage_before) == -1) {
      usage_status = kGetrusageFail;
    } else if (gettimeofday(&wall_before, NULL) == -1) {
      usage_status = kGettimeofdayFail;
    }
  }
}

void Timer::Stop() {
  if (report_stream_ && usage_status == kSucceeded) {
    if (getrusage(RUSAGE_SELF, &usage_after) == -1) {
      usage_status = kGetrusageFail;
    } else if (gettimeofday(&wall_after, NULL) == -1) {
      usage_status = kGettimeofdayFail;
    }
  }
}

void Timer::Report(const char* tag) {
  if (!report_stream_)
    return;

  switch (usage_status) {
    case kGetrusageFail:
      *report_stream_ << std::setw(30) << tag
                      << " ERROR: calling getrusage() fails";
      return;
    case kGettimeofdayFail:
      *report_stream_ << std::setw(30) << tag
                      << " ERROR: calling gettimeofday() fails";
      return;
    default:
      break;
  }

  report_stream_->precision(2);
  *report_stream_ << std::fixed << std::setw(30) << tag << std::setw(12)
                  << TimeDifference(usage_before.ru_utime,
                                    usage_after.ru_utime)
                  << std::setw(12) << TimeDifference(wall_before, wall_after)
                  << std::setw(12)
                  << TimeDifference(usage_before.ru_stime,
                                    usage_after.ru_stime)
#if defined(SPIRV_MEMORY_MEASUREMENT_ENABLED)
                  << std::setw(12)
                  << (usage_after.ru_maxrss - usage_before.ru_maxrss)
                  << std::setw(12)
                  << ((usage_after.ru_minflt - usage_before.ru_minflt) +
                      (usage_after.ru_majflt - usage_before.ru_majflt))
#endif  // defined(SPIRV_MEMORY_MEASUREMENT_ENABLED)
                  << std::endl;
}

void Timer::StopAndReport(const char* tag) {
  Stop();
  Report(tag);
}

#endif  // defined(SPIRV_TIMER_ENABLED)

}  // namespace spvutils
