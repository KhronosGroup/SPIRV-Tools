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

#include <sys/resource.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include <string>

namespace spvutils {

#if defined(SPIRV_TIMER_ENABLED)

// Print the description of resource types measured by Timer class. If |out| is
// NULL, it does nothing. Otherwise, it prints resource types. The second is
// optional and if it is true, the function also prints resource type fields
// related to memory. Its default is false. In usual, this must be placed before
// calling Timer::Report() to inform what those fields printed by
// Timer::Report() indicate.
void PrintTimerDescription(std::ostream* out, bool measure_mem_usage) {
  if (out) {
    *out << std::setw(30) << "PASS name" << std::setw(12) << "CPU time"
         << std::setw(12) << "WALL time" << std::setw(12) << "USR time"
         << std::setw(12) << "SYS time";
    if (measure_mem_usage) {
      *out << std::setw(12) << "RSS" << std::setw(12) << "Pagefault";
    }
    *out << std::endl;
  }
}

// Do not change the order of invoking system calls. We want to make CPU/Wall
// time correct as much as possible. Calling functions to get CPU/Wall time must
// closely surround the target code of measuring.
void Timer::Start() {
  if (report_stream_) {
    if (getrusage(RUSAGE_SELF, &usage_before_) == -1) {
      usage_status_ = kClockGettimeFailed;
    } else if (clock_gettime(CLOCK_MONOTONIC, &wall_before_) == -1) {
      usage_status_ = kClockGettimeFailed;
    } else if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_before_) == -1) {
      usage_status_ = kGetrusageFailed;
    }
  }
}

// The order of invoking system calls is important with the same reason as
// Timer::Start().
void Timer::Stop() {
  if (report_stream_ && usage_status_ == kSucceeded) {
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_after_) == -1) {
      usage_status_ = kClockGettimeFailed;
    } else if (clock_gettime(CLOCK_MONOTONIC, &wall_after_) == -1) {
      usage_status_ = kClockGettimeFailed;
    } else if (getrusage(RUSAGE_SELF, &usage_after_) == -1) {
      usage_status_ = kGetrusageFailed;
    }
  }
}

void Timer::Report(const char* tag) {
  if (!report_stream_) return;

  switch (usage_status_) {
    case kGetrusageFailed:
      *report_stream_ << std::setw(30) << tag
                      << " ERROR: calling getrusage() fails";
      return;
    case kClockGettimeFailed:
      *report_stream_ << std::setw(30) << tag
                      << " ERROR: calling clock_gettime() fails";
      return;
    default:
      break;
  }

  report_stream_->precision(3);
  *report_stream_ << std::fixed << std::setw(30) << tag << std::setw(12)
                  << CPUTime() << std::setw(12) << WallTime() << std::setw(12)
                  << UserTime() << std::setw(12) << SystemTime();
  if (measure_mem_usage_) {
    *report_stream_ << std::fixed << std::setw(12) << RSS() << std::setw(12)
                    << PageFault();
  }
  *report_stream_ << std::endl;
}

#endif  // defined(SPIRV_TIMER_ENABLED)

}  // namespace spvutils
