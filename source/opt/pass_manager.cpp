// Copyright (c) 2016 Google Inc.
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

#include "pass_manager.h"

#include <iostream>
#include <vector>

#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#elif defined(SPIRV_WINDOWS)
#endif

#include "ir_context.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {

namespace opt {

Pass::Status PassManager::Run(ir::IRContext* context) {
  auto status = Pass::Status::SuccessWithoutChange;

  // If print_all_stream_ is not null, prints the disassembly of the module
  // to that stream, with the given preamble and optionally the pass name.
  auto print_disassembly = [&context, this](const char* preamble, Pass* pass) {
    if (print_all_stream_) {
      std::vector<uint32_t> binary;
      context->module()->ToBinary(&binary, false);
      SpirvTools t(SPV_ENV_UNIVERSAL_1_2);
      std::string disassembly;
      t.Disassemble(binary, &disassembly, 0);
      *print_all_stream_ << preamble << (pass ? pass->name() : "") << "\n"
                         << disassembly << std::endl;
    }
  };

#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)
  double (*get_time_difference)(const timeval&, const timeval&) = NULL;
  if (ftime_report_stream_) {
    *ftime_report_stream_<< std::setw(30) << "Pass Name"
      << std::setw(16) << "CPU time"
      << std::setw(16) << "WALL time"
      << std::setw(16) << "SYS time"
      << std::setw(16) << "Pagefaults"
      << std::setw(16) << "RSS" << std::endl;
    get_time_difference = [](const timeval& before, const timeval& after) -> double {
      return static_cast<double>(after.tv_sec - before.tv_sec) +
        static_cast<double>(after.tv_usec - before.tv_usec) * .000001;
    };
  }
#endif
  for (const auto& pass : passes_) {
    print_disassembly("; IR before pass ", pass.get());
#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)
    bool usage_fail = false;
    rusage usage_before;
    timeval wall_before;
    if (ftime_report_stream_) {
      if (getrusage(RUSAGE_CHILDREN, &usage_before) == -1) {
        *ftime_report_stream_ << std::setw(30) << (pass ? pass->name() : "")
          << " ERROR: calling getrusage() fails";
        usage_fail = true;
      } else if (gettimeofday(&wall_before, NULL) == -1) {
        *ftime_report_stream_ << std::setw(30) << (pass ? pass->name() : "")
          << " ERROR: calling gettimeofday() fails";
        usage_fail = true;
      }
    }
#endif
    const auto one_status = pass->Run(context);
#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || \
    defined(SPIRV_FREEBSD)
    if (ftime_report_stream_ && !usage_fail) {
      rusage usage_after;
      timeval wall_after;
      if (getrusage(RUSAGE_CHILDREN, &usage_after) == -1) {
        *ftime_report_stream_ << std::setw(30) << (pass ? pass->name() : "")
          << " ERROR: calling getrusage() fails";
      } else if (gettimeofday(&wall_after, NULL) == -1) {
        *ftime_report_stream_ << std::setw(30) << (pass ? pass->name() : "")
          << " ERROR: calling gettimeofday() fails";
      } else {
        *ftime_report_stream_ << std::setw(30) << (pass ? pass->name() : "")
          << std::setw(16) << get_time_difference(usage_before.ru_utime, usage_after.ru_utime)
          << std::setw(16) << get_time_difference(wall_before, wall_after)
          << std::setw(16) << get_time_difference(usage_before.ru_stime, usage_after.ru_stime)
          << std::setw(16) << ((usage_after.ru_minflt - usage_before.ru_minflt)
              + (usage_after.ru_majflt - usage_before.ru_majflt))
          << std::setw(16) << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
      }
    }
#endif
    if (one_status == Pass::Status::Failure) return one_status;
    if (one_status == Pass::Status::SuccessWithChange) status = one_status;
  }
  print_disassembly("; IR after last pass", nullptr);

  // Set the Id bound in the header in case a pass forgot to do so.
  //
  // TODO(dnovillo): This should be unnecessary and automatically maintained by
  // the IRContext.
  if (status == Pass::Status::SuccessWithChange) {
    context->module()->SetIdBound(context->module()->ComputeIdBound());
  }
  passes_.clear();
  return status;
}

}  // namespace opt
}  // namespace spvtools
