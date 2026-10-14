// Copyright (c) 2026 Google Inc.
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

#ifndef SOURCE_OPT_TRIM_VARIABLE_POINTERS_CAPABILITIES_PASS_H_
#define SOURCE_OPT_TRIM_VARIABLE_POINTERS_CAPABILITIES_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class TrimVariablePointersCapabilitiesPass : public Pass {
 public:
  TrimVariablePointersCapabilitiesPass() = default;
  TrimVariablePointersCapabilitiesPass(
      const TrimVariablePointersCapabilitiesPass&) = delete;
  TrimVariablePointersCapabilitiesPass(
      TrimVariablePointersCapabilitiesPass&&) = delete;

  const char* name() const override {
    return "trim-variable-pointers-capabilities";
  }
  Status Process() override;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_TRIM_VARIABLE_POINTERS_CAPABILITIES_PASS_H_
