// Copyright (c) 2021 ZHOU He
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

#ifndef SOURCE_OPT_REMOVE_UNUSED_INTERFACE_VARIABLES_PASS_H_
#define SOURCE_OPT_REMOVE_UNUSED_INTERFACE_VARIABLES_PASS_H_

#include "source/opt/pass.h"
namespace spvtools {
namespace opt {

class RemoveUnusedInterfaceVariablesPass : public Pass {
 public:
  explicit RemoveUnusedInterfaceVariablesPass(bool preserve_interface = false)
      : preserve_interface_(preserve_interface) {}

  const char* name() const override {
    return "remove-unused-interface-variables-pass";
  }
  Status Process() override;

 private:
  // Preserve entry point interface if true. All variables in interface
  // will not be eliminated. This mode is needed by GPU-Assisted Validation
  // instrumentation where a change in the interface is not allowed.
  bool preserve_interface_;
};
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REMOVE_UNUSED_INTERFACE_VARIABLES_PASS_H_