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

#ifndef SPIRV_TOOLS_LIBSPIRV_HPP_
#define SPIRV_TOOLS_LIBSPIRV_HPP_

#include <memory>
#include <string>
#include <vector>

#include "module.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {

// C++ interface for SPIRV-Tools functionalities. It wraps the context
// (including target environment and the corresponding SPIR-V grammar) and
// provides methods for assembling, disassembling, validating, and optimizing.
//
// Instances of this class are thread-safe.
class SpvTools {
 public:
  // Creates an instance targeting the given environment |env|.
  SpvTools(spv_target_env env) : context_(spvContextCreate(env)) {}

  ~SpvTools() { spvContextDestroy(context_); }

  // TODO(antiagainst): handle error message in the following APIs.

  // Assembles the given assembly |text| and writes the result to |binary|.
  // Returns SPV_SUCCESS on successful assembling.
  spv_result_t Assemble(const std::string& text, std::vector<uint32_t>* binary);

  // Disassembles the given SPIR-V |binary| with the given options and returns
  // the assembly. By default the options are set to generate assembly with
  // friendly variable names and no SPIR-V assembly header. Returns SPV_SUCCESS
  // on successful disassembling.
  spv_result_t Disassemble(
      const std::vector<uint32_t>& binary, std::string* text,
      uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                         SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);

  // Builds and returns a Module from the given SPIR-V |binary|.
  std::unique_ptr<ir::Module> BuildModule(const std::vector<uint32_t>& binary);

  // Builds and returns a Module from the given SPIR-V assembly |text|.
  std::unique_ptr<ir::Module> BuildModule(const std::string& text);

 private:
  // Context for the current invocation. Thread-safety of this class depends on
  // the constness of this field.
  spv_context context_;
};

}  // namespace spvtools

#endif  // SPIRV_TOOLS_LIBSPIRV_HPP_
