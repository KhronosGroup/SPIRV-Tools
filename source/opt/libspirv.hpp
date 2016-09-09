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

#include "message.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {

// C++ interface for SPIRV-Tools functionalities. It wraps the context
// (including target environment and the corresponding SPIR-V grammar) and
// provides methods for assembling, disassembling, and validating.
//
// Instances of this class provide basic thread-safety guarantee.
class SpvTools {
 public:
  enum {
    // Default disassembling option used by Disassemble():
    // * Avoid prefix comments from decoding the SPIR-V module header, and
    // * Use friendly names for variables.
    kDefaultDisassembleOption = SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                                SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES
  };

  // Constructs an instance targeting the given environment |env|.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  explicit SpvTools(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  SpvTools(const SpvTools&) = delete;
  SpvTools(SpvTools&&) = delete;
  SpvTools& operator=(const SpvTools&) = delete;
  SpvTools& operator=(SpvTools&&) = delete;

  // Destructs this instance.
  ~SpvTools();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Assembles the given assembly |text| and writes the result to |binary|.
  // Returns true on successful assembling. |binary| will be kept untouched if
  // assembling is unsuccessful.
  bool Assemble(const std::string& text, std::vector<uint32_t>* binary) const;

  // Disassembles the given SPIR-V |binary| with the given |options| and writes
  // the assembly to |text|. Returns ture on successful disassembling. |text|
  // will be kept untouched if diassembling is unsuccessful.
  bool Disassemble(const std::vector<uint32_t>& binary, std::string* text,
                   uint32_t options = kDefaultDisassembleOption) const;

  // Validates the given SPIR-V |binary|. Returns true if no issues are found.
  // Otherwise, returns false and communicates issues via the message consumer
  // registered.
  bool Validate(const std::vector<uint32_t>& binary) const;

 private:
  struct Impl;  // Opaque struct for holding the data fields used by this class.
  std::unique_ptr<Impl> impl_;  // Unique pointer to implementation data.
};

}  // namespace spvtools

#endif  // SPIRV_TOOLS_LIBSPIRV_HPP_
