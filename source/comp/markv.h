// Copyright (c) 2017 Google Inc.
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

// MARK-V is a compression format for SPIR-V binaries. It strips away
// non-essential information (such as result ids which can be regenerated) and
// uses various bit reduction techiniques to reduce the size of the binary and
// make it more similar to other compressed SPIR-V files to further improve
// compression of the dataset.

#ifndef SPIRV_TOOLS_MARKV_HPP_
#define SPIRV_TOOLS_MARKV_HPP_

#include <string>
#include <vector>

#include "markv_model.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {

struct MarkvEncoderOptions {
  bool validate_spirv_binary = false;
};

struct MarkvDecoderOptions {
  bool validate_spirv_binary = false;
};

// Encodes the given SPIR-V binary to MARK-V binary.
// If |comments| is not nullptr, it would contain a textual description of
// how encoding was done (with snippets of disassembly and bit sequences).
spv_result_t SpirvToMarkv(spv_const_context context,
                          const std::vector<uint32_t>& spirv,
                          const MarkvEncoderOptions& options,
                          const MarkvModel& markv_model,
                          MessageConsumer message_consumer,
                          std::vector<uint8_t>* markv,
                          std::string* comments);

// Decodes a SPIR-V binary from the given MARK-V binary.
// If |comments| is not nullptr, it would contain a textual description of
// how decoding was done (with snippets of disassembly and bit sequences).
spv_result_t MarkvToSpirv(spv_const_context context,
                          const std::vector<uint8_t>& markv,
                          const MarkvDecoderOptions& options,
                          const MarkvModel& markv_model,
                          MessageConsumer message_consumer,
                          std::vector<uint32_t>* spirv,
                          std::string* comments);

}  // namespace spvtools

#endif  // SPIRV_TOOLS_MARKV_HPP_
