// Copyright (c) 2019 Google Inc.
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

#include <cstdint>
#include <cstring>  // memcpy
#include <vector>

#include "source/spirv_target_env.h"
#include "spirv-tools/libspirv.hpp"
#include "test/fuzzers/random_generator.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1) {
    return 0;
  }

  spvtools::fuzzers::RandomGenerator random_gen(data, size);
  const spv_context context = spvContextCreate(random_gen.GetTargetEnv());
  if (context == nullptr) {
    return 0;
  }

  std::vector<uint32_t> input;
  input.resize(size >> 2);
  size_t count = 0;
  for (size_t i = 0; (i + 3) < size; i += 4) {
    input[count++] = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) |
                     (data[i + 3]) << 24;
  }

  std::vector<char> input_str;
  size_t char_count = input.size() * sizeof(uint32_t) / sizeof(char);
  input_str.resize(char_count);
  memcpy(input_str.data(), input.data(), input.size() * sizeof(uint32_t));

  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  spvTextToBinaryWithOptions(context, input_str.data(), input_str.size(),
                             SPV_TEXT_TO_BINARY_OPTION_NONE, &binary,
                             &diagnostic);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    diagnostic = nullptr;
  }

  if (binary) {
    spvBinaryDestroy(binary);
    binary = nullptr;
  }

  spvTextToBinaryWithOptions(context, input_str.data(), input_str.size(),
                             SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS,
                             &binary, &diagnostic);
  if (diagnostic) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    diagnostic = nullptr;
  }

  if (binary) {
    spvBinaryDestroy(binary);
    binary = nullptr;
  }

  spvContextDestroy(context);

  return 0;
}
