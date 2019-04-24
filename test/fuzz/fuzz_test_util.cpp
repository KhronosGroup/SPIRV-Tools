// Copyright (c) 2019 Google LLC
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

#include <iostream>

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {

void CheckEqual(const spv_target_env env,
                const std::vector<uint32_t>& expected_binary,
                const std::vector<uint32_t>& actual_binary) {
  if (expected_binary != actual_binary) {
    SpirvTools t(env);
    std::string expected_disassembled;
    std::string actual_disassembled;
    ASSERT_TRUE(t.Disassemble(expected_binary, &expected_disassembled,
                              kFuzzDisassembleOption));
    ASSERT_TRUE(t.Disassemble(actual_binary, &actual_disassembled,
                              kFuzzDisassembleOption));
    ASSERT_EQ(expected_disassembled, actual_disassembled);
  }
}

void CheckEqual(const spv_target_env env, const std::string& expected_text,
                const std::vector<uint32_t>& actual_binary) {
  std::vector<uint32_t> expected_binary;
  SpirvTools t(env);
  ASSERT_TRUE(t.Assemble(expected_text, &expected_binary, kFuzzAssembleOption));
  CheckEqual(env, expected_binary, actual_binary);
}

void CheckEqual(const spv_target_env env, const std::string& expected_text,
                const opt::IRContext* actual_ir) {
  std::vector<uint32_t> actual_binary;
  actual_ir->module()->ToBinary(&actual_binary, false);
  CheckEqual(env, expected_text, actual_binary);
}

void CheckEqual(const spv_target_env env, const opt::IRContext* ir_1,
                const opt::IRContext* ir_2) {
  std::vector<uint32_t> binary_1;
  ir_1->module()->ToBinary(&binary_1, false);
  std::vector<uint32_t> binary_2;
  ir_2->module()->ToBinary(&binary_2, false);
  CheckEqual(env, binary_1, binary_2);
}

void CheckValid(spv_target_env env, const opt::IRContext* ir) {
  std::vector<uint32_t> binary;
  ir->module()->ToBinary(&binary, false);
  SpirvTools t(env);
  ASSERT_TRUE(t.Validate(binary));
}

std::string ToString(spv_target_env env, const opt::IRContext* ir) {
  std::vector<uint32_t> binary;
  ir->module()->ToBinary(&binary, false);
  SpirvTools t(env);
  std::string result;
  t.Disassemble(binary, &result, kFuzzDisassembleOption);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
