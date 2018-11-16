// Copyright (c) 2018 Google LLC
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

#ifndef TEST_REDUCE_REDUCE_TEST_UTIL_H_
#define TEST_REDUCE_REDUCE_TEST_UTIL_H_

#include "gtest/gtest.h"

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_opportunity.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace reduce {

template <class ReductionPassT>
class TestSubclass : public ReductionPassT {
 public:
  explicit TestSubclass(const spv_target_env env) : ReductionPassT(env) {}
  ~TestSubclass() = default;

  std::vector<std::unique_ptr<ReductionOpportunity>>
  WrapGetAvailableOpportunities(opt::IRContext* context) const {
    return ReductionPassT::GetAvailableOpportunities(context);
  }
};

// Checks whether the given binaries are bit-wise equal.
void CheckEqual(spv_target_env env,
                const std::vector<uint32_t>& expected_binary,
                const std::vector<uint32_t>& actual_binary);

// Assembles the given text and check whether the resulting binary is bit-wise
// equal to the given binary.
void CheckEqual(spv_target_env env, const std::string& expected_text,
                const std::vector<uint32_t>& actual_binary);

// Assembles the given text and turns the given IR into binary, then checks
// whether the resulting binaries are bit-wise equal.
void CheckEqual(spv_target_env env, const std::string& expected_text,
                const opt::IRContext* actual_ir);

const uint32_t kReduceAssembleOption =
    SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS;
const uint32_t kReduceDisassembleOption =
    SPV_BINARY_TO_TEXT_OPTION_NO_HEADER | SPV_BINARY_TO_TEXT_OPTION_INDENT;

}  // namespace reduce
}  // namespace spvtools

#endif  // TEST_REDUCE_REDUCE_TEST_UTIL_H_
