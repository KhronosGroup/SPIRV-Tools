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

#ifndef SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_
#define SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass for turning uses of constants into more complex forms.
class FuzzerPassObfuscateConstants : public FuzzerPass {
 public:
  FuzzerPassObfuscateConstants() = default;

  ~FuzzerPassObfuscateConstants() override = default;

  void Apply(opt::IRContext* ir_context, FactManager* fact_manager,
             FuzzerContext* fuzzer_context,
             protobufs::TransformationSequence* transformations) override;

 private:
  uint32_t FindScalarConstant(
      opt::IRContext* ir_context,
      const opt::analysis::ScalarConstant* scalar_constant);

  void ObfuscateConstant(uint32_t depth,
                         const protobufs::IdUseDescriptor& constant_use,
                         opt::IRContext* ir_context, FactManager* fact_manager,
                         FuzzerContext* fuzzer_context,
                         protobufs::TransformationSequence* transformations);

  void ObfuscateBoolConstantViaFloatConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t float_constant_id_1, uint32_t float_constant_id_2,
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  void ObfuscateBoolConstantViaSignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t signed_int_constant_id_1, uint32_t signed_int_constant_id_2,
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  void ObfuscateBoolConstantViaUnsignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t unsigned_int_constant_id_1, uint32_t unsigned_int_constant_id_2,
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);

  void ObfuscateBoolConstantViaConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      const std::vector<SpvOp>& greater_than_opcodes,
      const std::vector<SpvOp>& less_than_opcodes, uint32_t constant_id_1,
      uint32_t constant_id_2, bool first_constant_is_larger,
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // #define SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_
