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
// Examples include replacing 'true' with '42 < 52', and replacing '42' with
// 'a.b.c' if 'a.b.c' is known to hold the value '42'.
class FuzzerPassObfuscateConstants : public FuzzerPass {
 public:
  FuzzerPassObfuscateConstants(
      opt::IRContext* ir_context, FactManager* fact_manager,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations)
      : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

  ~FuzzerPassObfuscateConstants() override = default;

  void Apply() override;

 private:
  // TODO: this helper is necessitated by the way the fact manager stores
  //  constants, and to difficult interplay between the fact manager and
  //  constant manager.  It would be good to find a way to reduce this
  //  complexity.
  //
  // Helper method to find the id of a scalar constant if it is declared in the
  // module.  Needed because the constant might have a type that is different
  // to the type of a corresponding constant in the constant manager.
  uint32_t FindScalarConstant(
      const opt::analysis::ScalarConstant* scalar_constant);

  // Applies 0 or more transformations to potentially obfuscate the constant
  // use represented by |constant_use|.  The |depth| parameter controls how
  // deeply obfuscation can recurse.
  void ObfuscateConstant(uint32_t depth,
                         const protobufs::IdUseDescriptor& constant_use);

  // Applies a transformation to replace the boolean constant usage represented
  // by |bool_constant_use| with a binary expression involving
  // |float_constant_id_1| and |float_constant_id_2|, which must not be equal
  // to one another.  Possibly further obfuscates the uses of these float
  // constants.  The |depth| parameter controls how deeply obfuscation can
  // recurse.
  void ObfuscateBoolConstantViaFloatConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t float_constant_id_1, uint32_t float_constant_id_2);

  // Similar to the above, but for signed int constants.
  void ObfuscateBoolConstantViaSignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t signed_int_constant_id_1, uint32_t signed_int_constant_id_2);

  // Similar to the above, but for unsigned int constants.
  void ObfuscateBoolConstantViaUnsignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t unsigned_int_constant_id_1, uint32_t unsigned_int_constant_id_2);

  // A helper method to capture the common parts of the above methods.
  // The method is used to obfuscate the boolean constant usage represented by
  // |bool_constant_use| by replacing it with '|constant_id_1| OP
  // |constant_id_2|', where 'OP' is chosen from either |greater_than_opcodes|
  // or |less_than_opcodes|.
  //
  // The two constant ids must not represent the same value, and thus
  // |greater_than_opcodes| may include 'greater than or equal' opcodes
  // (similar for |less_than_opcodes|).
  void ObfuscateBoolConstantViaConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      const std::vector<SpvOp>& greater_than_opcodes,
      const std::vector<SpvOp>& less_than_opcodes, uint32_t constant_id_1,
      uint32_t constant_id_2, bool first_constant_is_larger);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // #define SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_
