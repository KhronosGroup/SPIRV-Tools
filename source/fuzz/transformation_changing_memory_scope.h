// Copyright (c) 2021 Mostafa Ashraf
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SCOPE_H_
#define SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SCOPE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// This transformation is responsible for changing memory scope.
class TransformationChangingMemoryScope : public Transformation {
 public:
  explicit TransformationChangingMemoryScope(
      protobufs::TransformationChangingMemoryScope message);

  TransformationChangingMemoryScope(
      const protobufs::InstructionDescriptor& needed_instruction,
      uint32_t memory_scope_new_value_id);

  //
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  //
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  // Returns the "in operand" index of the memory scope operand for the
  // instruction.
  // |opcode|: the instruction opcode.
  static uint32_t GetMemoryScopeInOperandIndex(SpvOp opcode);

  // Returns false incase of opcode not atomic instruction.
  static bool IsAtomicInstruction(SpvOp opcode);

  // Returns false if new memory scope not valid scope and if old memory scope
  // is wider than new.
  static bool IsValidScope(SpvScope new_memory_scope_value,
                           SpvScope old_memory_scope_value);

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationChangingMemoryScope message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_CHANGING_MEMORY_SCOPE_H_
