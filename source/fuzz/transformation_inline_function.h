// Copyright (c) 2020 André Perez Maselco
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationInlineFunction : public Transformation {
 public:
  explicit TransformationInlineFunction(
      const protobufs::TransformationInlineFunction& message);

  TransformationInlineFunction(
      const std::map<uint32_t, uint32_t>& result_id_map,
      uint32_t function_call_id);

  // - |message_.result_id_map| must map the instructions of the called function
  // to fresh ids.
  // - |message_.function_call_id| must be an OpFunctionCall instruction. The
  // called function must not have an early return.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Iterates over the called function blocks and clones each instruction in the
  // blocks. The cloned instruction result id and its operand ids are set to the
  // corresponding value in |message_.result_id_map|. Finally, the cloned
  // instructions are inserted into the caller function.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Requires that the function contains at most one OpReturnValue instruction.
  // Returns the id associated with this instruction if present, and 0
  // otherwise.
  static uint32_t GetReturnValueId(opt::IRContext* ir_context,
                                   opt::Function* function);

 private:
  protobufs::TransformationInlineFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_INLINE_FUNCTION_H_
