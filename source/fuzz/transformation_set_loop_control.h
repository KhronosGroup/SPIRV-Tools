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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetLoopControl : public Transformation {
 public:

  const static uint32_t kLoopControlMaskInOperandIndex = 2;
  const static uint32_t kLoopControlFirstLiteralInOperandIndex = 3;

  explicit TransformationSetLoopControl(
          const protobufs::TransformationSetLoopControl& message);

  TransformationSetLoopControl(uint32_t block_id,
                                    uint32_t loop_control, uint32_t peel_count, uint32_t partial_count);

  // - |message_.block_id| must be a block containing an OpLoopMerge
  //   instruction.
  // - |message_.loop_control| must be a legal loop control mask that does not
  //   change the number of type of loop control parameters that are already
  //   present.
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // - The loop control operand of the OpLoopMergeInstruction in
  //   |message_.block_id| is overwritten with |message_.loop_control|.
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

  // Does the version of SPIR-V being used support the PartialCount loop
  // control?
  static bool PartialCountIsSupported(opt::IRContext* context);

  // Does the version of SPIR-V being used support the PeelCount loop control?
  static bool PeelCountIsSupported(opt::IRContext* context);

 private:

  bool LoopControlBitIsAddedByTransformation(SpvLoopControlMask loop_control_single_bit_mask, uint32_t existing_loop_control_mask) const;

  protobufs::TransformationSetLoopControl message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_LOOP_CONTROL_H_
