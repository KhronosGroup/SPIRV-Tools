// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceBranchFromDeadBlockWithExit : public Transformation {
 public:
  explicit TransformationReplaceBranchFromDeadBlockWithExit(
      const protobufs::TransformationReplaceBranchFromDeadBlockWithExit& message);

  TransformationReplaceBranchFromDeadBlockWithExit(uint32_t block_id,
                                                   SpvOp opcode,
                                                   uint32_t return_value_id);

  // TODO comment
  bool IsApplicable(opt::IRContext* ir_context,
                    const TransformationContext& transformation_context) const override;

  // TODO comment
  void Apply(opt::IRContext* ir_context, TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceBranchFromDeadBlockWithExit message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_BRANCH_FROM_DEAD_BLOCK_WITH_EXIT_H_
