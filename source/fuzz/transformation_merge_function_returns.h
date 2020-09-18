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

#ifndef SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_
#define SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_

#include "source/fuzz/transformation.h"
/*
 *
  // function, except, at most, in the entry block.
  uint32 any_returnable_val_id = 5;

  // The information needed to modify the merge blocks of
  // loops containing return instructions.
  repeated ReturnMergingInfo return_merging_info = 6;

 */
namespace spvtools {
namespace fuzz {
class TransformationMergeFunctionReturns : public Transformation {
 public:
  explicit TransformationMergeFunctionReturns(
      const protobufs::TransformationMergeFunctionReturns& message);

  TransformationMergeFunctionReturns(
      uint32_t function_id, uint32_t outer_header_id, uint32_t outer_return_id,
      uint32_t return_val_id, uint32_t any_returnable_val_id,
      const std::vector<protobufs::ReturnMergingInfo>& returns_merging_info);

  // TODO: Comment.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // TODO: Comment.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationMergeFunctionReturns message_;
};
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_MERGE_FUNCTION_RETURNS_
