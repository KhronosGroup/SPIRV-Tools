// Copyright (c) 2021 Emiljano Gjiriti
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTIONS_H_
#define SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTIONS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSwapFunctions : public Transformation {
public:
  explicit TransformationSwapFunctions(
      protobufs::TransformationSwapFunctions message);

  TransformationSwapFunctions(uint32_t result_id1, uint32_t result_id2);

  // Swaps two functions with result_ids result_id1
  // and result_id2 if they are valid function ids
  // and they are not equal.
  bool IsApplicable(
      opt::IRContext *ir_context,
      const TransformationContext &transformation_context) const override;

  void Apply(opt::IRContext *ir_context,
             TransformationContext *transformation_context) const override;

private:
  // private
  std::unordered_set<uint32_t> GetFreshIds() const override;
  protobufs::Transformation ToMessage() const override;
  protobufs::TransformationSwapFunctions message_;
};

} // namespace fuzz
} // namespace spvtools

#endif // SOURCE_FUZZ_TRANSFORMATION_SWAP_FUNCTIONS_H_
