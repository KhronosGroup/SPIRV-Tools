// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_PERMUTE_FUNCTIONS_H_
#define SOURCE_FUZZ_TRANSFORMATION_PERMUTE_FUNCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationPermuteFunctions : public Transformation {
 public:
  explicit TransformationPermuteFunctions(
      const protobufs::TransformationPermuteFunctions& message);

  TransformationPermuteFunctions(
      const std::vector<uint32_t>& permutations);

  // - |permutation| is a set of [0..(n - 1)], where n is the number
  //   of functions in the module
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;


  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;
  protobufs::Transformation ToMessage() const override;


//  private:
  protobufs::TransformationPermuteFunctions message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_PERMUTE_FUNCTIONS_H_
