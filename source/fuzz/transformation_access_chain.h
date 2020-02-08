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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_
#define SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_

#include <utility>

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAccessChain : public Transformation {
 public:
  explicit TransformationAccessChain(
      const protobufs::TransformationAccessChain& message);

  TransformationAccessChain(
      uint32_t fresh_id, uint32_t pointer_id,
      const std::vector<uint32_t>& index,
      const protobufs::InstructionDescriptor& instruction_to_insert_before);

  // TODO comment
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // TODO comment
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // TODO comment
  std::pair<bool, uint32_t> GetIndexValue(opt::IRContext* context,
                                          uint32_t index_id) const;

  protobufs::TransformationAccessChain message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ACCESS_CHAIN_H_
