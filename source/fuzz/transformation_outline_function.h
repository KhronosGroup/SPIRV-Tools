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

#ifndef SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationOutlineFunction : public Transformation {
 public:
  explicit TransformationOutlineFunction(
      const protobufs::TransformationOutlineFunction& message);

  TransformationOutlineFunction(uint32_t entry_block, uint32_t exit_block,
                                uint32_t new_function_type_id,
                                uint32_t new_function_id,
                                uint32_t new_function_entry_block,
                                uint32_t new_function_exit_block,
                                uint32_t function_call_result_id);

  // TODO comment
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // TODO comment
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  // A helper method for the applicability check.  Returns true if and only if
  // |id| is (a) a fresh id for the module, and (b) an id that has not
  // previously been subject to this check.  We use this to check whether the
  // ids given for the transformation are not only fresh but also different from
  // one another.
  bool CheckIdIsFreshAndNotUsedByThisTransformation(
      uint32_t id, opt::IRContext* context,
      std::set<uint32_t>* ids_used_by_this_transformation) const;

  protobufs::TransformationOutlineFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
