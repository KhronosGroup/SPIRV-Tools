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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REMOVE_PARAMETERS_H_
#define SOURCE_FUZZ_TRANSFORMATION_REMOVE_PARAMETERS_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationRemoveParameters : public Transformation {
 public:
  explicit TransformationRemoveParameters(
      const protobufs::TransformationRemoveParameters& message);

  TransformationRemoveParameters(uint32_t function_id, uint32_t new_type_id,
                                 const std::vector<uint32_t>& parameter_index,
                                 const std::vector<uint32_t>& fresh_id,
                                 const std::vector<uint32_t>& initializer_id);

  // - |function_id| must be a valid result id of some non-entry-point function
  //   in the module.
  // - |new_type_id| is a result id of the OpTypeFunction instruction s.t.
  //   its return type is the same as the return type of the |function_id|,
  //   it doesn't contain |parameter_index[i]|'th parameter and the order of
  //   remaining parameters is preserved.
  // - |0 <= parameter_index[i] < N| where N is a number of parameters
  //   of |function_id|. All indices are unique.
  // - |fresh_id| is a vector of fresh ids.
  // - |initializer_inst[i].type_id() == param[parameter_index[i]].type_id()|
  //   where |initializer_inst[i]| is an instruction with result id
  //   |initializer_id[i]|, |param[i]| is i'th parameter of the function with
  //   result id |function_id|.
  // - |parameter_index.size() == fresh_id.size() == initializer_id.size()|.
  // - |%id = OpTypePointer Private %param[i].type_id()| must exist in the
  //   module for each |i| in |parameter_index|.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // For each |i| in |parameter_index|:
  // - remove i'th parameter from the function
  // - add a global variable to store the value for the i'th parameter
  // - add OpStore instruction before each function call to store parameter's
  //   value into the variable
  // - add OpLoad in the beginning to load the value from the variable into the
  //   old parameter's id
  // - mark created variable with PointeeIsIrrelevant fact.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationRemoveParameters message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REMOVE_PARAMETERS_H_
