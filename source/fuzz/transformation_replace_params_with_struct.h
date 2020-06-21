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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceParamsWithStruct : public Transformation {
 public:
  explicit TransformationReplaceParamsWithStruct(
      const protobufs::TransformationReplaceParamsWithStruct& message);

  TransformationReplaceParamsWithStruct(
      uint32_t function_id, const std::vector<uint32_t>& parameter_index,
      uint32_t new_type_id, uint32_t fresh_parameter_id,
      uint32_t fresh_composite_id);

  // - |function_id| must be a valid result id of some non-entry-point function
  //   in the module
  // - |0 <= parameter_index[i] < N| where N is the number of parameters in the
  //   function
  // - |new_type_id| is a valid result id of the OpTypeFunction in the module.
  // - this type should contain all operands' type ids, whose indices are not in
  //   |parameter_index|. The last operand must be a result id of an
  //   OpTypeStruct whose i'th component's type is the type of the
  //   |parameter_index[i]|'th function's parameter.
  // - |fresh_parameter_id| and |fresh_composite_id| are fresh ids.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Creates a new function parameter whose type id is the last operand to the
  // |new_type_id| function type. For each |i| in |parameter_index|:
  // - removes i'th parameter from the function
  // - adds OpCompositeExtract to load the value of the parameter from the
  //   struct object
  // Adjusts function calls accordingly: removes replaced parameters, inserts an
  // OpCompositeConstruct to create a struct object containing values of the
  // removed parameters.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationReplaceParamsWithStruct message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_
