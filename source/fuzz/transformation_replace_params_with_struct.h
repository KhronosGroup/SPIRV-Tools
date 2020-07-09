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
      const std::vector<uint32_t>& parameter_id,
      uint32_t fresh_function_type_id, uint32_t fresh_parameter_id,
      const std::vector<uint32_t>& fresh_composite_id);

  // - |parameter_id[i]| is a valid result id of some OpFunctionParameter
  //   instruction. All parameter ids must correspond to parameters of the same
  //   function. That function may not be an entry-point function.
  //   |parameter_id| may not be empty or contain duplicates.
  // - |fresh_composite_id.size()| is equal to the number of callees of the
  //   function (see GetNumberOfCallees method). All elements of this vector
  //   should be unique and fresh.
  // - |fresh_function_type_id|, |fresh_parameter_id|, |fresh_struct_type_id|
  //   are all fresh unique ids.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Creates a new function parameter with result id |fresh_parameter_id|.
  //   Parameter's type is OpTypeStruct with each components type equal to the
  //   type of the replaced parameter.
  // - This OpTypeStruct instruction is created with result id
  //   |fresh_struct_type_id| if no required type is present in the module.
  // - OpCompositeConstruct with result id from |fresh_composite_id| is inserted
  //   before each OpFunctionCall instruction.
  // - OpCompositeExtract with result id equal to the result id of the replaced
  //   parameter is created in the function.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the number of OpFunctionCall instructions in the module that call
  // |function_id|.
  static uint32_t GetNumberOfCallees(opt::IRContext* ir_context,
                                     uint32_t function_id);

  // Returns true if parameter's type is supported by this transformation.
  static bool IsParameterTypeSupported(const opt::analysis::Type& param_type);

 private:
  protobufs::TransformationReplaceParamsWithStruct message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_
