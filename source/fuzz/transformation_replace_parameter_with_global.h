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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceParameterWithGlobal : public Transformation {
 public:
  explicit TransformationReplaceParameterWithGlobal(
      const protobufs::TransformationReplaceParameterWithGlobal& message);

  TransformationReplaceParameterWithGlobal(uint32_t new_type_id,
                                           uint32_t parameter_id,
                                           uint32_t fresh_id,
                                           uint32_t initializer_id);

  // - |new_type_id| is a result id of the OpTypeFunction instruction s.t.
  //   its return type is the same as the return type of the |function_id|,
  //   it doesn't contain parameter with result id |parameter_id| and the order
  //   of remaining parameters is preserved.
  // - |parameter_id| is the result id of the parameter to replace.
  // - |fresh_id| is 0 if parameter is not a pointer or a pointer with Function
  //   storage class. Otherwise, this is a fresh id.
  // - |initializer_id| is a result id of an instruction used to initialize
  //   a global variable. Its type id must be equal to either the type of the
  //   parameter if the latter is a not a pointer, or to its pointee otherwise.
  //   If parameter is a pointer with Workgroup storage class, this is 0.
  // - the function that contains |parameter_id| may not be an entry-point
  //   function.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Removes parameter with result id |parameter_id| from its function
  // - Adds a global variable to store the value for the parameter
  // - Add an OpStore or OpCopyMemory instruction before each function call to
  //   store parameter's value into the variable
  // - Adds OpLoad or OpCopyMemory at the beginning of the function to load the
  //   value from the variable into the old parameter's id (or a local variable
  //   if the parameter is a pointer)
  // - Marks created variable with PointeeIsIrrelevant fact if needed.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if the type of the parameter is supported by this
  // transformation.
  static bool CanReplaceFunctionParameterType(opt::IRContext* ir_context,
                                              uint32_t param_type_id);

  static SpvStorageClass GetStorageClassForGlobalVariable(
      opt::IRContext* ir_context, uint32_t param_type_id);

 private:
  protobufs::TransformationReplaceParameterWithGlobal message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_
