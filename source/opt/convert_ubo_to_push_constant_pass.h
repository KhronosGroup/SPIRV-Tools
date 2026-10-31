// Copyright (c) 2026 Diligent Graphics LLC
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

#ifndef SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_
#define SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_

#include <string>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class ConvertUBOToPushConstantPass : public Pass {
 public:
  explicit ConvertUBOToPushConstantPass(const std::string& name)
      : name_(name), select_by_binding_(false) {}
  ConvertUBOToPushConstantPass(uint32_t descriptor_set, uint32_t binding)
      : descriptor_set_(descriptor_set),
        binding_(binding),
        select_by_binding_(true) {}

  const char* name() const override { return "convert-ubo-to-push-constant"; }
  Status Process() override;

 private:
  // Finds the unique Uniform variable with a Block-decorated struct matching
  // the requested name or descriptor set and binding. Reports an error and
  // returns nullptr if the selection is empty, missing, or ambiguous.
  Instruction* FindTargetUniformBuffer();

  // Checks that |target| has no initializer or NonReadable decoration and the
  // module has no existing push constant variable. Reports an error and returns
  // false if any of these conditions prevent conversion; otherwise returns
  // true.
  bool CheckTargetVariable(const Instruction& target);

  // Checks that |type_id| contains no runtime arrays and that the module has
  // the capabilities required to store its members in push constants. Reports
  // an error and returns false if conversion is unsupported; otherwise returns
  // true.
  bool CheckPushConstantType(uint32_t type_id);

  // Checks the memory-access operands of OpLoad or OpCopyMemory. Reports an
  // error and returns false if any mask contains NonPrivatePointer, which
  // cannot be preserved for push constants; otherwise returns true.
  bool CheckMemoryAccess(const Instruction& inst);

  // Appends |variable| and all pointers derived from it by access chains and
  // OpCopyObject to the initially empty |pointers| vector. These are the values
  // whose pointer types must change to PushConstant. Checks every use, reports
  // an error and returns false if any use is unsupported, or returns true if
  // all uses are supported. The module is not modified; |pointers| may be
  // partially populated on failure.
  bool CollectAndValidateBufferPointers(Instruction* variable,
                                        std::vector<Instruction*>* pointers);

  // Converts |target| and the values in |pointers| to PushConstant storage
  // while preserving pointee types and pointer types used by other buffers.
  // |pointers| must contain |target| and all derived pointers. Returns true on
  // success, or false if a required pointer type cannot be obtained.
  bool RewriteBufferPointers(Instruction* target,
                             const std::vector<Instruction*>& pointers);

  // Removes Binding, DescriptorSet, and NonWritable decorations from the
  // converted variable, preserving decorations shared with other variables.
  void RemoveBufferDecorations(uint32_t variable_id);

  // Returns whether |id| has |decoration| with the given single-word |value|,
  // including decorations applied through groups.
  bool HasDecorationValue(uint32_t id, spv::Decoration decoration,
                          uint32_t value);

  // Reports a pass-prefixed error through the message consumer, if present,
  // and returns false for use in failure paths.
  bool Fail(const std::string& message);

  std::string name_;
  uint32_t descriptor_set_ = 0;
  uint32_t binding_ = 0;
  bool select_by_binding_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_
