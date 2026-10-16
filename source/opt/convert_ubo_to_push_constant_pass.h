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
  Instruction* FindTarget();
  bool CheckType(uint32_t type_id);
  bool CollectPointers(Instruction* variable,
                       std::vector<Instruction*>* pointers);
  bool HasDecorationValue(uint32_t id, spv::Decoration decoration,
                          uint32_t value);
  bool Fail(const std::string& message);

  std::string name_;
  uint32_t descriptor_set_ = 0;
  uint32_t binding_ = 0;
  bool select_by_binding_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONVERT_UBO_TO_PUSH_CONSTANT_PASS_H_
