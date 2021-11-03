// Copyright (c) 2021 Tencent Inc.
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

#ifndef SOURCE_OPT_LEGALIZE_IMAGE_OPS_H_
#define SOURCE_OPT_LEGALIZE_IMAGE_OPS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class LegalizeImageOpsPass : public Pass {
 public:
  LegalizeImageOpsPass() = default;

  const char* name() const override { return "legalize-image-ops-pass"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants;
  }

 private:
  // Return type id for float with |width|
  analysis::Type* FloatScalarType(uint32_t width);

  // Return type id for vector of length |vlen| of float of |width|
  analysis::Type* FloatVectorType(uint32_t v_len, uint32_t width);

  void SetRelaxed(Instruction* inst);

  bool IsSameTypeImaged(Instruction* inst_a, Instruction* inst_b);

  uint32_t IndexOfType(Instruction* inst);

  void MoveType(Instruction* a, Module::inst_iterator& insert_point);

  bool ConvertOpTypeImage(Instruction* inst);

  bool ConvertImageOps(BasicBlock* bb, Instruction* inst);

  bool ProcessFunction(Function* func);

  Pass::Status ProcessImpl();

  void Initialize();

  // Set of sample operations
  std::unordered_set<uint32_t> image_ops_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LEGALIZE_IMAGE_OPS_H_