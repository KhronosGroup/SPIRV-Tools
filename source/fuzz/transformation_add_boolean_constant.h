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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_BOOLEAN_CONSTANT_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_BOOLEAN_CONSTANT_H_

#include "transformation.h"

namespace spvtools {
namespace fuzz {

// Supports adding the constants true and false to a module, which may be
// necessary in order to enable other transformations if they are not present.
class TransformationAddBooleanConstant : public Transformation {
 public:
  TransformationAddBooleanConstant(uint32_t fresh_id, bool is_true)
      : fresh_id_(fresh_id), is_true_(is_true) {}

  ~TransformationAddBooleanConstant() override = default;

  // - |fresh_id_| must not be used by the module.
  // - The module must already contain OpTypeBool.
  // - The module must not already contain an OpConstantTrue (OpConstantFalse)
  // instruction if is_true_ holds (does not hold).
  bool IsApplicable(opt::IRContext* context) override;

  // - Adds OpConstantTrue (OpConstantFalse) to the module with id |fresh_id_|
  // if is_true_ holds (does not hold).
  void Apply(opt::IRContext* context) override;

 private:
  const uint32_t fresh_id_;
  const bool is_true_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_BOOLEAN_CONSTANT_H_
