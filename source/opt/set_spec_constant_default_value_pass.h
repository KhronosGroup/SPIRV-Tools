// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_
#define LIBSPIRV_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "module.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// The optimization pass that sets the default values for the spec constants
// that have SpecId decorations (i.e. those defined by
// OpSpecConstant{|True|False} instructions).
class SetSpecConstantDefaultValuePass : public Pass {
 public:
  using SpecIdToValueStrMap = std::unordered_map<uint32_t, std::string>;
  using SpecIdToInstMap = std::unordered_map<uint32_t, ir::Instruction*>;

  // Constructs a pass instance with a map from spec ids to default values.
  explicit SetSpecConstantDefaultValuePass(
      const SpecIdToValueStrMap& default_values)
      : Pass(), spec_id_to_value_(default_values) {}
  explicit SetSpecConstantDefaultValuePass(SpecIdToValueStrMap&& default_values)
      : Pass(), spec_id_to_value_(std::move(default_values)) {}

  const char* name() const override { return "set-spec-const-default-value"; }
  bool Process(ir::Module*) override;

 private:
  // The mapping from spec ids to their default values to be set.
  const SpecIdToValueStrMap spec_id_to_value_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SET_SPEC_CONSTANT_DEFAULT_VALUE_PASS_H_
