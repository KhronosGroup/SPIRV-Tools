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

#ifndef SOURCE_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_
#define SOURCE_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_

#include <memory>
#include <unordered_set>
#include <utility>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class FreezeSpecConstantValuePass : public Pass {
 public:
  using SpecIdSet = std::unordered_set<uint32_t>;

  // Constructs a pass instance that will freeze all spec ids.
  FreezeSpecConstantValuePass() : spec_ids_() {}

  // Constructs a pass instance with a set of spec ids to freeze.
  explicit FreezeSpecConstantValuePass(const SpecIdSet& spec_ids)
      : spec_ids_(spec_ids) {}
  explicit FreezeSpecConstantValuePass(SpecIdSet&& spec_ids)
      : spec_ids_(std::move(spec_ids)) {}

  const char* name() const override { return "freeze-spec-const"; }
  Status Process() override;

  // Parses the given null-terminated C string to get a list of spec constant
  // ids to use as arguments to this pass. A valid string should follow the rule
  // below:
  //
  //  "<spec id A> <spec id B> ..."
  //  Example:
  //    "200   201"
  //
  //  Entries are separated with blank spaces (i.e.:' ', '\n', '\r', '\t', '\f',
  //  '\v'). Each entry corresponds to a Spec Id and multiple spaces are allowed
  //  between each.
  //
  //  <spec id>: specifies a spec id value.
  //    The text must represent a valid uint32_t number.
  //    Hex format with '0x' prefix is allowed.
  //    Specifying the same spec id multiple times is allowed.
  static std::unique_ptr<SpecIdSet> ParseSpecIdsString(const char* str);

 private:
  bool ShouldFreezeSpecId(uint32_t spec_id) const;

  // The spec-ids to freeze. If empty, freeze all spec-ids.
  const SpecIdSet spec_ids_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FREEZE_SPEC_CONSTANT_VALUE_PASS_H_
