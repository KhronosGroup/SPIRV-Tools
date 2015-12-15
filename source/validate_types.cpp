// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "headers/spirv.h"
#include "validate_types.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

using std::find;
using std::string;
using std::unordered_set;
using std::vector;

namespace {
const vector<vector<SpvOp>>& GetModuleOrder() {
  // See Section 2.4
  // clang-format off
  static const vector<vector<SpvOp>> moduleOrder = {
    {SpvOpCapability},
    {SpvOpExtension},
    {SpvOpExtInstImport},
    {SpvOpMemoryModel},
    {SpvOpEntryPoint},
    {SpvOpExecutionMode},
    {
      // first set of debug instructions
      SpvOpSourceContinued,
      SpvOpSource,
      SpvOpSourceExtension,
      SpvOpString,
    },
    {
      // second set of debug instructions
      SpvOpName,
      SpvOpMemberName
    },
    {
      // annotation instructions
      SpvOpDecorate,
      SpvOpMemberDecorate,
      SpvOpGroupDecorate,
      SpvOpGroupMemberDecorate,
      SpvOpDecorationGroup
    },
    {
      // All type and constant instructions
      SpvOpTypeVoid,
      SpvOpTypeBool,
      SpvOpTypeInt,
      SpvOpTypeFloat,
      SpvOpTypeVector,
      SpvOpTypeMatrix,
      SpvOpTypeImage,
      SpvOpTypeSampler,
      SpvOpTypeSampledImage,
      SpvOpTypeArray,
      SpvOpTypeRuntimeArray,
      SpvOpTypeStruct,
      SpvOpTypeOpaque,
      SpvOpTypePointer,
      SpvOpTypeFunction,
      SpvOpTypeEvent,
      SpvOpTypeDeviceEvent,
      SpvOpTypeReserveId,
      SpvOpTypeQueue,
      SpvOpTypePipe,
      SpvOpTypeForwardPointer,
      SpvOpConstantTrue,
      SpvOpConstantFalse,
      SpvOpConstant,
      SpvOpConstantComposite,
      SpvOpConstantSampler,
      SpvOpConstantNull,
      SpvOpSpecConstantTrue,
      SpvOpSpecConstantFalse,
      SpvOpSpecConstant,
      SpvOpSpecConstantComposite,
      SpvOpSpecConstantOp,
      SpvOpVariable,
      SpvOpLine
    }
  };
  // clang-format on

  return moduleOrder;
}
}

namespace libspirv {

ValidationState_t::ValidationState_t(spv_diagnostic* diag, uint32_t options)
    : diagnostic_(diag),
      instruction_counter_(0),
      defined_ids_{},
      unresolved_forward_ids_{},
      validation_flags_(options),
      operand_names_{},
      module_layout_order_stage_(0),
      current_layout_stage_(ModuleLayoutSection::kModule) {}

spv_result_t ValidationState_t::defineId(uint32_t id) {
  if (defined_ids_.find(id) == end(defined_ids_)) {
    defined_ids_.insert(id);
  } else {
    return diag(SPV_ERROR_INVALID_ID) << "ID cannot be assigned multiple times";
  }
  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::forwardDeclareId(uint32_t id) {
  unresolved_forward_ids_.insert(id);
  return SPV_SUCCESS;
}

spv_result_t ValidationState_t::removeIfForwardDeclared(uint32_t id) {
  unresolved_forward_ids_.erase(id);
  return SPV_SUCCESS;
}

void ValidationState_t::assignNameToId(uint32_t id, string name) {
  operand_names_[id] = name;
}

string ValidationState_t::getIdName(uint32_t id) const {
  std::stringstream out;
  out << id;
  if (operand_names_.find(id) != end(operand_names_)) {
    out << "[" << operand_names_.at(id) << "]";
  }
  return out.str();
}

size_t ValidationState_t::unresolvedForwardIdCount() const {
  return unresolved_forward_ids_.size();
}

vector<uint32_t> ValidationState_t::unresolvedForwardIds() const {
  vector<uint32_t> out(begin(unresolved_forward_ids_),
                       end(unresolved_forward_ids_));
  return out;
}

bool ValidationState_t::isDefinedId(uint32_t id) const {
  return defined_ids_.find(id) != end(defined_ids_);
}

bool ValidationState_t::is_enabled(spv_validate_options_t flag) const {
  return (flag & validation_flags_) == flag;
}

// Increments the instruction count. Used for diagnostic
int ValidationState_t::incrementInstructionCount() {
  return instruction_counter_++;
}

ModuleLayoutSection ValidationState_t::getLayoutStage() const {
  return current_layout_stage_;
}

void ValidationState_t::progressToNextLayoutStageOrder() {
  module_layout_order_stage_ +=
      module_layout_order_stage_ < GetModuleOrder().size();
  if (module_layout_order_stage_ >= GetModuleOrder().size()) {
    current_layout_stage_ = libspirv::ModuleLayoutSection::kFunction;
  }
}

bool ValidationState_t::isOpcodeInCurrentLayoutStage(SpvOp op) {
  const vector<SpvOp>& currentStage =
      GetModuleOrder()[module_layout_order_stage_];
  return end(currentStage) != find(begin(currentStage), end(currentStage), op);
}

libspirv::DiagnosticStream ValidationState_t::diag(
    spv_result_t error_code) const {
  return libspirv::DiagnosticStream(
      {0, 0, static_cast<size_t>(instruction_counter_)}, diagnostic_,
      error_code);
}
}
