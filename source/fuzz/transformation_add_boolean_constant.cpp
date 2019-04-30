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

#include "source/fuzz/transformation_add_boolean_constant.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/opt/ir_context.h"
#include "source/opt/types.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

bool transformation::IsApplicable(
    const protobufs::TransformationAddBooleanConstant& message,
    IRContext* context, const FactManager& /*unused*/) {
  opt::analysis::Bool bool_type;
  if (!context->get_type_mgr()->GetId(&bool_type)) {
    // No OpTypeBool is present.
    return false;
  }
  opt::analysis::BoolConstant bool_constant(&bool_type, message.is_true());
  if (context->get_constant_mgr()->FindConstant(&bool_constant)) {
    // The desired constant is already present.
    return false;
  }
  if (context->get_def_use_mgr()->GetDef(message.fresh_id())) {
    // We require the id for the new block to be unused.
    return false;
  }
  return true;
}

void transformation::Apply(
    const protobufs::TransformationAddBooleanConstant& message,
    IRContext* context, FactManager* /*unused*/) {
  opt::analysis::Bool bool_type;
  // Add the boolean constant to the module, ensuring the module's id bound is
  // high enough.
  fuzzerutil::UpdateModuleIdBound(context, message.fresh_id());
  context->module()->AddGlobalValue(
      message.is_true() ? SpvOpConstantTrue : SpvOpConstantFalse,
      message.fresh_id(), context->get_type_mgr()->GetId(&bool_type));
  // Invalidate all analyses
  context->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
}

protobufs::TransformationAddBooleanConstant
transformation::MakeTransformationAddBooleanConstant(uint32_t fresh_id,
                                                     bool is_true) {
  protobufs::TransformationAddBooleanConstant result;
  result.set_fresh_id(fresh_id);
  result.set_is_true(is_true);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
