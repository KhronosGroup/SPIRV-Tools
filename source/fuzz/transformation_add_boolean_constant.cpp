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

bool TransformationAddBooleanConstant::IsApplicable(IRContext* context) {
  opt::analysis::Bool bool_type;
  if (!context->get_type_mgr()->GetId(&bool_type)) {
    // No OpTypeBool is present.
    return false;
  }
  opt::analysis::BoolConstant bool_constant(&bool_type, is_true_);
  if (context->get_constant_mgr()->FindConstant(&bool_constant)) {
    // The desired constant is already present.
    return false;
  }
  if (context->get_def_use_mgr()->GetDef(fresh_id_)) {
    // We require the id for the new block to be unused.
    return false;
  }
  return true;
}

void TransformationAddBooleanConstant::Apply(IRContext* context) {
  opt::analysis::Bool bool_type;
  // Add the boolean constant to the module, ensuring the module's id bound is
  // high enough.
  fuzzerutil::UpdateModuleIdBound(context, fresh_id_);
  context->module()->AddGlobalValue(
      is_true_ ? SpvOpConstantTrue : SpvOpConstantFalse, fresh_id_,
      context->get_type_mgr()->GetId(&bool_type));
  // Invalidate all analyses
  context->InvalidateAnalysesExceptFor(IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddBooleanConstant::ToMessage() {
  auto add_boolean_constant_message =
      new protobufs::TransformationAddBooleanConstant;
  add_boolean_constant_message->set_fresh_id(fresh_id_);
  add_boolean_constant_message->set_is_true(is_true_);
  protobufs::Transformation result;
  result.set_allocated_add_boolean_constant(add_boolean_constant_message);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
