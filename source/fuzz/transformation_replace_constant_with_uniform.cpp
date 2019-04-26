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

#include "source/fuzz/transformation_replace_constant_with_uniform.h"

namespace spvtools {
namespace fuzz {

bool TransformationReplaceConstantWithUniform::IsApplicable(
    spvtools::opt::IRContext* context,
    const spvtools::fuzz::FactManager& fact_manager) {
  // 1. Check that the id we are trying to replace is that of a constant.
  auto declared_constant = context->get_constant_mgr()->FindDeclaredConstant(
      id_use_descriptor_.GetIdOfInterest());

  if (!declared_constant) {
    return false;
  }

  if (!declared_constant->AsScalarConstant()) {
    return false;
  }

  auto constant_associated_with_uniform =
      fact_manager.GetConstantFromUniformDescriptor(uniform_descriptor_);
  if (!constant_associated_with_uniform) {
    return false;
  }

  if (!constant_associated_with_uniform->AsScalarConstant()) {
    return false;
  }

  if (!declared_constant->type()->IsSame(
          constant_associated_with_uniform->type())) {
    return false;
  }

  if (declared_constant->AsScalarConstant()->words() !=
      constant_associated_with_uniform->AsScalarConstant()->words()) {
    return false;
  }

  auto instruction_using_constant = id_use_descriptor_.FindInstruction(context);
  if (!instruction_using_constant) {
    return false;
  }

  return true;
}

void TransformationReplaceConstantWithUniform::Apply(
    spvtools::opt::IRContext* context,
    spvtools::fuzz::FactManager* fact_manager) {
  (void)(context);
  (void)(fact_manager);
  assert(false && "TODO");
}

protobufs::Transformation
TransformationReplaceConstantWithUniform::ToMessage() {
  assert(false && "TODO");
  return protobufs::Transformation();
}

}  // namespace fuzz
}  // namespace spvtools
