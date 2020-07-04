// Copyright (c) 2020 Stefano Milizia
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

#include "transformation_record_synonymous_constants.h"

namespace spvtools {
namespace fuzz {
TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(
        const protobufs::TransformationRecordSynonymousConstants& message)
    : message_(message) {}

TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(uint32_t constant_id,
                                            uint32_t synonym_id) {
  message_.set_constant_id(constant_id);
  message_.set_synonym_id(synonym_id);
}

static inline bool IsStaticZeroConstant(
    const opt::analysis::Constant* constant) {
  return constant->AsScalarConstant() && constant->IsZero();
}

bool TransformationRecordSynonymousConstants::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  auto constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant_id());
  auto synonym = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.synonym_id());

  // The constants must exist
  if (constant == nullptr || synonym == nullptr) {
    return false;
  }

  // The types must be the same
  if (!constant->type()->IsSame(synonym->type())) {
    return false;
  }

  return (constant->AsNullConstant() && IsStaticZeroConstant(synonym)) ||
         (IsStaticZeroConstant(constant) && synonym->AsNullConstant());
}

// TODO
void TransformationRecordSynonymousConstants::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

protobufs::Transformation TransformationRecordSynonymousConstants::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_record_synonymous_constants() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools