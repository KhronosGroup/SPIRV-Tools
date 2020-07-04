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
  // The ids must be different
  if (message_.constant_id() == message_.synonym_id()) {
    return false;
  }

  auto constant = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant_id());
  auto synonym = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.synonym_id());

  // The constants must exist
  if (constant == nullptr || synonym == nullptr) {
    return false;
  }

  // If the constants are equal, then they are equivalent
  if (constant == synonym) {
    return true;
  }

  // The types must be the same
  if (!constant->type()->IsSame(synonym->type())) {
    return false;
  }

  // The constants are equivalent if one is null and the other is a static
  // constant with value 0.
  return (constant->AsNullConstant() && IsStaticZeroConstant(synonym)) ||
         (IsStaticZeroConstant(constant) && synonym->AsNullConstant());
}

void TransformationRecordSynonymousConstants::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  protobufs::FactDataSynonym fact_data_synonym;
  // Define the two equivalent data descriptors (just containing the ids)
  *fact_data_synonym.mutable_data1() =
      MakeDataDescriptor(message_.constant_id(), {});
  *fact_data_synonym.mutable_data2() =
      MakeDataDescriptor(message_.synonym_id(), {});
  protobufs::Fact fact;
  *fact.mutable_data_synonym_fact() = fact_data_synonym;

  // Add the fact to the fact manager
  transformation_context->GetFactManager()->AddFact(fact, ir_context);
}

protobufs::Transformation TransformationRecordSynonymousConstants::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_record_synonymous_constants() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools