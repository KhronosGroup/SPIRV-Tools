// Copyright (c) 2020 Stefano Milizia
// Copyright (c) 2020 Google LLC
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

namespace {

// Returns true if the two given constants are equivalent
// (the description of IsApplicable specifies the conditions they must satisfy
// to be considered equivalent)
bool AreEquivalentConstants(opt::IRContext* ir_context,
                            const opt::analysis::Constant& constant1,
                            const opt::analysis::Constant& constant2) {
  // Check that the type ids are the same
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3536): Somehow
  // relax this for integers (so that unsigned integer and signed integer are
  // considered the same type)
  uint32_t type_id1 = ir_context->get_type_mgr()->GetId(constant1.type());
  uint32_t type_id2 = ir_context->get_type_mgr()->GetId(constant2.type());
  if (type_id1 != type_id2) {
    return false;
  }

  // If either constant is null, the other is equivalent iff it is zero-like
  if (constant1.AsNullConstant()) {
    return constant2.IsZero();
  }

  if (constant2.AsNullConstant()) {
    return constant1.IsZero();
  }

  // If the constants are scalar, they are equal iff their words are the same
  if (auto scalar1 = constant1.AsScalarConstant()) {
    return scalar1->words() == constant2.AsScalarConstant()->words();
  }

  // If the constants are composite, they are equivalent iff their components
  // match
  if (constant1.AsCompositeConstant()) {
    auto components1 = constant1.AsCompositeConstant()->GetComponents();
    auto components2 = constant2.AsCompositeConstant()->GetComponents();

    // Since the types match, we already know that the number of components is
    // the same, so we just need to check equivalence for each one
    for (size_t i = 0; i < components1.size(); i++) {
      if (!AreEquivalentConstants(ir_context, *components1[i],
                                  *components2[i])) {
        return false;
      }
    }

    // If we get here, all the components are equivalent
    return true;
  }

  // If we get here, the type is not sensible to check for equivalence
  assert(false &&
         "Equivalency of constants can only be checked with scalar, composite "
         "or null constants.");
  return false;
}
}  // namespace

TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(
        const protobufs::TransformationRecordSynonymousConstants& message)
    : message_(message) {}

TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(uint32_t constant1_id,
                                            uint32_t constant2_id) {
  message_.set_constant1_id(constant1_id);
  message_.set_constant2_id(constant2_id);
}

bool TransformationRecordSynonymousConstants::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& /* unused */) const {
  // The ids must be different
  if (message_.constant1_id() == message_.constant2_id()) {
    return false;
  }

  auto constant1 = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant1_id());
  auto constant2 = ir_context->get_constant_mgr()->FindDeclaredConstant(
      message_.constant2_id());

  // The constants must exist
  if (constant1 == nullptr || constant2 == nullptr) {
    return false;
  }

  return AreEquivalentConstants(ir_context, *constant1, *constant2);
}

void TransformationRecordSynonymousConstants::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  protobufs::FactDataSynonym fact_data_synonym;
  // Define the two equivalent data descriptors (just containing the ids)
  *fact_data_synonym.mutable_data1() =
      MakeDataDescriptor(message_.constant1_id(), {});
  *fact_data_synonym.mutable_data2() =
      MakeDataDescriptor(message_.constant2_id(), {});
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
