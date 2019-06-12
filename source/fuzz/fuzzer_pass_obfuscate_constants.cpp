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

#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"

#include "source/fuzz/transformation_replace_boolean_constant_with_constant_binary.h"
#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

FuzzerPassObfuscateConstants::FuzzerPassObfuscateConstants(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassObfuscateConstants::~FuzzerPassObfuscateConstants() = default;

void FuzzerPassObfuscateConstants::ObfuscateBoolConstantViaConstantPair(
    uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
    const std::vector<SpvOp>& greater_than_opcodes,
    const std::vector<SpvOp>& less_than_opcodes, uint32_t constant_id_1,
    uint32_t constant_id_2, bool first_constant_is_larger) {
  auto bool_constant_opcode = GetIRContext()
                                  ->get_def_use_mgr()
                                  ->GetDef(bool_constant_use.id_of_interest())
                                  ->opcode();
  assert((bool_constant_opcode == SpvOpConstantFalse ||
          bool_constant_opcode == SpvOpConstantTrue) &&
         "Precondition: this must be a usage of a boolean constant.");

  // Pick an opcode at random.  First randomly decide whether to generate
  // a 'greater than' or 'less than' kind of opcode, and then select a
  // random opcode from the resulting subset.
  SpvOp comparison_opcode;
  if (GetFuzzerContext()->GetRandomGenerator()->RandomBool()) {
    comparison_opcode = greater_than_opcodes
        [GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
            static_cast<uint32_t>(greater_than_opcodes.size()))];
  } else {
    comparison_opcode = less_than_opcodes
        [GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
            static_cast<uint32_t>(less_than_opcodes.size()))];
  }

  // We now need to decide how to order constant_id_1 and constant_id_2 such
  // that 'constant_id_1 comparison_opcode constant_id_2' evaluates to the
  // boolean constant.  The following considers the rather large number of
  // cases in turn.  There might be a more compact representation.
  uint32_t lhs_id;
  uint32_t rhs_id;
  if (bool_constant_opcode == SpvOpConstantTrue) {
    if (first_constant_is_larger) {
      if (std::find(greater_than_opcodes.begin(), greater_than_opcodes.end(),
                    comparison_opcode) != greater_than_opcodes.end()) {
        lhs_id = constant_id_1;
        rhs_id = constant_id_2;
      } else {
        lhs_id = constant_id_2;
        rhs_id = constant_id_1;
      }
    } else {
      if (std::find(greater_than_opcodes.begin(), greater_than_opcodes.end(),
                    comparison_opcode) != greater_than_opcodes.end()) {
        lhs_id = constant_id_2;
        rhs_id = constant_id_1;
      } else {
        lhs_id = constant_id_1;
        rhs_id = constant_id_2;
      }
    }
  } else {
    if (first_constant_is_larger) {
      if (std::find(greater_than_opcodes.begin(), greater_than_opcodes.end(),
                    comparison_opcode) != greater_than_opcodes.end()) {
        lhs_id = constant_id_2;
        rhs_id = constant_id_1;
      } else {
        lhs_id = constant_id_1;
        rhs_id = constant_id_2;
      }
    } else {
      if (std::find(greater_than_opcodes.begin(), greater_than_opcodes.end(),
                    comparison_opcode) != greater_than_opcodes.end()) {
        lhs_id = constant_id_1;
        rhs_id = constant_id_2;
      } else {
        lhs_id = constant_id_2;
        rhs_id = constant_id_1;
      }
    }
  }

  auto transformation = transformation::
      MakeTransformationReplaceBooleanConstantWithConstantBinary(
          bool_constant_use, lhs_id, rhs_id, comparison_opcode,
          GetFuzzerContext()->GetFreshId());
  // The transformation should be applicable by construction.
  assert(transformation::IsApplicable(transformation, GetIRContext(),
                                      *GetFactManager()));

  auto binary_operator_instruction =
      transformation::Apply(transformation, GetIRContext(), GetFactManager());

  *GetTransformations()
       ->add_transformation()
       ->mutable_replace_boolean_constant_with_constant_binary() =
      transformation;

  for (uint32_t index : {0u, 1u}) {
    if (GetFuzzerContext()->GoDeeperInConstantObfuscation()(
            depth, GetFuzzerContext()->GetRandomGenerator())) {
      auto in_operand_use = transformation::MakeIdUseDescriptor(
          binary_operator_instruction->GetSingleWordInOperand(index),
          binary_operator_instruction->opcode(), index,
          binary_operator_instruction->result_id(), 0);
      ObfuscateConstant(depth + 1, in_operand_use);
    }
  }
}

void FuzzerPassObfuscateConstants::ObfuscateBoolConstantViaFloatConstantPair(
    uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
    uint32_t float_constant_id_1, uint32_t float_constant_id_2) {
  auto float_constant_1 = GetIRContext()
                              ->get_constant_mgr()
                              ->FindDeclaredConstant(float_constant_id_1)
                              ->AsFloatConstant();
  auto float_constant_2 = GetIRContext()
                              ->get_constant_mgr()
                              ->FindDeclaredConstant(float_constant_id_2)
                              ->AsFloatConstant();
  assert(float_constant_1->words() != float_constant_2->words() &&
         "The constants should not be identical.");
  bool first_constant_is_larger;
  assert(float_constant_1->type()->AsFloat()->width() ==
             float_constant_2->type()->AsFloat()->width() &&
         "First and second floating-point constants must have the same width.");
  if (float_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        float_constant_1->GetFloat() > float_constant_2->GetFloat();
  } else {
    assert(float_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported floating-point widths are 32 and 64.");
    first_constant_is_larger =
        float_constant_1->GetDouble() > float_constant_2->GetDouble();
  }
  std::vector<SpvOp> greater_than_opcodes{
      SpvOpFOrdGreaterThan, SpvOpFOrdGreaterThanEqual, SpvOpFUnordGreaterThan,
      SpvOpFUnordGreaterThanEqual};
  std::vector<SpvOp> less_than_opcodes{
      SpvOpFOrdGreaterThan, SpvOpFOrdGreaterThanEqual, SpvOpFUnordGreaterThan,
      SpvOpFUnordGreaterThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      float_constant_id_1, float_constant_id_2, first_constant_is_larger);
}

void FuzzerPassObfuscateConstants::
    ObfuscateBoolConstantViaSignedIntConstantPair(
        uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
        uint32_t signed_int_constant_id_1, uint32_t signed_int_constant_id_2) {
  auto signed_int_constant_1 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(signed_int_constant_id_1)
          ->AsIntConstant();
  auto signed_int_constant_2 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(signed_int_constant_id_2)
          ->AsIntConstant();
  assert(signed_int_constant_1->words() != signed_int_constant_2->words() &&
         "The constants should not be identical.");
  bool first_constant_is_larger;
  assert(signed_int_constant_1->type()->AsInteger()->width() ==
             signed_int_constant_2->type()->AsInteger()->width() &&
         "First and second floating-point constants must have the same width.");
  assert(signed_int_constant_1->type()->AsInteger()->IsSigned());
  assert(signed_int_constant_2->type()->AsInteger()->IsSigned());
  if (signed_int_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        signed_int_constant_1->GetS32() > signed_int_constant_2->GetS32();
  } else {
    assert(signed_int_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported integer widths are 32 and 64.");
    first_constant_is_larger =
        signed_int_constant_1->GetS64() > signed_int_constant_2->GetS64();
  }
  std::vector<SpvOp> greater_than_opcodes{SpvOpSGreaterThan,
                                          SpvOpSGreaterThanEqual};
  std::vector<SpvOp> less_than_opcodes{SpvOpSLessThan, SpvOpSLessThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      signed_int_constant_id_1, signed_int_constant_id_2,
      first_constant_is_larger);
}

void FuzzerPassObfuscateConstants::
    ObfuscateBoolConstantViaUnsignedIntConstantPair(
        uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
        uint32_t unsigned_int_constant_id_1,
        uint32_t unsigned_int_constant_id_2) {
  auto unsigned_int_constant_1 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(unsigned_int_constant_id_1)
          ->AsIntConstant();
  auto unsigned_int_constant_2 =
      GetIRContext()
          ->get_constant_mgr()
          ->FindDeclaredConstant(unsigned_int_constant_id_2)
          ->AsIntConstant();
  assert(unsigned_int_constant_1->words() != unsigned_int_constant_2->words() &&
         "The constants should not be identical.");
  bool first_constant_is_larger;
  assert(unsigned_int_constant_1->type()->AsInteger()->width() ==
             unsigned_int_constant_2->type()->AsInteger()->width() &&
         "First and second floating-point constants must have the same width.");
  assert(!unsigned_int_constant_1->type()->AsInteger()->IsSigned());
  assert(!unsigned_int_constant_2->type()->AsInteger()->IsSigned());
  if (unsigned_int_constant_1->type()->AsFloat()->width() == 32) {
    first_constant_is_larger =
        unsigned_int_constant_1->GetU32() > unsigned_int_constant_2->GetU32();
  } else {
    assert(unsigned_int_constant_1->type()->AsFloat()->width() == 64 &&
           "Supported integer widths are 32 and 64.");
    first_constant_is_larger =
        unsigned_int_constant_1->GetU64() > unsigned_int_constant_2->GetU64();
  }
  std::vector<SpvOp> greater_than_opcodes{SpvOpUGreaterThan,
                                          SpvOpUGreaterThanEqual};
  std::vector<SpvOp> less_than_opcodes{SpvOpULessThan, SpvOpULessThanEqual};

  ObfuscateBoolConstantViaConstantPair(
      depth, bool_constant_use, greater_than_opcodes, less_than_opcodes,
      unsigned_int_constant_id_1, unsigned_int_constant_id_2,
      first_constant_is_larger);
}

void FuzzerPassObfuscateConstants::ObfuscateConstant(
    uint32_t depth, const protobufs::IdUseDescriptor& constant_use) {
  switch (GetIRContext()
              ->get_def_use_mgr()
              ->GetDef(constant_use.id_of_interest())
              ->opcode()) {
    case SpvOpConstantTrue:
    case SpvOpConstantFalse: {
      auto available_types_with_uniforms =
          GetFactManager()->GetTypesForWhichUniformValuesAreKnown();
      if (available_types_with_uniforms.empty()) {
        // Do not try to obfuscate if we do not have access to any uniform
        // elements with known values.
        return;
      }
      auto chosen_type_id = available_types_with_uniforms
          [GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
              static_cast<uint32_t>(available_types_with_uniforms.size()))];
      auto available_constants =
          GetFactManager()->GetConstantsAvailableFromUniformsForType(
              GetIRContext(), chosen_type_id);
      if (available_constants.size() == 1) {
        // TODO(afd): for now we only obfuscate a boolean if there are at least
        //  two constants available from uniforms, so that we can do a
        //  comparison between them. It would be good to be able to do the
        //  obfuscation even if there is only one such constant, if there is
        //  also another regular constant available.
        return;
      }
      auto constant_index_1 =
          GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
              static_cast<uint32_t>(available_constants.size()));
      uint32_t constant_index_2;
      do {
        constant_index_2 =
            GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
                static_cast<uint32_t>(available_constants.size()));
      } while (constant_index_1 == constant_index_2);
      auto constant_id_1 = available_constants[constant_index_1];
      auto constant_id_2 = available_constants[constant_index_2];
      if (constant_id_1 == 0 || constant_id_2 == 0) {
        return;
      }
      auto chosen_type =
          GetIRContext()->get_type_mgr()->GetType(chosen_type_id);
      if (chosen_type->AsFloat()) {
        ObfuscateBoolConstantViaFloatConstantPair(depth, constant_use,
                                                  constant_id_1, constant_id_2);
      } else {
        assert(chosen_type->AsInteger() &&
               "We should only have uniform facts about ints and floats.");
        if (chosen_type->AsInteger()->IsSigned()) {
          ObfuscateBoolConstantViaSignedIntConstantPair(
              depth, constant_use, constant_id_1, constant_id_2);
        } else {
          ObfuscateBoolConstantViaUnsignedIntConstantPair(
              depth, constant_use, constant_id_1, constant_id_2);
        }
      }
    } break;
    case SpvOpConstant: {
      auto uniform_descriptors =
          GetFactManager()->GetUniformDescriptorsForConstant(
              GetIRContext(), constant_use.id_of_interest());
      if (uniform_descriptors.empty()) {
        break;
      }
      protobufs::UniformBufferElementDescriptor uniform_descriptor =
          uniform_descriptors
              [GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
                  static_cast<uint32_t>(uniform_descriptors.size()))];
      auto transformation =
          transformation::MakeTransformationReplaceConstantWithUniform(
              constant_use, uniform_descriptor,
              GetFuzzerContext()->GetFreshId(),
              GetFuzzerContext()->GetFreshId());
      // Transformation should be applicable by construction.
      assert(transformation::IsApplicable(transformation, GetIRContext(),
                                          *GetFactManager()));
      transformation::Apply(transformation, GetIRContext(), GetFactManager());
      *GetTransformations()
           ->add_transformation()
           ->mutable_replace_constant_with_uniform() = transformation;
    } break;
    default:
      assert(false && "The opcode should be one of the above.");
      break;
  }
}

void FuzzerPassObfuscateConstants::Apply() {
  std::vector<protobufs::IdUseDescriptor> candidate_constant_uses;

  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      uint32_t base_instruction_result_id = block.id();
      std::map<SpvOp, uint32_t> skipped_opcode_count;
      for (auto& inst : block) {
        if (inst.HasResultId()) {
          base_instruction_result_id = inst.result_id();
          skipped_opcode_count.clear();
        }
        for (uint32_t in_operand_index = 0;
             in_operand_index < inst.NumInOperands(); in_operand_index++) {
          if (inst.GetInOperand(in_operand_index).type != SPV_OPERAND_TYPE_ID) {
            continue;
          }
          auto operand_id = inst.GetSingleWordInOperand(in_operand_index);
          auto operand_definition =
              GetIRContext()->get_def_use_mgr()->GetDef(operand_id);
          switch (operand_definition->opcode()) {
            case SpvOpConstantFalse:
            case SpvOpConstantTrue:
            case SpvOpConstant: {
              protobufs::IdUseDescriptor id_use_descriptor;
              id_use_descriptor.set_id_of_interest(operand_id);
              id_use_descriptor.set_target_instruction_opcode(inst.opcode());
              id_use_descriptor.set_in_operand_index(in_operand_index);
              id_use_descriptor.set_base_instruction_result_id(
                  base_instruction_result_id);
              id_use_descriptor.set_num_opcodes_to_ignore(
                  skipped_opcode_count.find(inst.opcode()) ==
                          skipped_opcode_count.end()
                      ? 0
                      : skipped_opcode_count[inst.opcode()]);
              candidate_constant_uses.push_back(id_use_descriptor);
            } break;
            default:
              break;
          }
        }
        if (!inst.HasResultId()) {
          skipped_opcode_count[inst.opcode()] =
              skipped_opcode_count.find(inst.opcode()) ==
                      skipped_opcode_count.end()
                  ? 0
                  : skipped_opcode_count[inst.opcode()] + 1;
        }
      }
    }
  }

  while (!candidate_constant_uses.empty()) {
    auto index = GetFuzzerContext()->GetRandomGenerator()->RandomUint32(
        static_cast<uint32_t>(candidate_constant_uses.size()));
    auto constant_use = std::move(candidate_constant_uses[index]);
    candidate_constant_uses.erase(candidate_constant_uses.begin() + index);
    if (GetFuzzerContext()->GetRandomGenerator()->RandomPercentage() >
        GetFuzzerContext()->GetChanceOfObfuscatingConstant()) {
      continue;
    }
    ObfuscateConstant(0, constant_use);
  }
}

}  // namespace fuzz
}  // namespace spvtools
