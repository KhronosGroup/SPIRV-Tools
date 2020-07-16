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

#include "source/fuzz/transformation_access_chain.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

namespace {
// Given a context, a value and a type id, returns the id of an integer
// constant with the same type and given value.
// The type id must correspond to an integer type.
uint32_t FindIntConstant(opt::IRContext* ir_context, uint32_t value,
                         uint32_t int_type_id) {
  auto int_type_inst = ir_context->get_def_use_mgr()->GetDef(int_type_id);

  assert(int_type_inst && "The given type id must exist.");

  auto int_type = ir_context->get_type_mgr()
                      ->GetType(int_type_inst->result_id())
                      ->AsInteger();

  assert(int_type && "The given type id must correspond to an integer type.");

  opt::analysis::IntConstant bound_minus_one(int_type, {value});

  // Check that the constant exists in the module
  if (!ir_context->get_constant_mgr()->FindConstant(&bound_minus_one)) {
    return 0;
  }

  return ir_context->get_constant_mgr()
      ->GetDefiningInstruction(&bound_minus_one)
      ->result_id();
}

// TODO: This is copied from TransformationAddFunction. Move it somewhere
// where it can be accessed by both.
uint32_t GetBoundForCompositeIndex(
    opt::IRContext* ir_context, const opt::Instruction& composite_type_inst) {
  switch (composite_type_inst.opcode()) {
    case SpvOpTypeArray:
      return fuzzerutil::GetArraySize(composite_type_inst, ir_context);
    case SpvOpTypeMatrix:
    case SpvOpTypeVector:
      return composite_type_inst.GetSingleWordInOperand(1);
    case SpvOpTypeStruct: {
      return fuzzerutil::GetNumberOfStructMembers(composite_type_inst);
    }
    case SpvOpTypeRuntimeArray:
      assert(false &&
             "GetBoundForCompositeIndex should not be invoked with an "
             "OpTypeRuntimeArray, which does not have a static bound.");
      return 0;
    default:
      assert(false && "Unknown composite type.");
      return 0;
  }
}

}  // namespace

TransformationAccessChain::TransformationAccessChain(
    const spvtools::fuzz::protobufs::TransformationAccessChain& message)
    : message_(message) {}

TransformationAccessChain::TransformationAccessChain(
    uint32_t fresh_id, uint32_t pointer_id,
    const std::vector<uint32_t>& index_id,
    const protobufs::InstructionDescriptor& instruction_to_insert_before,
    const std::vector<uint32_t>& fresh_id_for_clamping) {
  message_.set_fresh_id(fresh_id);
  message_.set_pointer_id(pointer_id);
  for (auto id : index_id) {
    message_.add_index_id(id);
  }
  *message_.mutable_instruction_to_insert_before() =
      instruction_to_insert_before;
  for (auto clamping_id : fresh_id_for_clamping) {
    message_.add_fresh_id_for_clamping(clamping_id);
  }
}

bool TransformationAccessChain::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The result id must be fresh
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The pointer id must exist and have a type.
  auto pointer = ir_context->get_def_use_mgr()->GetDef(message_.pointer_id());
  if (!pointer || !pointer->type_id()) {
    return false;
  }
  // The type must indeed be a pointer
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(pointer->type_id());
  if (pointer_type->opcode() != SpvOpTypePointer) {
    return false;
  }

  // The described instruction to insert before must exist and be a suitable
  // point where an OpAccessChain instruction could be inserted.
  auto instruction_to_insert_before =
      FindInstruction(message_.instruction_to_insert_before(), ir_context);
  if (!instruction_to_insert_before) {
    return false;
  }
  if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
          SpvOpAccessChain, instruction_to_insert_before)) {
    return false;
  }

  // Do not allow making an access chain from a null or undefined pointer, as
  // we do not want to allow accessing such pointers.  This might be acceptable
  // in dead blocks, but we conservatively avoid it.
  switch (pointer->opcode()) {
    case SpvOpConstantNull:
    case SpvOpUndef:
      assert(
          false &&
          "Access chains should not be created from null/undefined pointers");
      return false;
    default:
      break;
  }

  // The pointer on which the access chain is to be based needs to be available
  // (according to dominance rules) at the insertion point.
  if (!fuzzerutil::IdIsAvailableBeforeInstruction(
          ir_context, instruction_to_insert_before, message_.pointer_id())) {
    return false;
  }

  // We now need to use the given indices to walk the type structure of the
  // base type of the pointer, making sure that (a) the indices correspond to
  // integers, and (b) these integer values are in-bounds.

  // Start from the base type of the pointer.
  uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);

  uint32_t clamping_ids_used = 0;

  // Consider the given index ids in turn.
  for (auto index_id : message_.index_id()) {
    // Try to get the integer value associated with this index is.  The first
    // component of the result will be false if the id did not correspond to an
    // integer.  Otherwise, the integer with which the id is associated is the
    // second component.
    bool found_index_value;
    uint32_t index_value;

    std::vector<uint32_t> clamping_ids;

    // If the index is not a constant, we need to use two ids for clamping
    if (!ir_context->get_constant_mgr()->FindDeclaredConstant(index_id)) {
      if (message_.fresh_id_for_clamping().size() - clamping_ids_used < 2) {
        // We don't have enough ids
        return false;
      }

      // Get two new ids to use and update the amount used
      clamping_ids.push_back(clamping_ids_used++);
      clamping_ids.push_back(clamping_ids_used++);
    }

    std::tie(found_index_value, index_value, std::ignore) = GetIndexValueAndId(
        ir_context, index_id, subobject_type_id, false, clamping_ids);

    if (!found_index_value) {
      // This index cannot be used
      return false;
    }

    // Try to walk down the type using this index.  This will yield 0 if the
    // type is not a composite or the index is out of bounds, and the id of
    // the next type otherwise.
    subobject_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
        ir_context, subobject_type_id, index_value);
    if (!subobject_type_id) {
      // Either the type was not a composite (so that too many indices were
      // provided), or the index was out of bounds.
      return false;
    }
  }
  // At this point, |subobject_type_id| is the type of the value targeted by
  // the new access chain.  The result type of the access chain should be a
  // pointer to this type, with the same storage class as for the original
  // pointer.  Such a pointer type needs to exist in the module.
  //
  // We do not use the type manager to look up this type, due to problems
  // associated with pointers to isomorphic structs being regarded as the same.
  return fuzzerutil::MaybeGetPointerType(
             ir_context, subobject_type_id,
             static_cast<SpvStorageClass>(
                 pointer_type->GetSingleWordInOperand(0))) != 0;
}

void TransformationAccessChain::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // The operands to the access chain are the pointer followed by the indices.
  // The result type of the access chain is determined by where the indices
  // lead.  We thus push the pointer to a sequence of operands, and then follow
  // the indices, pushing each to the operand list and tracking the type
  // obtained by following it.  Ultimately this yields the type of the
  // component reached by following all the indices, and the result type is
  // a pointer to this component type.
  opt::Instruction::OperandList operands;

  // Add the pointer id itself.
  operands.push_back({SPV_OPERAND_TYPE_ID, {message_.pointer_id()}});

  // Start walking the indices, starting with the pointer's base type.
  auto pointer_type = ir_context->get_def_use_mgr()->GetDef(
      ir_context->get_def_use_mgr()->GetDef(message_.pointer_id())->type_id());
  uint32_t subobject_type_id = pointer_type->GetSingleWordInOperand(1);

  uint32_t clamping_ids_used = 0;

  // Go through the index ids in turn.
  for (auto index_id : message_.index_id()) {
    std::vector<uint32_t> clamping_ids;

    uint32_t index_value;
    uint32_t new_index_id;

    if (!ir_context->get_constant_mgr()->FindDeclaredConstant(index_id)) {
      // Get two new ids to use and update the amount used
      clamping_ids.push_back(clamping_ids_used++);
      clamping_ids.push_back(clamping_ids_used++);
    }

    // Get the integer value associated with the index id.
    std::tie(std::ignore, index_value, new_index_id) = GetIndexValueAndId(
        ir_context, index_id, subobject_type_id, true, clamping_ids);

    // Add the correct index id to the operands.
    operands.push_back({SPV_OPERAND_TYPE_ID, {new_index_id}});

    // Walk to the next type in the composite object using this index.
    subobject_type_id = fuzzerutil::WalkOneCompositeTypeIndex(
        ir_context, subobject_type_id, index_value);
  }
  // The access chain's result type is a pointer to the composite component that
  // was reached after following all indices.  The storage class is that of the
  // original pointer.
  uint32_t result_type = fuzzerutil::MaybeGetPointerType(
      ir_context, subobject_type_id,
      static_cast<SpvStorageClass>(pointer_type->GetSingleWordInOperand(0)));

  // Add the access chain instruction to the module, and update the module's id
  // bound.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  FindInstruction(message_.instruction_to_insert_before(), ir_context)
      ->InsertBefore(MakeUnique<opt::Instruction>(
          ir_context, SpvOpAccessChain, result_type, message_.fresh_id(),
          operands));

  // Conservatively invalidate all analyses.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // If the base pointer's pointee value was irrelevant, the same is true of the
  // pointee value of the result of this access chain.
  if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
          message_.pointer_id())) {
    transformation_context->GetFactManager()->AddFactValueOfPointeeIsIrrelevant(
        message_.fresh_id());
  }
}

protobufs::Transformation TransformationAccessChain::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_access_chain() = message_;
  return result;
}

std::tuple<bool, uint32_t, uint32_t>
TransformationAccessChain::GetIndexValueAndId(
    opt::IRContext* ir_context, uint32_t index_id, uint32_t object_type_id,
    bool add_clamping_instructions, std::vector<uint32_t> fresh_ids) const {
  auto object_type_def = ir_context->get_def_use_mgr()->GetDef(object_type_id);
  // The object being indexed must be a composite
  if (!spvOpcodeIsComposite(object_type_def->opcode())) {
    return {false, 0, 0};
  }

  // Get the defining instruction of the index
  auto index_instruction = ir_context->get_def_use_mgr()->GetDef(index_id);
  if (!index_instruction) {
    return {false, 0, 0};
  }

  // The index type must be 32-bit integer
  auto index_type =
      ir_context->get_def_use_mgr()->GetDef(index_instruction->type_id());
  if (index_type->opcode() != SpvOpTypeInt ||
      index_type->GetSingleWordInOperand(0) != 32) {
    return {false, 0, 0};
  }

  uint32_t bound = GetBoundForCompositeIndex(
      ir_context, *ir_context->get_def_use_mgr()->GetDef(object_type_id));

  // If the index is a constant, just get its value if it is in bounds
  if (spvOpcodeIsConstant(index_instruction->opcode())) {
    uint32_t value = index_instruction->GetSingleWordInOperand(0);

    if (value < bound) {
      return {true, value, index_id};
    }

    // The constant is out of bound. We need to use a constant with value
    // bound-1
    uint32_t bound_minus_one_id =
        FindIntConstant(ir_context, bound - 1, index_instruction->type_id());
    if (bound_minus_one_id == 0) {
      // Constant with value bound-1 not found
      return {false, 0, 0};
    }

    return {true, bound - 1, bound_minus_one_id};
  }

  // The index is not a constant

  // Structs can only be accessed via constants
  if (object_type_def->opcode() == SpvOpTypeStruct) {
    return {false, 0, 0};
  }

  // The index is not a constant or it is not in bounds

  // We need at least two fresh ids to clamp the index variable
  if (fresh_ids.size() < 2) {
    return {false, 0, 0};
  }

  // Perform the clamping using the fresh ids at our disposal.
  // The module will not be changed if |add_clamping_instructions| is not set.
  if (!TryToClampIntVariable(ir_context, *index_instruction, bound /*bound*/,
                             std::make_pair(fresh_ids[0], fresh_ids[1]),
                             add_clamping_instructions)) {
    // It was not possible to clamp the variable
    return {false, 0, 0};
  }

  // The clamped variable will be at id |fresh_ids[1]|
  return {true, 0, fresh_ids[1]};
}

bool TransformationAccessChain::TryToClampIntVariable(
    opt::IRContext* ir_context, const opt::Instruction& int_inst,
    uint32_t bound, std::pair<uint32_t, uint32_t> fresh_ids,
    bool add_clamping_instructions) const {
  // The module must have an integer constant of value the bound - 1
  auto bound_minus_one_id =
      FindIntConstant(ir_context, bound - 1, int_inst.type_id());
  if (!bound_minus_one_id) {
    return false;
  }

  // The module must have the definition of bool type to make a comparison
  opt::analysis::Bool bool_type;
  uint32_t bool_type_id = ir_context->get_type_mgr()->GetId(&bool_type);
  if (!bool_type_id) {
    return false;
  }

  auto int_type_inst =
      ir_context->get_def_use_mgr()->GetDef(int_inst.type_id());

  // Clamp the variable and add the corresponding instructions in the module
  // if |add_clamping_instructions| is set
  if (add_clamping_instructions) {
    auto instruction_to_insert_before =
        FindInstruction(message_.instruction_to_insert_before(), ir_context);

    // Compare the index with the bound via an instruction of the form:
    //   %fresh_ids.first = OpULessThanEqual %bool %int_id %bound_minus_one
    fuzzerutil::UpdateModuleIdBound(ir_context, fresh_ids.first);
    instruction_to_insert_before->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpULessThanEqual, bool_type_id, fresh_ids.first,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {int_inst.result_id()}},
             {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));

    // Select the index if in-bounds, otherwise one less than the bound:
    //   %fresh_ids.second = OpSelect %int_type %fresh_ids.first %int_id
    //                           %bound_minus_one
    fuzzerutil::UpdateModuleIdBound(ir_context, fresh_ids.second);
    instruction_to_insert_before->InsertBefore(MakeUnique<opt::Instruction>(
        ir_context, SpvOpSelect, int_type_inst->result_id(), fresh_ids.second,
        opt::Instruction::OperandList(
            {{SPV_OPERAND_TYPE_ID, {fresh_ids.first}},
             {SPV_OPERAND_TYPE_ID, {int_inst.result_id()}},
             {SPV_OPERAND_TYPE_ID, {bound_minus_one_id}}})));
  }

  return true;
}

}  // namespace fuzz
}  // namespace spvtools
