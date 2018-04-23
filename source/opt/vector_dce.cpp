// Copyright (c) 2018 Google LLC.
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

#include "vector_dce.h"

namespace {
const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;
}  // namespace

namespace spvtools {
namespace opt {

Pass::Status VectorDCE::Process(ir::IRContext* ctx) {
  InitializeProcessing(ctx);

  bool modified = false;
  for (ir::Function& function : *get_module()) {
    modified |= VectorDCEFunction(&function);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool VectorDCE::VectorDCEFunction(ir::Function* function) {
  LiveComponentMap live_components;
  FindLiveComponents(function, &live_components);
  return RewriteInstructions(function, live_components);
}

void VectorDCE::FindLiveComponents(ir::Function* function,
                                   LiveComponentMap* live_components) {
  std::vector<WorkListItem> work_list;

  // Prime the work list.  We will assume that any instruction that does
  // not result in a vector is live.
  //
  // TODO: If we want to remove all of the dead code with the current
  // implementation, we will have to iterate VectorDCE and ADCE.  That can be
  // avoided if we treat scalar results as if they are vectors with 1 element.
  //
  // Extending to structures and matrices is not as straight forward because of
  // the nesting.  We cannot simply us a bit vector to keep track of which
  // components are live because of arbitrary nesting of structs.
  function->ForEachInst(
      [&work_list, this, live_components](ir::Instruction* current_inst) {
        bool is_vector = HasVectorResult(current_inst);
        if (!is_vector) {
          switch (current_inst->opcode()) {
            case SpvOpCompositeExtract:
              MarkExtractUseAsLive(current_inst, live_components, &work_list);
              break;

            default:
              MarkUsesAsLive(current_inst, all_components_live_,
                             live_components, &work_list);
              break;
          }
        }
      });

  // Process the work list propagating liveness.
  for (uint32_t i = 0; i < work_list.size(); i++) {
    WorkListItem current_item = work_list[i];
    ir::Instruction* current_inst = current_item.instruction;

    switch (current_inst->opcode()) {
      case SpvOpCompositeInsert:
        MarkInsertUsesAsLive(current_item, live_components, &work_list);
        break;
      case SpvOpVectorShuffle:
        MarkVectorShuffleUsesAsLive(current_item, live_components, &work_list);
        break;
      case SpvOpCompositeConstruct:
        // TODO: If we have a vector parameter, then we can determine which
        // of its components are live.
        MarkUsesAsLive(current_inst, all_components_live_, live_components,
                       &work_list);
        break;
      default:
        if (spvOpcodeIsScalarizable(current_inst->opcode())) {
          MarkUsesAsLive(current_inst, current_item.components, live_components,
                         &work_list);
        } else {
          MarkUsesAsLive(current_inst, all_components_live_, live_components,
                         &work_list);
        }
        break;
    }
  }
}

void VectorDCE::MarkExtractUseAsLive(const ir::Instruction* current_inst,
                                     LiveComponentMap* live_components,
                                     std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t operand_id =
      current_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
  ir::Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

  if (HasVectorResult(operand_inst)) {
    WorkListItem new_item;
    new_item.instruction = operand_inst;
    new_item.components.Set(current_inst->GetSingleWordInOperand(1));
    AddItemToWorkListIfNeeded(new_item, live_components, work_list);
  }
}

void VectorDCE::MarkInsertUsesAsLive(
    const VectorDCE::WorkListItem& current_item,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  uint32_t operand_id =
      current_item.instruction->GetSingleWordInOperand(kInsertCompositeIdInIdx);
  ir::Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

  WorkListItem new_item;
  new_item.instruction = operand_inst;
  new_item.components = current_item.components;
  new_item.components.Clear(
      current_item.instruction->GetSingleWordInOperand(2));

  AddItemToWorkListIfNeeded(new_item, live_components, work_list);
}

void VectorDCE::MarkVectorShuffleUsesAsLive(
    const WorkListItem& current_item,
    VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  WorkListItem first_operand;
  first_operand.instruction =
      def_use_mgr->GetDef(current_item.instruction->GetSingleWordInOperand(0));
  WorkListItem second_operand;
  second_operand.instruction =
      def_use_mgr->GetDef(current_item.instruction->GetSingleWordInOperand(1));

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Vector* first_type =
      type_mgr->GetType(first_operand.instruction->type_id())->AsVector();
  uint32_t size_of_first_operand = first_type->element_count();

  for (uint32_t in_op = 2; in_op < current_item.instruction->NumInOperands();
       ++in_op) {
    uint32_t index = current_item.instruction->GetSingleWordInOperand(in_op);
    if (current_item.components.Get(in_op - 2)) {
      if (index < size_of_first_operand) {
        first_operand.components.Set(index);
      } else {
        second_operand.components.Set(index - size_of_first_operand);
      }
    }
  }

  AddItemToWorkListIfNeeded(first_operand, live_components, work_list);
  AddItemToWorkListIfNeeded(second_operand, live_components, work_list);
}

void VectorDCE::MarkUsesAsLive(
    ir::Instruction* current_inst, const utils::BitVector& live_elements,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  current_inst->ForEachInId([&work_list, &live_elements, this, live_components,
                             def_use_mgr](uint32_t* operand_id) {
    ir::Instruction* operand_inst = def_use_mgr->GetDef(*operand_id);

    if (HasVectorResult(operand_inst)) {
      WorkListItem new_item;
      new_item.instruction = operand_inst;
      new_item.components = live_elements;
      AddItemToWorkListIfNeeded(new_item, live_components, work_list);
    }
  });
}

bool VectorDCE::HasVectorResult(const ir::Instruction* inst) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  if (inst->type_id() == 0) {
    return false;
  }

  const analysis::Type* current_type = type_mgr->GetType(inst->type_id());
  bool is_vector = current_type->AsVector() != nullptr;
  return is_vector;
}

bool VectorDCE::RewriteInstructions(
    ir::Function* function,
    const VectorDCE::LiveComponentMap& live_components) {
  bool modified = false;
  function->ForEachInst(
      [&modified, this, live_components](ir::Instruction* current_inst) {
        if (!context()->IsCombinatorInstruction(current_inst)) {
          return;
        }

        auto live_component = live_components.find(current_inst->result_id());
        if (live_component == live_components.end()) {
          // If this instruction is not in live_components then it does not
          // produce a vector, or it is never referenced and ADCE will remove
          // it.  No point in trying to differentiate.
          return;
        }

        // If no element in the current instruction is used replace it with an
        // OpUndef.
        if (live_component->second.Empty()) {
          modified = true;
          uint32_t undef_id = this->Type2Undef(current_inst->type_id());
          context()->KillNamesAndDecorates(current_inst);
          context()->ReplaceAllUsesWith(current_inst->result_id(), undef_id);
          context()->KillInst(current_inst);
          return;
        }

        switch (current_inst->opcode()) {
          case SpvOpCompositeInsert: {
            // If the value being inserted is not live, then we can skip the
            // insert.
            uint32_t insert_index = current_inst->GetSingleWordInOperand(2);
            if (!live_component->second.Get(insert_index)) {
              modified = true;
              context()->KillNamesAndDecorates(current_inst->result_id());
              uint32_t composite_id =
                  current_inst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
              context()->ReplaceAllUsesWith(current_inst->result_id(),
                                            composite_id);
            }
          } break;
          case SpvOpCompositeConstruct:
            // TODO: The members that are not live can be replaced by an undef
            // or constant. This will remove uses of those values, and possibly
            // create opportunities for ADCE.
            break;
          default:
            // Do nothing.
            break;
        }
      });
  return modified;
}

void VectorDCE::AddItemToWorkListIfNeeded(
    WorkListItem work_item, VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  ir::Instruction* current_inst = work_item.instruction;
  auto it = live_components->find(current_inst->result_id());
  if (it == live_components->end()) {
    live_components->emplace(
        std::make_pair(current_inst->result_id(), work_item.components));
    work_list->emplace_back(work_item);
  } else {
    if (it->second.Or(work_item.components)) {
      work_list->emplace_back(work_item);
    }
  }
}

}  // namespace opt
}  // namespace spvtools
