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

#include "copy_prop_arrays.h"
#include "ir_builder.h"

namespace {
const uint32_t kLoadPointerInOperand = 0;
const uint32_t kStorePointerInOperand = 0;
const uint32_t kStoreObjectInOperand = 1;
const uint32_t kCompositeExtractObjectInOperand = 0;
}  // namespace

namespace spvtools {
namespace opt {

Pass::Status CopyPropagateArrays::Process(ir::IRContext* ctx) {
  InitializeProcessing(ctx);

  bool modified = false;
  for (ir::Function& function : *get_module()) {
    ir::BasicBlock* entry_bb = &*function.begin();

    for (auto var_inst = entry_bb->begin(); var_inst->opcode() == SpvOpVariable;
         ++var_inst) {
      if (!IsPointerToArrayType(var_inst->type_id())) {
        continue;
      }

      std::unique_ptr<MemoryObject> source_object =
          FindSourceObjectIfPossible(&*var_inst);

      if (source_object != nullptr) {
        if (CanUpdateUses(&*var_inst, source_object->GetPointerTypeId())) {
          modified = true;
          PropagateObject(&*var_inst, source_object.get());
        }
      }
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

std::unique_ptr<CopyPropagateArrays::MemoryObject>
CopyPropagateArrays::FindSourceObjectIfPossible(ir::Instruction* var_inst) {
  assert(var_inst->opcode() == SpvOpVariable && "Expecting a variable.");

  // Check that the variable is a composite object with single store that
  // dominates all of its loads.

  // Find the only store to the entire memory location, if it exists.
  ir::Instruction* store_inst = nullptr;
  get_def_use_mgr()->WhileEachUser(
      var_inst, [&store_inst, var_inst](ir::Instruction* use) {
        if (use->opcode() == SpvOpStore &&
            use->GetSingleWordInOperand(kStorePointerInOperand) ==
                var_inst->result_id()) {
          if (store_inst == nullptr) {
            store_inst = use;
          } else {
            store_inst = nullptr;
            return false;
          }
        }
        return true;
      });

  if (!store_inst) {
    return nullptr;
  }

  // Look at the loads to ensure they are dominated by the store.
  if (!HasValidReferencesOnly(var_inst, store_inst)) {
    return nullptr;
  }

  // If so, look at the store to see if it is the copy of an object.
  std::unique_ptr<MemoryObject> source = GetSourceObjectIfAny(
      store_inst->GetSingleWordInOperand(kStoreObjectInOperand));

  if (!source) {
    return nullptr;
  }

  // Ensure that |source| does not change between the point at which it is
  // loaded, and the position in which |var_inst| is loaded.
  //
  // For now we will go with the easy to implement approach, and check that the
  // entire variable (not just the specific component) is never written to.

  if (!HasNoStores(source->GetVariable())) {
    return nullptr;
  }
  return source;
}

void CopyPropagateArrays::PropagateObject(ir::Instruction* var_inst,
                                          MemoryObject* source) {
  assert(var_inst->opcode() == SpvOpVariable &&
         "This function propagates variables.");

  ir::Instruction* insertion_point = var_inst->NextNode();
  while (insertion_point->opcode() == SpvOpVariable) {
    insertion_point = insertion_point->NextNode();
  }
  ir::Instruction* new_access_chain =
      BuildNewAccessChain(insertion_point, source);
  context()->KillNamesAndDecorates(var_inst);
  UpdateUses(var_inst, new_access_chain);
}

ir::Instruction* CopyPropagateArrays::BuildNewAccessChain(
    ir::Instruction* insertion_point,
    CopyPropagateArrays::MemoryObject* source) const {
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  InstructionBuilder builder(context(), insertion_point,
                             ir::IRContext::kAnalysisDefUse |
                                 ir::IRContext::kAnalysisInstrToBlockMapping);

  analysis::Integer int_type(32, false);
  const analysis::Type* uint32_type =
      context()->get_type_mgr()->GetRegisteredType(&int_type);

  // Convert the access chain in the source to a series of ids that can be used
  // by the |OpAccessChain| instruction.
  std::vector<uint32_t> index_ids;
  for (uint32_t index : source->AccessChain()) {
    const analysis::Constant* index_const =
        const_mgr->GetConstant(uint32_type, {index});
    index_ids.push_back(
        const_mgr->GetDefiningInstruction(index_const)->result_id());
  }

  // Get the type for the result of the OpAccessChain
  uint32_t pointer_type_id = source->GetPointerTypeId();

  // Build the access chain instruction.
  return builder.AddAccessChain(pointer_type_id,
                                source->GetVariable()->result_id(), index_ids);
}

bool CopyPropagateArrays::HasNoStores(ir::Instruction* ptr_inst) {
  return get_def_use_mgr()->WhileEachUser(
      ptr_inst, [this](ir::Instruction* use) {
        if (use->opcode() == SpvOpLoad) {
          return true;
        } else if (use->opcode() == SpvOpAccessChain) {
          return HasNoStores(use);
        } else if (use->IsDecoration() || use->opcode() == SpvOpName) {
          return true;
        } else if (use->opcode() == SpvOpStore) {
          return false;
        }
        // Some other instruction.  Be conservative.
        return false;
      });
}

bool CopyPropagateArrays::HasValidReferencesOnly(ir::Instruction* ptr_inst,
                                                 ir::Instruction* store_inst) {
  ir::BasicBlock* store_block = context()->get_instr_block(store_inst);
  opt::DominatorAnalysis* dominator_analysis =
      context()->GetDominatorAnalysis(store_block->GetParent(), *cfg());

  return get_def_use_mgr()->WhileEachUser(
      ptr_inst,
      [this, store_inst, dominator_analysis, ptr_inst](ir::Instruction* use) {
        if (use->opcode() == SpvOpLoad) {
          // TODO: If there are many load in the same BB as |store_inst| the
          // time to do the multiple traverses can add up.  Consider collecting
          // those loads and doing a single traversal.
          return dominator_analysis->Dominates(store_inst, use);
        } else if (use->opcode() == SpvOpAccessChain) {
          return HasValidReferencesOnly(use, store_inst);
        } else if (use->IsDecoration() || use->opcode() == SpvOpName) {
          return true;
        } else if (use->opcode() == SpvOpStore) {
          // If we are storing to part of the object it is not an candidate.
          return ptr_inst->opcode() == SpvOpVariable &&
                 store_inst->GetSingleWordInOperand(kStorePointerInOperand) ==
                     ptr_inst->result_id();
        }
        // Some other instruction.  Be conservative.
        return false;
      });
}

std::unique_ptr<CopyPropagateArrays::MemoryObject>
CopyPropagateArrays::GetSourceObjectIfAny(uint32_t result) {
  ir::Instruction* result_inst = context()->get_def_use_mgr()->GetDef(result);

  switch (result_inst->opcode()) {
    case SpvOpLoad:
      return BuildMemoryObjectFromLoad(result_inst);
    case SpvOpCompositeExtract:
      return BuildMemoryObjectFromExtract(result_inst);
    case SpvOpCompositeConstruct:
      return BuildMemoryObjectFromCompositeConstruct(result_inst);
    case SpvOpCopyObject:
      return GetSourceObjectIfAny(result_inst->GetSingleWordInOperand(0));
    default:
      return nullptr;
  }
}

std::unique_ptr<CopyPropagateArrays::MemoryObject>
CopyPropagateArrays::BuildMemoryObjectFromLoad(ir::Instruction* load_inst) {
  std::vector<uint32_t> components_in_reverse;
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  ir::Instruction* current_inst = def_use_mgr->GetDef(
      load_inst->GetSingleWordInOperand(kLoadPointerInOperand));

  // Build the access chain for the memory object by collecting the indices used
  // in the OpAccessChain instructions.  If we find a variable index, then
  // return |nullptr| because we cannot know for sure which memory location is
  // used.
  //
  // It is built in reverse order because the different |OpAccessChain|
  // instructions are visited in reverse order from which they are applied.
  while (current_inst->opcode() == SpvOpAccessChain) {
    for (uint32_t i = current_inst->NumInOperands() - 1; i >= 1; --i) {
      uint32_t element_index_id = current_inst->GetSingleWordInOperand(i);
      const analysis::Constant* element_index_const =
          const_mgr->FindDeclaredConstant(element_index_id);
      if (!element_index_const) {
        return nullptr;
      }
      assert(element_index_const->AsIntConstant());
      components_in_reverse.push_back(
          element_index_const->AsIntConstant()->GetU32());
    }
    current_inst = def_use_mgr->GetDef(current_inst->GetSingleWordInOperand(0));
  }

  // If the address in the load is not constructed from an |OpVariable|
  // instruction followed by a series of |OpAccessChain| instructions, then
  // return |nullptr| because we cannot identify the owner or access chain
  // exactly.
  if (current_inst->opcode() != SpvOpVariable) {
    return nullptr;
  }

  // Build the memory object.  Use |rbegin| and |rend| to put the access chain
  // back in the correct order.
  return std::unique_ptr<CopyPropagateArrays::MemoryObject>(
      new MemoryObject(current_inst, components_in_reverse.rbegin(),
                       components_in_reverse.rend()));
}

std::unique_ptr<CopyPropagateArrays::MemoryObject>
CopyPropagateArrays::BuildMemoryObjectFromExtract(
    ir::Instruction* extract_inst) {
  assert(extract_inst->opcode() == SpvOpCompositeExtract &&
         "Expecting an OpCompositeExtract instruction.");

  std::unique_ptr<MemoryObject> result = GetSourceObjectIfAny(
      extract_inst->GetSingleWordInOperand(kCompositeExtractObjectInOperand));

  if (result) {
    std::vector<uint32_t> components;
    for (uint32_t i = 1; i < extract_inst->NumInOperands(); ++i) {
      components.emplace_back(extract_inst->GetSingleWordInOperand(i));
    }
    result->GetMember(components);
    return result;
  }
  return nullptr;
}

std::unique_ptr<CopyPropagateArrays::MemoryObject>
CopyPropagateArrays::BuildMemoryObjectFromCompositeConstruct(
    ir::Instruction* conststruct_inst) {
  assert(conststruct_inst->opcode() == SpvOpCompositeConstruct &&
         "Expecting an OpCompositeConstruct instruction.");

  // If every operand in the instruction are part of the same memory object, and
  // are being combined in the same order, then the result is the same as the
  // parent.

  std::unique_ptr<MemoryObject> memory_object =
      GetSourceObjectIfAny(conststruct_inst->GetSingleWordInOperand(0));

  if (!memory_object) {
    return nullptr;
  }

  if (!memory_object->IsMember()) {
    return nullptr;
  }

  if (memory_object->AccessChain().back() != 0) {
    return nullptr;
  }

  memory_object->GetParent();

  if (memory_object->GetNumberOfMembers() !=
      conststruct_inst->NumInOperands()) {
    return nullptr;
  }

  for (uint32_t i = 1; i < conststruct_inst->NumInOperands(); ++i) {
    std::unique_ptr<MemoryObject> member_object =
        GetSourceObjectIfAny(conststruct_inst->GetSingleWordInOperand(i));

    if (member_object->GetVariable() != memory_object->GetVariable()) {
      return nullptr;
    }

    if (member_object->AccessChain().back() != i) {
      return nullptr;
    }
  }
  return memory_object;
}

bool CopyPropagateArrays::IsPointerToArrayType(uint32_t type_id) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Pointer* pointer_type = type_mgr->GetType(type_id)->AsPointer();
  if (pointer_type) {
    return pointer_type->pointee_type()->AsArray() != nullptr;
  }
  return false;
}

bool CopyPropagateArrays::CanUpdateUses(ir::Instruction* original_ptr_inst,
                                        uint32_t type_id) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  return def_use_mgr->WhileEachUse(original_ptr_inst, [this, type_id, type_mgr,
                                                       const_mgr](
                                                          ir::Instruction* use,
                                                          uint32_t index) {
    analysis::Pointer* pointer_type = nullptr;
    switch (use->opcode()) {
      case SpvOpLoad: {
        pointer_type = type_mgr->GetType(type_id)->AsPointer();
        uint32_t new_type_id = type_mgr->GetId(pointer_type->pointee_type());

        if (new_type_id != use->type_id()) {
          return CanUpdateUses(use, new_type_id);
        }
        return true;
      }
      case SpvOpAccessChain: {
        pointer_type = type_mgr->GetType(type_id)->AsPointer();
        const analysis::Type* pointee_type = pointer_type->pointee_type();

        std::vector<uint32_t> access_chain;
        for (uint32_t i = 1; i < use->NumInOperands(); ++i) {
          const analysis::Constant* index_const =
              const_mgr->FindDeclaredConstant(use->GetSingleWordInOperand(i));
          if (index_const) {
            access_chain.push_back(index_const->AsIntConstant()->GetU32());
          } else {
            // Variable index means the type is a type where every element
            // is the same type.  Use element 0 to get the type.
            access_chain.push_back(0);
          }
        }

        const analysis::Type* new_pointee_type =
            type_mgr->GetMemberType(pointee_type, access_chain);
        opt::analysis::Pointer pointerTy(new_pointee_type,
                                         pointer_type->storage_class());
        uint32_t new_pointer_type_id =
            context()->get_type_mgr()->GetTypeInstruction(&pointerTy);

        if (new_pointer_type_id != use->type_id()) {
          return CanUpdateUses(use, new_pointer_type_id);
        }
        return true;
      }
      case SpvOpCompositeExtract: {
        std::vector<uint32_t> access_chain;
        for (uint32_t i = 1; i < use->NumInOperands(); ++i) {
          const analysis::Constant* index_const =
              const_mgr->FindDeclaredConstant(use->GetSingleWordInOperand(i));
          if (index_const) {
            access_chain.push_back(index_const->AsIntConstant()->GetU32());
          } else {
            // Variable index means the type is an type where every element
            // is the same type.  Use element 0 to get the type.
            access_chain.push_back(0);
          }
        }

        const analysis::Type* type = type_mgr->GetType(type_id);
        const analysis::Type* new_type =
            type_mgr->GetMemberType(type, access_chain);
        uint32_t new_type_id = type_mgr->GetTypeInstruction(new_type);

        if (new_type_id != use->type_id()) {
          return CanUpdateUses(use, new_type_id);
        }
        return true;
      }
      case SpvOpStore:
        // Can't handle changing the type of a store.  There are too many other
        // things that might need to change as well.  Not worth the effort.
        // Punting for now.

        // TODO (s-perron): This can be handled by expanding the store into a
        // series of extracts, composite constructs, and a store.
        return index != 1;
      default:
        return false;
    }
  });
}
void CopyPropagateArrays::UpdateUses(ir::Instruction* original_ptr_inst,
                                     ir::Instruction* new_ptr_inst) {
  // TODO (s-perron): Keep the def-use manager up to date.  Not done now because
  // it can cause problems for the |ForEachUse| traversals.  Can be use by
  // keeping a list of instructions that need updating, and then updating them
  // in |PropagateObject|.

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  def_use_mgr->ForEachUse(original_ptr_inst, [this, new_ptr_inst, type_mgr,
                                              const_mgr](ir::Instruction* use,
                                                         uint32_t index) {
    analysis::Pointer* pointer_type = nullptr;
    switch (use->opcode()) {
      case SpvOpLoad: {
        // Replace the actual use.
        use->SetOperand(index, {new_ptr_inst->result_id()});

        // Update the type.
        pointer_type = type_mgr->GetType(new_ptr_inst->type_id())->AsPointer();
        uint32_t new_type_id = type_mgr->GetId(pointer_type->pointee_type());
        if (new_type_id != use->type_id()) {
          use->SetResultType(new_type_id);
          UpdateUses(use, use);
        }
      } break;
      case SpvOpAccessChain: {
        // Update the actual use.
        use->SetOperand(index, {new_ptr_inst->result_id()});

        // Update the result type.
        pointer_type = type_mgr->GetType(new_ptr_inst->type_id())->AsPointer();
        const analysis::Type* pointee_type = pointer_type->pointee_type();

        // Convert the ids on the OpAccessChain to indices that can be used to
        // get the specific member.
        std::vector<uint32_t> access_chain;
        for (uint32_t i = 1; i < use->NumInOperands(); ++i) {
          const analysis::Constant* index_const =
              const_mgr->FindDeclaredConstant(use->GetSingleWordInOperand(i));
          if (index_const) {
            access_chain.push_back(index_const->AsIntConstant()->GetU32());
          } else {
            // Variable index means the type is an type where every element
            // is the same type.  Use element 0 to get the type.
            access_chain.push_back(0);
          }
        }
        const analysis::Type* new_pointee_type =
            type_mgr->GetMemberType(pointee_type, access_chain);

        // Now build a pointer to the type of the member.
        opt::analysis::Pointer new_pointer_type(new_pointee_type,
                                                pointer_type->storage_class());
        uint32_t new_pointer_type_id =
            context()->get_type_mgr()->GetTypeInstruction(&new_pointer_type);

        if (new_pointer_type_id != use->type_id()) {
          use->SetResultType(new_pointer_type_id);
          UpdateUses(use, use);
        }
      } break;
      case SpvOpCompositeExtract: {
        // Update the actual use.
        use->SetOperand(index, {new_ptr_inst->result_id()});

        std::vector<uint32_t> access_chain;
        for (uint32_t i = 1; i < use->NumInOperands(); ++i) {
          access_chain.push_back(use->GetSingleWordInOperand(i));
        }

        const analysis::Type* type = type_mgr->GetType(new_ptr_inst->type_id());
        const analysis::Type* new_type =
            type_mgr->GetMemberType(type, access_chain);
        uint32_t new_type_id = type_mgr->GetTypeInstruction(new_type);

        if (new_type_id != use->type_id()) {
          use->SetResultType(new_type_id);
          UpdateUses(use, use);
        }
      } break;
      case SpvOpStore:
        // If the use is the pointer, then it is the single store to that
        // variable.  We do not want to replace it.  Instead, it will become
        // dead after all of the loads are removed, and ADCE will get rid of it.
        //
        // If the use is the object being stored, we do not know how to change
        // the type, so we assume that |CanUpdateUse| would have returned false,
        // and we should not have called this function.
        assert(index != 1 && "Have to change the type of the stored object.");
        break;

      default:
        assert(false && "Don't know how to rewrite instruction");
        break;
    }
  });
}

void CopyPropagateArrays::MemoryObject::GetMember(
    const std::vector<uint32_t>& access_chain) {
  access_chain_.insert(access_chain_.end(), access_chain.begin(),
                       access_chain.end());
}

uint32_t CopyPropagateArrays::MemoryObject::GetNumberOfMembers() {
  ir::IRContext* context = variable_inst_->context();
  analysis::TypeManager* type_mgr = context->get_type_mgr();

  const analysis::Type* type = type_mgr->GetType(variable_inst_->type_id());
  type = type->AsPointer()->pointee_type();
  type = type_mgr->GetMemberType(type, access_chain_);

  if (const analysis::Struct* struct_type = type->AsStruct()) {
    return static_cast<uint32_t>(struct_type->element_types().size());
  } else if (const analysis::Array* array_type = type->AsArray()) {
    const analysis::Constant* length_const =
        context->get_constant_mgr()->FindDeclaredConstant(
            array_type->LengthId());
    assert(length_const->AsIntConstant());
    return length_const->AsIntConstant()->GetU32();
  } else if (const analysis::Vector* vector_type = type->AsVector()) {
    return vector_type->element_count();
  } else if (const analysis::Matrix* matrix_type = type->AsMatrix()) {
    return matrix_type->element_count();
  } else {
    return 0;
  }
}

template <class iterator>
CopyPropagateArrays::MemoryObject::MemoryObject(ir::Instruction* var_inst,
                                                iterator begin, iterator end)
    : variable_inst_(var_inst), access_chain_(begin, end) {}

}  // namespace opt
}  // namespace spvtools
