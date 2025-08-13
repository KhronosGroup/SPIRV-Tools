// Copyright (c) 2025 Lee Gao
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

#include "source/opt/remove_clip_cull_dist_pass.h"

#include "source/opt/ir_builder.h"
#include "source/util/hex_float.h"

#define LOG(fmt, ...) consumer()(SPV_MSG_INFO, __FUNCTION__, {__LINE__}, "lower-clip-cull-dist: " fmt, ## __VA_ARGS__);
#define LOGE(fmt, ...) consumer()(SPV_MSG_ERROR, __FUNCTION__, {__LINE__}, "lower-clip-cull-dist: " fmt, ## __VA_ARGS__);

namespace spvtools {
namespace opt {

namespace {
constexpr int kEntryPointExecutionModelInIdx = 0;
constexpr int kEntryPointFunctionIdInIdx = 1;
constexpr int kEntryPointInterfaceInIdx = 3;
constexpr int kBuiltInDecorationInIdx = 2;
constexpr int kMemberBuiltInDecorationInIdx = 3;
constexpr int kMemberIndexInIdx = 1;
}  // namespace

Pass::Status LowerClipCullDistancePass::Process() {
  // Prerequisite checks.
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    return Status::SuccessWithoutChange;
  }
  if (context()->get_feature_mgr()->HasCapability(
          spv::Capability::VariablePointers) ||
      context()->get_feature_mgr()->HasCapability(
          spv::Capability::VariablePointersStorageBuffer)) {
    LOGE("This pass does not support VariablePointers capabilities.");
    return Status::Failure;
  }

  std::unordered_set<Instruction*> dead_builtins;
  bool changed = false;
  for (auto& entry_point : get_module()->entry_points()) {
    LowerClipCullDistancePass::PassStatus status = ProcessEntryPoint(&entry_point, &dead_builtins);
    if (status == EMULATED || status == CLEANUP_ONLY) {
      changed = true;
    }
  }

  bool needs_cleanup = Cleanup();
  if (needs_cleanup) {
    LOG("Clip/CullDistance cleanup was performed");
  }

  return changed || needs_cleanup ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

LowerClipCullDistancePass::PassStatus LowerClipCullDistancePass::ProcessEntryPoint(
    Instruction* entry_point,
    std::unordered_set<Instruction*>* dead_builtins) {
  LowerClipCullDistancePass::PassStatus result = NO_CHANGES;
  spv::ExecutionModel exec_model = static_cast<spv::ExecutionModel>(
      entry_point->GetSingleWordInOperand(kEntryPointExecutionModelInIdx));

  switch (exec_model) {
    case spv::ExecutionModel::Vertex:
    case spv::ExecutionModel::TessellationEvaluation:
    case spv::ExecutionModel::Geometry:
    case spv::ExecutionModel::TessellationControl:
    case spv::ExecutionModel::Fragment:
      break;
    default:
      return result;
  }

  BuiltinVariableInfo builtins;
  result = FindBuiltinVariables(entry_point, &builtins);
  if (result == NO_CHANGES) {
    return result;
  }

  std::vector<Instruction*> stores_to_process;

  if (builtins.clip_dist_var) {
    FindRelevantStores(builtins.clip_dist_var, &stores_to_process);
    dead_builtins->insert(builtins.clip_dist_var);
  }
  if (builtins.cull_dist_var) {
    FindRelevantStores(builtins.cull_dist_var, &stores_to_process);
    dead_builtins->insert(builtins.cull_dist_var);
  }

  result = CLEANUP_ONLY;
  // for (Instruction* store : stores_to_process) {
  //   // InjectClippingCode(store, builtins, exec_model);
  //   // result = EMULATED;
  // }
  
  if (builtins.clip_dist_var) {
    LOG("Killing dead ClipDistance OpDecorate instruction");
    CleanupVariable(entry_point, builtins.clip_dist_var);
  }
  if (builtins.cull_dist_var) {
    LOG("Killing dead CullDistance OpDecorate instruction");
    CleanupVariable(entry_point, builtins.cull_dist_var);
  }

  // TODO: this causes spirv-val failure that crashes in vvl for some reason
  // if (builtins.clip_dist_mem_decoration) {
  //   LOG("Killing dead ClipDistance OpMemberDecorate instruction");
  //   context()->KillInst(builtins.clip_dist_mem_decoration);
  // }
  // if (builtins.cull_dist_mem_decoration) {
  //   LOG("Killing dead CullDistance OpMemberDecorate instruction");
  //   context()->KillInst(builtins.cull_dist_mem_decoration);
  // }

  return CLEANUP_ONLY;
}

void LowerClipCullDistancePass::CleanupVariable(Instruction* entry_point, Instruction* var) {
  std::vector<Instruction*> users;
  FindAllUses(var, &users);
  for (Instruction* store : users) {
    context()->KillInst(store);
  }

  std::vector<Operand> new_operands;
  for (uint32_t i = 0; i < entry_point->NumInOperands(); ++i) {
    if (i >= kEntryPointInterfaceInIdx &&
        entry_point->GetSingleWordInOperand(i) == var->result_id()) {
      continue;
    }
    new_operands.push_back(entry_point->GetInOperand(i));
  }
  if (new_operands.size() < entry_point->NumInOperands()) {
    entry_point->SetInOperands(std::move(new_operands));
    context()->get_def_use_mgr()->AnalyzeInstUse(entry_point);
  }

  context()->KillInst(var);
}

LowerClipCullDistancePass::PassStatus LowerClipCullDistancePass::FindBuiltinVariables(
    Instruction* entry_point, BuiltinVariableInfo* info) {
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();

  for (uint32_t i = kEntryPointInterfaceInIdx; i < entry_point->NumInOperands();
       ++i) {
    Instruction* var =
        get_def_use_mgr()->GetDef(entry_point->GetSingleWordInOperand(i));
    uint32_t var_id = var->result_id();

    // For Clip/CullDistance decorations
    deco_mgr->ForEachDecoration(
        var_id, uint32_t(spv::Decoration::BuiltIn),
        [&](const Instruction& decoration) {
          if (decoration.opcode() != spv::Op::OpDecorate) return;
          spv::BuiltIn builtin = static_cast<spv::BuiltIn>(
              decoration.GetSingleWordInOperand(kBuiltInDecorationInIdx));
          if (builtin == spv::BuiltIn::ClipDistance) {
            LOG("Found OpDecorate ... ClipDistance");
            info->clip_dist_var = var;
            info->clip_dist_decoration = (Instruction*) &decoration;
          } else if (builtin == spv::BuiltIn::CullDistance) {
            LOG("Found OpDecorate ... CullDistance");
            info->cull_dist_var = var;
            info->cull_dist_decoration = (Instruction*) &decoration;
          }
        });

    // For Position decorations
    uint32_t pointee_type_id = GetPointeeTypeId(var);
    Instruction* pointee_type_inst = get_def_use_mgr()->GetDef(pointee_type_id);
    if (pointee_type_inst &&
        pointee_type_inst->opcode() == spv::Op::OpTypeStruct) {
      deco_mgr->ForEachDecoration(
          pointee_type_inst->result_id(), uint32_t(spv::Decoration::BuiltIn),
          [&](const Instruction& decoration) {
            // TODO: find OpMemberDecorate of clip/cull distance too
            if (decoration.opcode() == spv::Op::OpMemberDecorate) {
              spv::BuiltIn builtin = static_cast<spv::BuiltIn>(
                  decoration.GetSingleWordInOperand(kMemberBuiltInDecorationInIdx));
              if (builtin == spv::BuiltIn::Position) {
                info->position_var = var;
                info->position_member_index =
                    decoration.GetSingleWordInOperand(kMemberIndexInIdx);
                LOG("Found OpMemberDecorate Position");
              } else if (builtin == spv::BuiltIn::ClipDistance) {
                LOG("Found OpMemberDecorate ClipDistance");
                info->clip_dist_mem_var = var;
                info->clip_dist_mem_idx = decoration.GetSingleWordInOperand(kMemberIndexInIdx);
                info->clip_dist_mem_decoration = (Instruction*) &decoration;
              } else if (builtin == spv::BuiltIn::CullDistance) {
                LOG("Found OpMemberDecorate CullDistance");
                info->cull_dist_mem_var = var;
                info->cull_dist_mem_idx = decoration.GetSingleWordInOperand(kMemberIndexInIdx);
                info->cull_dist_mem_decoration = (Instruction*) &decoration;
              }
            }
          });
    }
  }

  if (info->clip_dist_var == nullptr && info->cull_dist_var == nullptr 
    && info->clip_dist_mem_var == nullptr && info->cull_dist_mem_var == nullptr) {
    return NO_CHANGES;  // Nothing to do.
  }

  if (info->position_var == nullptr) {
    LOG("Shader uses ClipDistance or CullDistance but does not declare a Position built-in.");
    return CLEANUP_ONLY;
  }

  return EMULATED;
}

void LowerClipCullDistancePass::FindRelevantStores(
    Instruction* builtin_var, std::vector<Instruction*>* stores) {
  std::unordered_set<Instruction*> visited;
  std::queue<Instruction*> worklist;
  worklist.push(builtin_var);
  visited.insert(builtin_var);

  while (!worklist.empty()) {
    Instruction* current = worklist.front();
    worklist.pop();

    get_def_use_mgr()->ForEachUser(current, [&](Instruction* user) {
      if (visited.count(user)) return;
      visited.insert(user);
      switch (user->opcode()) {
        case spv::Op::OpStore:
          stores->push_back(user);
          break;
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpCopyObject:
          worklist.push(user);
          break;
        default:
          break;
      }
    });
  }
}

void LowerClipCullDistancePass::FindAllUses(
    Instruction* builtin_var, std::vector<Instruction*>* stores) {
  std::unordered_set<Instruction*> visited;
  std::queue<Instruction*> worklist;
  worklist.push(builtin_var);
  visited.insert(builtin_var);

  while (!worklist.empty()) {
    Instruction* current = worklist.front();
    worklist.pop();

    get_def_use_mgr()->ForEachUser(current, [&](Instruction* user) {
      if (visited.count(user)) return;
      visited.insert(user);
      switch (user->opcode()) {
        case spv::Op::OpStore:
          stores->push_back(user);
          break;
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpCopyObject:
        case spv::Op::OpCompositeConstruct:
        case spv::Op::OpLoad:
          stores->push_back(user);
          worklist.push(user);
          break;
        default:
          break;
      }
    });
  }
}

void LowerClipCullDistancePass::InjectClippingCode(
    Instruction* store_inst, const BuiltinVariableInfo& builtins,
    spv::ExecutionModel exec_model) {
  Instruction* value_inst =
      get_def_use_mgr()->GetDef(store_inst->GetSingleWordInOperand(1));
  analysis::Type* value_type =
      context()->get_type_mgr()->GetType(value_inst->type_id());

  if (value_type->AsFloat()) {
    InjectScalarCheck(store_inst, builtins, exec_model);
  } else if (value_type->AsVector()) {
    InjectVectorCheck(store_inst, builtins, exec_model);
  } else {
    LOGE("Unsupported aggregate type for ClipDistance/CullDistance. Run scalar-replacement first.");
  }
}

void LowerClipCullDistancePass::InjectScalarCheck(
    Instruction* store_inst, const BuiltinVariableInfo& builtins,
    spv::ExecutionModel exec_model) {
  BasicBlock* current_block = context()->get_instr_block(store_inst);
  Function* function = current_block->GetParent();

  auto clip_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, TakeNextId(),
                              std::initializer_list<Operand>{}));
  uint32_t clip_block_id = clip_block->id();
  context()->AnalyzeDefUse(clip_block->GetLabelInst());
  context()->set_instr_block(clip_block->GetLabelInst(), clip_block.get());

  InstructionBuilder clip_block_builder(context(), clip_block.get(),
                                        IRContext::kAnalysisDefUse);
  Instruction* pos_ptr =
      FindPositionPointerForStore(store_inst, builtins, exec_model);
  if (!pos_ptr) return;

  std::vector<uint32_t> indices = {GetConstUintId(builtins.position_member_index),
                                   GetConstUintId(3)}; // the w coord
  uint32_t pos_ptr_w_id =
      clip_block_builder
          .AddAccessChain(GetPointerToFloatTypeId(), pos_ptr->result_id(),
                          indices)
          ->result_id();
  clip_block_builder.AddStore(pos_ptr_w_id, GetConstFloatId(-2.0f));

  auto split_iter = BasicBlock::iterator(store_inst);
  ++split_iter;
  BasicBlock* merge_block =
      current_block->SplitBasicBlock(context(), TakeNextId(), split_iter);
  uint32_t merge_block_id = merge_block->id();
  clip_block_builder.AddBranch(merge_block_id);

  InstructionBuilder builder(
      context(), current_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t stored_val_id = store_inst->GetSingleWordInOperand(1);
  uint32_t is_neg_id =
      builder
          .AddBinaryOp(GetBoolTypeId(), spv::Op::OpFOrdLessThan, stored_val_id,
                       GetConstFloatId(0.0f))
          ->result_id();

  builder.AddSelectionMerge(
      merge_block_id,
      static_cast<uint32_t>(spv::SelectionControlMask::MaskNone));
  builder.AddConditionalBranch(is_neg_id, clip_block_id, merge_block_id);

  auto current_block_iter = function->FindBlock(current_block->id());
  ++current_block_iter;
  current_block_iter.InsertBefore(std::move(clip_block));
}

void LowerClipCullDistancePass::InjectVectorCheck(
    Instruction* store_inst, const BuiltinVariableInfo& builtins,
    spv::ExecutionModel exec_model) {
  BasicBlock* current_block = context()->get_instr_block(store_inst);
  Function* function = current_block->GetParent();

  auto clip_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, TakeNextId(),
                              std::initializer_list<Operand>{}));
  uint32_t clip_block_id = clip_block->id();
  context()->AnalyzeDefUse(clip_block->GetLabelInst());
  context()->set_instr_block(clip_block->GetLabelInst(), clip_block.get());

  InstructionBuilder clip_block_builder(context(), clip_block.get(),
                                        IRContext::kAnalysisDefUse);
  Instruction* pos_ptr =
      FindPositionPointerForStore(store_inst, builtins, exec_model);
  if (!pos_ptr) return;

  std::vector<uint32_t> indices = {GetConstUintId(builtins.position_member_index),
                                   GetConstUintId(3)}; // the w coord
  uint32_t pos_ptr_w_id =
      clip_block_builder
          .AddAccessChain(GetPointerToFloatTypeId(), pos_ptr->result_id(),
                          indices)
          ->result_id();
  clip_block_builder.AddStore(pos_ptr_w_id, GetConstFloatId(-2.0f));

  auto split_iter = BasicBlock::iterator(store_inst);
  ++split_iter;
  BasicBlock* merge_block =
      current_block->SplitBasicBlock(context(), TakeNextId(), split_iter);
  uint32_t merge_block_id = merge_block->id();
  clip_block_builder.AddBranch(merge_block_id);

  InstructionBuilder builder(
      context(), current_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  uint32_t stored_val_id = store_inst->GetSingleWordInOperand(1);
  Instruction* stored_val = get_def_use_mgr()->GetDef(stored_val_id);
  analysis::Vector* vec_type =
      context()->get_type_mgr()->GetType(stored_val->type_id())->AsVector();
  uint32_t num_components = vec_type->element_count();

  std::vector<uint32_t> zero_components(num_components, GetConstFloatId(0.0f));
  uint32_t vec_zero_id =
      builder.AddCompositeConstruct(stored_val->type_id(), zero_components)
          ->result_id();

  analysis::Vector bool_vec_type(
      context()->get_type_mgr()->GetType(GetBoolTypeId()), num_components);
  uint32_t bool_vec_type_id =
      context()->get_type_mgr()->GetTypeInstruction(&bool_vec_type);
  uint32_t vec_is_neg_id =
      builder
          .AddBinaryOp(bool_vec_type_id, spv::Op::OpFOrdLessThan, stored_val_id,
                       vec_zero_id)
          ->result_id();

  uint32_t any_is_neg_id =
      builder.AddUnaryOp(GetBoolTypeId(), spv::Op::OpAny, vec_is_neg_id)
          ->result_id();

  builder.AddSelectionMerge(
      merge_block_id,
      static_cast<uint32_t>(spv::SelectionControlMask::MaskNone));
  builder.AddConditionalBranch(any_is_neg_id, clip_block_id, merge_block_id);

  auto current_block_iter = function->FindBlock(current_block->id());
  ++current_block_iter;
  current_block_iter.InsertBefore(std::move(clip_block));
}

Instruction* LowerClipCullDistancePass::FindPositionPointerForStore(
    Instruction* store_inst, const BuiltinVariableInfo& builtins,
    spv::ExecutionModel exec_model) {
  if (exec_model != spv::ExecutionModel::Geometry) {
    return builtins.position_var;
  }

  if (!store_inst) {
    return nullptr;
  }

  // BasicBlock* current_block = context()->get_instr_block(store_inst);
  // auto iter = BasicBlock::iterator(store_inst);
  // ++iter;

  // for (; iter != current_block->end(); ++iter) {
  //   if (iter->opcode() == spv::Op::OpEmitVertex) {
  //     auto search_iter = iter;
  //     while (true) {
  //       if (search_iter->opcode() == spv::Op::OpStore) {
  //         uint32_t ptr_id = search_iter->GetSingleWordInOperand(0);
  //         Instruction* base_ptr = GetPtr(ptr_id, nullptr);
  //         if (base_ptr == builtins.position_var) {
  //           return get_def_use_mgr()->GetDef(ptr_id);
  //         }
  //       }
  //       if (search_iter == current_block->begin()) break;
  //       --search_iter;
  //     }
  //     break;
  //   }
  // }

  LOG("ClipDistance/CullDistance emulation not supported in GeometryShader");
  return nullptr;
}

bool LowerClipCullDistancePass::Cleanup() {
  bool changed = false;

  // std::vector<Instruction*> dead_decorates;

  // for (const auto& var : context()->module()->types_values()) {
  //   if (var.opcode() != spv::Op::OpVariable) continue;
  //   context()->get_decoration_mgr()->ForEachDecoration(
  //       var.result_id(), uint32_t(spv::Decoration::BuiltIn),
  //       [&](const Instruction& decoration) {
  //         if (decoration.opcode() != spv::Op::OpMemberDecorate) return;
  //         spv::BuiltIn builtin = static_cast<spv::BuiltIn>(
  //             decoration.GetSingleWordInOperand(3));
  //         if (builtin == spv::BuiltIn::ClipDistance) {
  //           dead_decorates.push_back(&decoration);
  //           context()->KillInst(var);
  //           context()->get_decoration_mgr()->RemoveDecoration(&decoration);
  //         }
  //         if (builtin == spv::BuiltIn::CullDistance) {
  //           dead_decorates.push_back(&decoration);
  //         }
  //       });
  // }

  if (context()->RemoveCapability(spv::Capability::ClipDistance)) {
    LOG("Killing OpCapability ClipDistance");
    changed = true;
  }
  if (context()->RemoveCapability(spv::Capability::CullDistance)) {
    LOG("Killing OpCapability CullDistance");
    changed = true;
  }

  return changed;
}

uint32_t LowerClipCullDistancePass::GetConstFloatId(float value) {
  auto it = float_const_ids_.find(value);
  if (it != float_const_ids_.end()) {
    return it->second;
  }

  analysis::Type* float_type = context()->get_type_mgr()->GetFloatType();
  const analysis::Constant* float_const =
      context()->get_constant_mgr()->GetConstant(
          float_type, {utils::FloatProxy<float>(value).GetWords()[0]});
  uint32_t id = context()->get_constant_mgr()
                    ->GetDefiningInstruction(float_const)
                    ->result_id();
  float_const_ids_[value] = id;
  return id;
}

uint32_t LowerClipCullDistancePass::GetConstUintId(uint32_t value) {
  auto it = uint_const_ids_.find(value);
  if (it != uint_const_ids_.end()) {
    return it->second;
  }

  analysis::Type* uint_type = context()->get_type_mgr()->GetUIntType();
  const analysis::Constant* uint_const =
      context()->get_constant_mgr()->GetConstant(uint_type, {value});
  uint32_t id =
      context()->get_constant_mgr()->GetDefiningInstruction(uint_const)->result_id();
  uint_const_ids_[value] = id;
  return id;
}

uint32_t LowerClipCullDistancePass::GetPointerToFloatTypeId() {
  if (ptr_to_float_type_id_ != 0) {
    return ptr_to_float_type_id_;
  }
  analysis::Type* float_type = context()->get_type_mgr()->GetFloatType();
  uint32_t float_type_id =
      context()->get_type_mgr()->GetTypeInstruction(float_type);
  ptr_to_float_type_id_ = context()->get_type_mgr()->FindPointerToType(
      float_type_id, spv::StorageClass::Output);
  return ptr_to_float_type_id_;
}

uint32_t LowerClipCullDistancePass::GetBoolTypeId() {
  if (bool_type_id_ != 0) {
    return bool_type_id_;
  }
  analysis::Bool bool_type;
  bool_type_id_ = context()->get_type_mgr()->GetTypeInstruction(&bool_type);
  return bool_type_id_;
}

}  // namespace opt
}  // namespace spvtools
