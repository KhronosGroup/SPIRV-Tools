// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "inline_pass.h"

// Indices of operands in SPIR-V instructions

static const int kSpvEntryPointFunctionId = 1;
static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFuncitonCallArgumentId = 3;
static const int kSpvFunctionParameterResultId = 1;
static const int kSpvReturnValueId = 0;
static const int kSpvTypePointerStorageClass = 1;
static const int kSpvTypePointerTypeId = 2;

namespace spvtools {
namespace opt {

uint32_t InlinePass::FindPointerToType(uint32_t type_id, uint32_t storage_id) {
  ir::Module::inst_iterator type_itr = module_->types_values_begin();
  for (; type_itr != module_->types_values_end(); ++type_itr) {
    const ir::Instruction* type_inst = &*type_itr;
    if (type_inst->opcode() == SpvOpTypePointer &&
        type_inst->GetOperand(kSpvTypePointerTypeId).words[0] == type_id &&
        type_inst->GetOperand(kSpvTypePointerStorageClass).words[0] ==
            storage_id)
      break;
  }
  return (type_itr != module_->types_values_end()) ? type_itr->result_id() : 0;
}

uint32_t InlinePass::AddPointerToType(uint32_t type_id) {
  uint32_t resultId = TakeNextId();
  std::vector<ir::Operand> in_operands;
  in_operands.emplace_back(
      spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
      std::initializer_list<uint32_t>{uint32_t(SpvStorageClassFunction)});
  in_operands.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                           std::initializer_list<uint32_t>{uint32_t(type_id)});
  std::unique_ptr<ir::Instruction> type_inst(
      new ir::Instruction(SpvOpTypePointer, 0, resultId, in_operands));
  module_->AddType(std::move(type_inst));
  return resultId;
}

void InlinePass::AddBranch(uint32_t label_id,
                           std::unique_ptr<ir::BasicBlock>* bp) {
  std::vector<ir::Operand> branch_in_operands;
  branch_in_operands.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                  std::initializer_list<uint32_t>{label_id}));
  std::unique_ptr<ir::Instruction> newBranch(
      new ir::Instruction(SpvOpBranch, 0, 0, branch_in_operands));
  (*bp)->AddInstruction(std::move(newBranch));
}

void InlinePass::AddStore(uint32_t ptr_id, uint32_t val_id,
                          std::unique_ptr<ir::BasicBlock>* bp) {
  std::vector<ir::Operand> store_in_operands;
  store_in_operands.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                  std::initializer_list<uint32_t>{ptr_id}));
  store_in_operands.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                  std::initializer_list<uint32_t>{val_id}));
  std::unique_ptr<ir::Instruction> newStore(
      new ir::Instruction(SpvOpStore, 0, 0, store_in_operands));
  (*bp)->AddInstruction(std::move(newStore));
}

void InlinePass::AddLoad(uint32_t type_id, uint32_t resultId, uint32_t ptr_id,
                         std::unique_ptr<ir::BasicBlock>* bp) {
  std::vector<ir::Operand> load_in_operands;
  load_in_operands.push_back(
      ir::Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                  std::initializer_list<uint32_t>{ptr_id}));
  std::unique_ptr<ir::Instruction> newLoad(
      new ir::Instruction(SpvOpLoad, type_id, resultId, load_in_operands));
  (*bp)->AddInstruction(std::move(newLoad));
}

// Generate callee code into newBlocks to be inlined for the function call at
// call_ii. Also add new function variables into caller func

void InlinePass::GenInlineCode(
    std::vector<std::unique_ptr<ir::BasicBlock>>* newBlocks,
    std::vector<std::unique_ptr<ir::Instruction>>* newVars,
    ir::UptrVectorIterator<ir::Instruction> call_ii,
    ir::UptrVectorIterator<ir::BasicBlock> call_bi) {
  // Map from callee id to caller id
  std::unordered_map<uint32_t, uint32_t> callee2caller;
  // Pre-call OpSampledImage Insts
  std::unordered_map<uint32_t, ir::Instruction*> preCallSI;
  // Post-call OpSampledImage Ids
  std::unordered_map<uint32_t, uint32_t> postCallSI;

  const uint32_t calleeId =
      call_ii->GetOperand(kSpvFunctionCallFunctionId).words[0];
  ir::Function* calleeFn = id2function_[calleeId];

  // Map parameters to actual arguments
  int i = 0;
  calleeFn->ForEachParam(
      [&call_ii, &i, &callee2caller](const ir::Instruction* cpi) {
        const uint32_t pid =
            cpi->GetOperand(kSpvFunctionParameterResultId).words[0];
        callee2caller[pid] =
            call_ii->GetOperand(kSpvFuncitonCallArgumentId + i).words[0];
        i++;
      });

  // Define caller local variables for all callee variables and create map to
  // them
  auto cbi = calleeFn->begin();
  auto cvi = cbi->begin();
  while (cvi->opcode() == SpvOp::SpvOpVariable) {
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(*cvi));
    uint32_t newId = TakeNextId();
    var_inst->SetResultId(newId);
    callee2caller[cvi->result_id()] = newId;
    newVars->push_back(std::move(var_inst));
    cvi++;
  }

  // Create return var if needed
  uint32_t returnVarId = 0;
  const uint32_t calleeTypeId = calleeFn->type_id();
  const ir::Instruction* calleeType =
      def_use_mgr_->id_to_defs().find(calleeTypeId)->second;
  if (calleeType->opcode() != SpvOpTypeVoid) {
    // find or create ptr to callee return type
    uint32_t returnVarTypeId =
        FindPointerToType(calleeTypeId, SpvStorageClassFunction);
    if (returnVarTypeId == 0) returnVarTypeId = AddPointerToType(calleeTypeId);
    // Add return var to new function scope variables
    returnVarId = TakeNextId();
    std::vector<ir::Operand> in_operands;
    in_operands.emplace_back(
        spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
        std::initializer_list<uint32_t>{uint32_t(SpvStorageClassFunction)});
    std::unique_ptr<ir::Instruction> var_inst(new ir::Instruction(
        SpvOpVariable, returnVarTypeId, returnVarId, in_operands));
    newVars->push_back(std::move(var_inst));
  }

  // Clone and map callee code
  bool prevInstWasReturn = false;
  uint32_t returnLabelId = 0;
  bool multiBlocks = false;
  std::unique_ptr<ir::BasicBlock> bp;
  calleeFn->ForEachInst(
      [&newBlocks, &callee2caller, &call_bi, &call_ii, &bp, &prevInstWasReturn,
       &returnLabelId, &returnVarId, &calleeTypeId, &multiBlocks, &postCallSI,
       &preCallSI, this](const ir::Instruction* cpi) {
        switch (cpi->opcode()) {
          case SpvOpFunction:
          case SpvOpFunctionParameter:
          case SpvOpVariable:
            // already processed
            break;
          case SpvOpLabel: {
            // if previous instruction was early return, insert branch
            // instruction
            // to return block
            if (prevInstWasReturn) {
              if (returnLabelId == 0) returnLabelId = this->TakeNextId();
              AddBranch(returnLabelId, &bp);
              prevInstWasReturn = false;
            }
            // finish current block (if it exists) and get label for next block
            uint32_t labelId;
            bool firstBlock = false;
            if (bp != nullptr) {
              newBlocks->push_back(std::move(bp));
              // if result id is already mapped, use it, otherwise get a new
              // one.
              const uint32_t rid = cpi->result_id();
              const auto mapItr = callee2caller.find(rid);
              labelId = (mapItr != callee2caller.end()) ? mapItr->second
                                                        : this->TakeNextId();
            } else {
              // first block needs to use label of original block
              // but map callee label in case of phi reference
              labelId = call_bi->label_id();
              callee2caller[cpi->result_id()] = labelId;
              firstBlock = true;
            }
            // create first/next block
            const std::vector<ir::Operand> label_in_operands;
            std::unique_ptr<ir::Instruction> newLabel(
                new ir::Instruction(SpvOpLabel, 0, labelId, label_in_operands));
            bp.reset(new ir::BasicBlock(std::move(newLabel)));
            if (firstBlock) {
              // Copy contents of original caller block up to call instruction
              for (auto cii = call_bi->begin(); cii != call_ii; cii++) {
                std::unique_ptr<ir::Instruction> spv_inst(
                    new ir::Instruction(*cii));
                // remember OpSampledImages for possible regeneration
                if (spv_inst->opcode() == SpvOpSampledImage) {
                  auto* samp_inst_ptr = spv_inst.get();
                  preCallSI[spv_inst->result_id()] = samp_inst_ptr;
                }
                bp->AddInstruction(std::move(spv_inst));
              }
            } else
              multiBlocks = true;
          } break;
          case SpvOpReturnValue: {
            // store return value to return variable
            assert(returnVarId != 0);
            uint32_t valId = cpi->GetInOperand(kSpvReturnValueId).words[0];
            const auto mapItr = callee2caller.find(valId);
            if (mapItr != callee2caller.end()) {
              valId = mapItr->second;
            }
            AddStore(returnVarId, valId, &bp);

            // Remember we saw a return; if followed by a label, will need to
            // insert
            // branch
            prevInstWasReturn = true;
          } break;
          case SpvOpReturn: {
            // Remember we saw a return; if followed by a label, will need to
            // insert
            // branch
            prevInstWasReturn = true;
          } break;
          case SpvOpFunctionEnd: {
            // if there was an early return, create return label/block
            // if previous instruction was return, insert branch instruction
            // to return block
            if (returnLabelId != 0) {
              if (prevInstWasReturn) AddBranch(returnLabelId, &bp);
              newBlocks->push_back(std::move(bp));
              const std::vector<ir::Operand> label_in_operands;
              std::unique_ptr<ir::Instruction> newLabel(new ir::Instruction(
                  SpvOpLabel, 0, returnLabelId, label_in_operands));
              bp.reset(new ir::BasicBlock(std::move(newLabel)));
              multiBlocks = true;
            }
            // load return value into result id of call, if it exists
            if (returnVarId != 0) {
              const uint32_t resId = call_ii->result_id();
              assert(resId != 0);
              AddLoad(calleeTypeId, resId, returnVarId, &bp);
            }
            // copy remaining instructions from caller block.
            auto cii = call_ii;
            cii++;
            for (; cii != call_bi->end(); cii++) {
              std::unique_ptr<ir::Instruction> spv_inst(
                  new ir::Instruction(*cii));
              // if multiple blocks generated, regenerate any OpSampledImage
              // instruction that has not been seen in this last block.
              if (multiBlocks) {
                spv_inst->ForEachInId(
                    [&postCallSI, &preCallSI, &cpi, &bp, this](uint32_t* iid) {
                      const auto mapItr = postCallSI.find(*iid);
                      if (mapItr == postCallSI.end()) {
                        const auto mapItr2 = preCallSI.find(*iid);
                        if (mapItr2 != preCallSI.end()) {
                          // clone pre-call OpSampledImage, map result id
                          const ir::Instruction* inInst = mapItr2->second;
                          std::unique_ptr<ir::Instruction> samp_inst(
                              new ir::Instruction(*inInst));
                          const uint32_t rid = samp_inst->result_id();
                          const uint32_t nid = this->TakeNextId();
                          samp_inst->SetResultId(nid);
                          postCallSI[rid] = nid;
                          *iid = nid;
                          bp->AddInstruction(std::move(samp_inst));
                        }
                      } else
                        // reset OpSampledImage operand
                        *iid = mapItr->second;
                    });
                // remember OpSampledImage in this block
                if (spv_inst->opcode() == SpvOpSampledImage) {
                  const uint32_t rid = spv_inst->result_id();
                  postCallSI[rid] = rid;
                }
              }
              bp->AddInstruction(std::move(spv_inst));
            }
            // finalize
            newBlocks->push_back(std::move(bp));
          } break;
          default: {
            // copy callee instruction and remap all input Ids
            std::unique_ptr<ir::Instruction> spv_inst(
                new ir::Instruction(*cpi));
            spv_inst->ForEachInId([&callee2caller, &cpi, this](uint32_t* iid) {
              const auto mapItr = callee2caller.find(*iid);
              if (mapItr != callee2caller.end()) {
                *iid = mapItr->second;
              } else if (cpi->IsControlFlow()) {
                const ir::Instruction* inst =
                    def_use_mgr_->id_to_defs().find(*iid)->second;
                if (inst->opcode() == SpvOpLabel) {
                  // forward label reference. allocate a new label id, map it,
                  // use
                  // it and check for it at each label.
                  const uint32_t nid = this->TakeNextId();
                  callee2caller[*iid] = nid;
                  *iid = nid;
                }
              }
            });
            // map and reset result id
            const uint32_t rid = spv_inst->result_id();
            if (rid != 0) {
              const uint32_t nid = this->TakeNextId();
              callee2caller[rid] = nid;
              spv_inst->SetResultId(nid);
            }
            bp->AddInstruction(std::move(spv_inst));
          } break;
        }
      });
}

bool InlinePass::Inline(ir::Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end(); bi++) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (ii->opcode() == SpvOp::SpvOpFunctionCall) {
        // Inline call
        std::vector<std::unique_ptr<ir::BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<ir::Instruction>> newVars;
        GenInlineCode(&newBlocks, &newVars, ii, bi);
        // update block map given replacement blocks
        for (auto& blk : newBlocks) {
          id2block_[blk->label_id()] = &*blk;
        }
        // update phi functions in succesor blocks if call block
        // is replaced with more than one block
        if (newBlocks.size() > 1) {
          const auto firstBlk = newBlocks.begin();
          const auto lastBlk = newBlocks.end() - 1;
          const uint32_t firstId = (*firstBlk)->label_id();
          const uint32_t lastId = (*lastBlk)->label_id();
          (*lastBlk)
              ->ForEachSuccessorLabel([&firstId, &lastId, this](uint32_t succ) {
                ir::BasicBlock* sbp = this->id2block_[succ];
                sbp->ForEachPhiInst([&firstId, &lastId](ir::Instruction* phi) {
                  phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
                    if (*id == firstId) *id = lastId;
                  });
                });
              });
        }
        // replace old calling block with new block(s)
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // insert new function variables
        if (newVars.size() > 0) {
          auto vbi = func->begin();
          auto vii = vbi->begin();
          vii.InsertBefore(&newVars);
        }
        // restart inlining at beginning of calling block
        ii = bi->begin();
        modified = true;
      } else {
        ii++;
      }
    }
  }
  return modified;
}

void InlinePass::Initialize(ir::Module* module) {
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

  // Initialize next unused Id
  next_id_ = 0;
  for (const auto& id_def : def_use_mgr_->id_to_defs()) {
    next_id_ = std::max(next_id_, id_def.first);
  }
  next_id_++;

  module_ = module;

  // initialize function and block maps
  id2function_.clear();
  id2block_.clear();
  for (auto& fn : *module_) {
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.label_id()] = &blk;
    }
  }
};

Pass::Status InlinePass::ProcessImpl() {
  // do exhaustive inlining on each entry point function in module
  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetOperand(kSpvEntryPointFunctionId).words[0]];
    modified = modified || Inline(fn);
  }

  FinalizeNextId(module_);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InlinePass::InlinePass()
    : module_(nullptr), def_use_mgr_(nullptr), next_id_(0) {}

Pass::Status InlinePass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
