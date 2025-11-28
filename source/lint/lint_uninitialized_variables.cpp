#include "source/lint/uninitialized_analysis.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "source/lint/variable_state.h"
#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "source/opt/type_manager.h"
#include "source/opt/types.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"

namespace spvtools {
namespace lint {
namespace uninitialized_variables {

State StateForAccessChain(opt::IRContext& context,
                          const opt::Instruction& op_type,
                          const std::vector<AccessChainField>& chain,
                          size_t chain_offset, size_t chain_end,
                          std::unordered_set<TypeId>& resolving_types) {
  // Don't go into infinite loop
  assert(chain_offset <= chain.size());
  assert(chain_end <= chain.size());
  if (chain_end == chain_offset) {
    return State::NewLeaf(Initialized::Yes);
  }
  opt::analysis::TypeManager& tym = *context.get_type_mgr();
  opt::analysis::DefUseManager& defm = *context.get_def_use_mgr();

  const auto recurse = [&](const opt::Instruction& inst) {
    assert(resolving_types.find(inst.result_id()) == resolving_types.cend());
    resolving_types.insert(inst.result_id());
    State result = StateForAccessChain(context, inst, chain, chain_offset + 1,
                                       chain_end, resolving_types);
    resolving_types.erase(inst.result_id());
    return result;
  };

  const AccessChainField field = chain.at(chain_offset);

  if (op_type.opcode() == spv::Op::OpTypeStruct) {
    const uint32_t num_fields = op_type.NumInOperands();
    assert(field.is_constant && field.value < num_fields);
    std::vector<State> members(num_fields, State::NewLeaf(Initialized::No));
    const opt::Instruction* member_def =
        defm.GetDef(op_type.GetSingleWordInOperand(field.value));
    assert(member_def != nullptr);
    members.at(field.value) = recurse(*member_def);
    return State::NewComposite(members);
  } else if (op_type.opcode() == spv::Op::OpTypeArray) {
    const uint32_t element_id = op_type.GetSingleWordInOperand(0);
    const opt::Instruction* element_def = defm.GetDef(element_id);
    assert(element_def != nullptr);
    const opt::analysis::Type* element_type = tym.GetType(element_id);
    assert(element_type != nullptr);

    const uint32_t length_id = op_type.GetSingleWordInOperand(1);
    const opt::Instruction* length_def = defm.GetDef(length_id);
    assert(length_def != nullptr);
    if (length_def->opcode() == spv::Op::OpConstant) {
      const uint32_t length = length_def->GetSingleWordInOperand(0);
      State element = recurse(*element_def);
      if (field.is_constant) {
        std::vector<State> members(length, State::NewLeaf(Initialized::No));
        assert(field.value < length);
        members.at(field.value) = element;
        return State::NewComposite(members);
      } else {
        std::vector<State> members(
            length,
            element.TryUnion(State::NewLeaf(Initialized::Unknown)).value());
        return State::NewComposite(members);
      }
    } else if (length_def->opcode() == spv::Op::OpSpecConstant) {
      // Set all array elements to the same state (if the access chain
      // continues), clamped to Unknown
      return recurse(*element_def)
          .TryIntersect(State::NewLeaf(Initialized::Unknown))
          .value();
    } else {
      assert(false && "unreachable");
      return State::NewLeaf(Initialized::Unknown);
    }
  } else if (op_type.opcode() == spv::Op::OpTypeVector) {
    const uint32_t element_id = op_type.GetSingleWordInOperand(0);
    const opt::Instruction* element_def = defm.GetDef(element_id);
    assert(element_def != nullptr);

    const uint32_t length = op_type.GetSingleWordInOperand(1);
    State element = recurse(*element_def);
    if (field.is_constant) {
      std::vector<State> members(length, State::NewLeaf(Initialized::No));
      assert(field.value < length);
      members.at(field.value) = element;
      return State::NewComposite(members);
    } else {
      std::vector<State> members(
          length,
          element.TryUnion(State::NewLeaf(Initialized::Unknown)).value());
      return State::NewComposite(members);
    }
  } else {
    assert(op_type.opcode() == spv::Op::OpTypeMatrix);
    const uint32_t vec_id = op_type.GetSingleWordInOperand(0);
    const opt::Instruction* vec_def = defm.GetDef(vec_id);
    assert(vec_def != nullptr);
    assert(vec_def->opcode() == spv::Op::OpTypeVector);

    const uint32_t num_cols = op_type.GetSingleWordInOperand(1);
    State column = recurse(*vec_def);
    if (field.is_constant) {
      std::vector<State> columns(num_cols, State::NewLeaf(Initialized::No));
      assert(field.value < num_cols);
      columns.at(field.value) = column;
      return State::NewComposite(columns);
    } else {
      std::vector<State> columns(
          num_cols,
          column.TryUnion(State::NewLeaf(Initialized::Unknown)).value());
      return State::NewComposite(columns);
    }
  }
}

struct VarAndState {
  VarId var_id;
  State state;
};

std::optional<VarAndState> StateForLoadStore(
    opt::IRContext& context, const opt::Instruction& op_loadstore) {
  assert(op_loadstore.opcode() == spv::Op::OpLoad ||
         op_loadstore.opcode() == spv::Op::OpStore);
  const opt::analysis::DefUseManager& defm = *context.get_def_use_mgr();

  const VarId target = op_loadstore.GetSingleWordInOperand(0);
  const opt::Instruction* target_def = defm.GetDef(target);
  assert(target_def != nullptr);
  if (target_def->opcode() == spv::Op::OpVariable) {
    const spv::StorageClass storage_class =
        spv::StorageClass(target_def->GetSingleWordInOperand(0));
    if (storage_class != spv::StorageClass::Function &&
        storage_class != spv::StorageClass::Private) {
      return std::nullopt;
    }
    return {{.var_id = target_def->result_id(),
             .state = State::NewLeaf(Initialized::Yes)}};
  } else if (target_def->opcode() == spv::Op::OpFunctionParameter) {
    // Always Initialized, do nothing
    return std::nullopt;
  }

  assert(target_def->opcode() == spv::Op::OpAccessChain ||
         target_def->opcode() == spv::Op::OpInBoundsAccessChain);
  const opt::Instruction* base_def =
      defm.GetDef(target_def->GetSingleWordInOperand(0));
  if (base_def->opcode() != spv::Op::OpVariable) {
    return std::nullopt;
  }
  const spv::StorageClass storage_class =
      spv::StorageClass(base_def->GetSingleWordInOperand(0));
  if (storage_class != spv::StorageClass::Function &&
      storage_class != spv::StorageClass::Private) {
    return std::nullopt;
  }
  const opt::Instruction* ptr_def = defm.GetDef(base_def->type_id());
  assert(ptr_def != nullptr && ptr_def->opcode() == spv::Op::OpTypePointer);
  const opt::Instruction* type_def =
      defm.GetDef(ptr_def->GetSingleWordInOperand(1));
  assert(type_def != nullptr);

  std::vector<AccessChainField> chain;
  for (uint32_t i = 1; i < target_def->NumInOperands(); ++i) {
    const opt::Instruction* field_def = context.get_def_use_mgr()->GetDef(
        target_def->GetSingleWordInOperand(i));
    assert(field_def != nullptr);
    if (field_def->opcode() == spv::Op::OpConstant) {
      const uint32_t value = field_def->GetSingleWordInOperand(0);
      chain.push_back({.value = value, .is_constant = true});
    } else {
      chain.push_back({.value = field_def->result_id(), .is_constant = false});
    }
  }
  std::unordered_set<TypeId> recursion_guard;
  std::optional<State> state = StateForAccessChain(
      context, *type_def, chain, 0, chain.size(), recursion_guard);
  if (!state) {
    return std::nullopt;
  }
  return {{.var_id = base_def->result_id(), .state = *state}};
}

std::vector<AccessChainField> AccessChainFields(
    opt::IRContext& context, const opt::Instruction& op_chain) {
  assert(op_chain.opcode() == spv::Op::OpAccessChain ||
         op_chain.opcode() == spv::Op::OpInBoundsAccessChain);
  std::vector<AccessChainField> fields;
  for (uint32_t i = 1; i < op_chain.NumInOperands(); ++i) {
    const opt::Instruction* field_def =
        context.get_def_use_mgr()->GetDef(op_chain.GetSingleWordInOperand(i));
    assert(field_def != nullptr);
    if (field_def->opcode() == spv::Op::OpConstant) {
      const uint32_t value = field_def->GetSingleWordInOperand(0);
      fields.push_back({.value = value, .is_constant = true});
    } else {
      fields.push_back({.value = field_def->result_id(), .is_constant = false});
    }
  }
  return fields;
}

CFSequentialAnalysis AnalyzeBasicBlocks(opt::IRContext& context,
                               const std::vector<BBId>& seq_blocks) {
  std::vector<AnyAnalysis> blocks;
  std::vector<VarAccess> var_accesses;

  for (const BBId bb_id : seq_blocks) {
    const opt::BasicBlock& bb = *context.cfg()->block(bb_id);
    bb.ForEachInst([&](const opt::Instruction* inst) {
      if (inst->opcode() == spv::Op::OpVariable) {
        const spv::StorageClass storage_class =
            spv::StorageClass(inst->GetSingleWordInOperand(0));
        if (storage_class != spv::StorageClass::Function &&
            storage_class == spv::StorageClass::Private) {
          return;
        }
        const bool init_at_decl = inst->NumInOperands() == 2;
        // Initialized at declaration
        if (storage_class == spv::StorageClass::Function ||
            storage_class == spv::StorageClass::Private) {
          const State s = init_at_decl ? State::NewLeaf(Initialized::Yes)
                                       : State::NewLeaf(Initialized::No);
          var_accesses.push_back(VarStore{
              .var_id = inst->result_id(), .instruction = inst, .state = s});
        }
      } else if (inst->opcode() == spv::Op::OpFunctionCall) {
        if (!var_accesses.empty()) {
          BasicBlockAnalysis bba{std::move(var_accesses)};
          blocks.push_back(bba);
          var_accesses = {};
        }
        blocks.push_back(CFCallAnalysis{inst});
      } else if (inst->opcode() == spv::Op::OpLoad) {
        std::optional<VarAndState> vs = StateForLoadStore(context, *inst);
        if (vs.has_value()) {
          var_accesses.push_back(VarLoad{
              .var_id = vs->var_id, .instruction = inst, .state = vs->state});
        }
      } else if (inst->opcode() == spv::Op::OpStore) {
        std::optional<VarAndState> vs = StateForLoadStore(context, *inst);
        if (!vs.has_value()) {
          return;
        };
        const auto found_last_store = std::find_if(
            var_accesses.rbegin(), var_accesses.rend(),
            [&](const VarAccess& access) {
              if (const VarStore* store = std::get_if<VarStore>(&access)) {
                return store->var_id == vs->var_id &&
                       store->state.MinState() == Initialized::Yes;
              }
              return false;
            });
        if (found_last_store == var_accesses.rend()) {
          var_accesses.push_back(VarStore{
              .var_id = vs->var_id, .instruction = inst, .state = vs->state});
        }
      } else if (inst->opcode() == spv::Op::OpFunctionCall) {
        blocks.push_back(BasicBlockAnalysis { std::move(var_accesses) });
        var_accesses = std::vector<VarAccess>();
        blocks.push_back(CFCallAnalysis { inst });
      }
    });
  }
  if (!var_accesses.empty()) {
    blocks.push_back(BasicBlockAnalysis{std::move(var_accesses)});
  }
  return CFSequentialAnalysis{blocks};
}

std::optional<BBId> UninitializedReadAnalysis::CollectUnconditionalSuccessors(
    const opt::BasicBlock& start, std::vector<BBId>& out_bbs,
    std::optional<BBId> stop_at_label) const {
  const opt::BasicBlock* bb = &start;
  assert(bb->cend() != bb->cbegin());
  std::unordered_set<BBId> cycle_guard;
  while (bb->hasSuccessor() &&
         bb->terminator()->opcode() == spv::Op::OpBranch) {
    out_bbs.push_back(bb->id());
    const BBId label = bb->terminator()->GetSingleWordInOperand(0);
    assert(cycle_guard.find(label) == cycle_guard.cend() &&
           "bug or cycle in unconditional branches");
    if (label == stop_at_label) {
      // reached the end of the control flow construct we're processing
      break;
    }
    bb = context_.cfg()->block(label);
    cycle_guard.insert(label);
  };
  if (!bb->hasSuccessor()) {
    out_bbs.push_back(bb->id());
    return {};
  } else {
    return bb->id();
  }
}

CFSwitchAnalysis UninitializedReadAnalysis::AnalyzeSwitch(
    const opt::BasicBlock& header_bb) const {
  // bb is the block containing the merge, so before the construct
  const opt::Instruction& merge_inst = *header_bb.GetMergeInst();
  const BBId merge_bb = merge_inst.GetSingleWordInOperand(0);

  CFSequentialAnalysis header_analysis = 
      AnalyzeBasicBlocks(context_, std::vector{header_bb.id()});

  const opt::Instruction& branch_inst = *header_bb.terminator();
  std::vector<CFSequentialAnalysis> branch_blocks{};
  if (branch_inst.opcode() == spv::Op::OpSwitch) {
    const BBId default_label = branch_inst.GetSingleWordInOperand(1);
    const opt::BasicBlock& default_block =
        *context_.cfg()->block(default_label);
    branch_blocks.push_back(AnalyzeSequential(
        default_block, StopAtLabel{.label = merge_bb, .inclusive = false}));
    for (uint32_t i = 2; i < branch_inst.NumInOperandWords(); i += 2) {
      const BBId label = branch_inst.GetSingleWordInOperand(i + 1);
      const opt::BasicBlock& block = *context_.cfg()->block(label);
      branch_blocks.push_back(AnalyzeSequential(
          block, StopAtLabel{.label = merge_bb, .inclusive = false}));
    }
  } else {
    assert(branch_inst.opcode() == spv::Op::OpBranchConditional);
    const BBId true_label = branch_inst.GetSingleWordInOperand(1);
    const opt::BasicBlock& true_bb = *context_.cfg()->block(true_label);
    const BBId false_label = branch_inst.GetSingleWordInOperand(2);
    const opt::BasicBlock& false_bb = *context_.cfg()->block(false_label);
    branch_blocks.push_back(AnalyzeSequential(
        true_bb, StopAtLabel{.label = merge_bb, .inclusive = false}));
    branch_blocks.push_back(AnalyzeSequential(
        false_bb, StopAtLabel{.label = merge_bb, .inclusive = false}));
  }
  return CFSwitchAnalysis{
      .header = header_analysis,
      .branches = branch_blocks,
  };
}

CFLoopAnalysis UninitializedReadAnalysis::AnalyzeLoop(
    const opt::BasicBlock& header_bb) const {
  assert(header_bb.IsLoopHeader());
  const opt::Instruction& merge_inst = *header_bb.GetLoopMergeInst();
  const BBId merge_label = merge_inst.GetSingleWordInOperand(0);
  const BBId continue_label = merge_inst.GetSingleWordInOperand(1);
  const opt::BasicBlock& continue_bb = *context_.cfg()->block(continue_label);

  std::vector<BBId> unconditional_before_ids;
  const BBId cond_label =
      CollectUnconditionalSuccessors(header_bb, unconditional_before_ids)
          .value();

  const opt::BasicBlock& cond_bb = *context_.cfg()->block(cond_label);
  const opt::Instruction* cond_inst = cond_bb.terminator();
  assert(cond_inst != nullptr &&
         cond_inst->opcode() == spv::Op::OpBranchConditional);
  const BBId true_label = cond_inst->GetSingleWordInOperand(1);
  const opt::BasicBlock& true_bb = *context_.cfg()->block(true_label);
  const BBId false_label = cond_inst->GetSingleWordInOperand(2);
  const opt::BasicBlock& false_bb = *context_.cfg()->block(false_label);

  CFSequentialAnalysis header_analysis = AnalyzeSequential(
      header_bb, StopAtLabel{.label = cond_label, .inclusive = true},
      header_bb.id());
  CFSequentialAnalysis trailer_analysis =
      merge_label == false_label
          ? CFSequentialAnalysis{}
          : AnalyzeSequential(false_bb, StopAtLabel{.label = merge_label,
                                                    .inclusive = false});

  auto& header_pred = context_.cfg()->preds(header_bb.id());
  const bool do_while = std::find(header_pred.begin(), header_pred.end(),
                                  cond_label) != header_pred.end();
  CFSequentialAnalysis cond_continue = AnalyzeSequential(
      continue_bb,
      do_while ? StopAtLabel{.label = cond_label, .inclusive = true}
               : StopAtLabel{.label = header_bb.id(), .inclusive = false});
  CFSequentialAnalysis cond_body =
      do_while ? CFSequentialAnalysis{}
               : AnalyzeSequential(true_bb, StopAtLabel{.label = continue_label,
                                                        .inclusive = false});

  return CFLoopAnalysis{
      .header_analysis = header_analysis,
      .body_analysis = cond_body,
      .continue_analysis = cond_continue,
      .trailer_analysis = trailer_analysis,
  };
}

// FIXME: this whole thing is horrid and initially did something completely different
CFSequentialAnalysis UninitializedReadAnalysis::AnalyzeSequential(
    const opt::BasicBlock& start_bb, std::optional<StopAtLabel> stop_at_label,
    std::optional<BBId> ignore_header_block) const {
  std::vector<AnyAnalysis> blocks;
  // No loop/cond branch are handled here, so we should never see any label
  // twice
  std::unordered_set<BBId> cycle_guard;
  const opt::BasicBlock* bb = &start_bb;
  if (stop_at_label && start_bb.id() == stop_at_label->label &&
      !stop_at_label->inclusive) {
    return CFSequentialAnalysis{};
  }
  bool stop = false;
  while (!stop) {
    std::vector<BBId> seq_blocks{};
    assert(bb->cend() != bb->cbegin());
    while (bb->hasSuccessor()) {
      if (stop_at_label && stop_at_label->inclusive &&
          bb->id() == stop_at_label->label) {
        seq_blocks.push_back(bb->id());
        stop = true;
        break;
      }
      if (bb->terminator()->opcode() != spv::Op::OpBranch) {
        break;
      }
      seq_blocks.push_back(bb->id());
      const BBId label = bb->terminator()->GetSingleWordInOperand(0);
      if (stop_at_label && !stop_at_label->inclusive &&
          label == stop_at_label->label) {
        stop = true;
        break;
      }
      if (bb->GetMergeInst() != nullptr && bb->id() != ignore_header_block) {
        break;
      }
      assert(cycle_guard.find(label) == cycle_guard.cend() &&
             "bug or cycle in unconditional branches");
      bb = context_.cfg()->block(label);
      cycle_guard.insert(label);
    };
    if (!bb->hasSuccessor()) {
      seq_blocks.push_back(bb->id());
      stop = true;
    }
    if (!seq_blocks.empty()) {
      AnyAnalysis any = AnalyzeBasicBlocks(context_, seq_blocks);
      if (BasicBlockAnalysis* bba = std::get_if<BasicBlockAnalysis>(&any)) {
        blocks.push_back(std::move(*bba));
      } else {
        CFSequentialAnalysis seq = std::get<CFSequentialAnalysis>(any);
        blocks.insert(blocks.end(), std::make_move_iterator(seq.blocks.begin()),
                      std::make_move_iterator(seq.blocks.end()));
      }
    }

    if (stop || bb->IsReturnOrAbort()) {
      break;
    }

    const opt::Instruction* merge_inst = bb->GetMergeInst();
    assert(merge_inst != nullptr);
    const BBId merge_label = merge_inst->GetSingleWordInOperand(0);

    if (merge_inst->opcode() == spv::Op::OpLoopMerge) {
      blocks.push_back(AnalyzeLoop(*bb));
    } else {
      assert(merge_inst->opcode() == spv::Op::OpSelectionMerge);
      blocks.push_back(AnalyzeSwitch(*bb));
    }
    bb = context_.cfg()->block(merge_label);
  }
  return CFSequentialAnalysis{.blocks = blocks};
}

spv::StorageClass UninitializedReadAnalysis::GetStorageClass(VarId id) const {
  const opt::Instruction& op = *context_.get_def_use_mgr()->GetDef(id);
  assert(op.opcode() == spv::Op::OpVariable);
  return spv::StorageClass(op.GetSingleWordInOperand(0));
}

bool StateSatisfies(const State& have, const State& need) {
  // Clamp needed to Unknown
  return have
      .TryAllGreaterOrEqual(
          need.TryIntersect(State::NewLeaf(Initialized::Unknown)).value())
      .value();
}

CheckResult UninitializedReadAnalysis::Check(
    const AnyAnalysis& analysis, const VarStateMap& preconditions,
    spv::StorageClass storage_class,
    const std::unordered_map<uint32_t, CheckResult>& functions = {}) const {
  VarStateMap postconditions = preconditions;
  std::vector<UnmetPrecondition> unmet_preconditions;
  const auto append_unmet = [&](const std::vector<UnmetPrecondition>& pcs) {
    unmet_preconditions.insert(unmet_preconditions.end(), pcs.begin(),
                               pcs.end());
  };
  const auto merge_result = [&](const CheckResult& r) {
    postconditions = postconditions.Union(r.postconditions);
    append_unmet(r.unmet_preconditions);
  };
  if (const BasicBlockAnalysis* bb_analysis =
          std::get_if<BasicBlockAnalysis>(&analysis)) {
    for (const VarAccess& access : bb_analysis->var_accesses) {
      if (const VarStore* store = std::get_if<VarStore>(&access)) {
        if (GetStorageClass(store->var_id) != storage_class) {
          continue;
        }
        postconditions = postconditions.Union(store->var_id, store->state);
      } else if (const VarLoad* load = std::get_if<VarLoad>(&access)) {
        if (GetStorageClass(load->var_id) != storage_class) {
          continue;
        }
        std::optional<State> found = postconditions.Get(load->var_id);
        if (!found) {
          // Never seen before global (ok), or undeclared local variable (should
          // not happen normally)
          unmet_preconditions.push_back(
              UnmetPrecondition{.var_id = load->var_id,
                                .inst = load->instruction,
                                .state_have = State::NewLeaf(Initialized::No),
                                .state_need = load->state});
        } else if (!StateSatisfies(*found, load->state)) {
          unmet_preconditions.push_back(UnmetPrecondition{
              .var_id = load->var_id,
              .inst = load->instruction,
              .state_have = *found,
              .state_need = load->state.TryDifference(*found).value()});
        }
      }
    }
  } else if (const CFSequentialAnalysis* seq_analysis =
                 std::get_if<CFSequentialAnalysis>(&analysis)) {
    for (const auto& block : seq_analysis->blocks) {
      merge_result(Check(block, postconditions, storage_class, functions));
    }
  } else if (const CFLoopAnalysis* loop_analysis =
                 std::get_if<CFLoopAnalysis>(&analysis)) {
    merge_result(Check(loop_analysis->header_analysis, postconditions,
                       storage_class, functions));

    // Downgrade all Initialized::Yes inside the body to Unknown
    // Very ugly to do it here
    CheckResult result_body = Check(loop_analysis->body_analysis,
                                    postconditions, storage_class, functions);
    append_unmet(result_body.unmet_preconditions);
    CheckResult result_continue =
        Check(loop_analysis->continue_analysis, result_body.postconditions,
              storage_class, functions);
    result_continue.postconditions =
        result_continue.postconditions.ClampAll(Initialized::Unknown);

    merge_result(result_continue);
    merge_result(Check(loop_analysis->trailer_analysis, postconditions,
                       storage_class, functions));
  } else if (const CFSwitchAnalysis* select_analysis =
                 std::get_if<CFSwitchAnalysis>(&analysis)) {
    merge_result(Check(select_analysis->header, postconditions, storage_class,
                       functions));
    VarStateMap state_before = postconditions;
    VarStateMap branch_intersect;
    bool first = true;

    for (const auto& branch : select_analysis->branches) {
      CheckResult br = Check(branch, state_before, storage_class, functions);
      append_unmet(br.unmet_preconditions);
      if (first) {
        branch_intersect = br.postconditions;
        first = false;
      } else {
        branch_intersect = branch_intersect.Intersect(br.postconditions);
      }
    }
    postconditions = postconditions.Union(branch_intersect);
  } else if (const CFCallAnalysis* call =
                 std::get_if<CFCallAnalysis>(&analysis);
             call && storage_class == spv::StorageClass::Private) {
    // Function calls don't affect local variables, only globals
    const auto& found_callee =
        functions.find(call->inst->GetSingleWordInOperand(0));
    if (found_callee != functions.cend()) {
      const CheckResult& callee = found_callee->second;
      for (UnmetPrecondition pc : callee.unmet_preconditions) {
        std::optional<State> found = postconditions.Get(pc.var_id);
        if (!found) {
          pc.call_trace.push_back(call->inst);
          unmet_preconditions.push_back(pc);
        } else if (!StateSatisfies(found.value(), pc.state_need)) {
          pc.call_trace.push_back(call->inst);
          pc.state_have = pc.state_have.TryUnion(*found).value();
          pc.state_need = pc.state_need.TryDifference(*found).value();
          unmet_preconditions.push_back(pc);
        }
      }
      postconditions = postconditions.Union(callee.postconditions);
    }
  }
  return CheckResult{unmet_preconditions, postconditions};
}

CheckResult UninitializedReadAnalysis::RunLocal() const {
  const CFSequentialAnalysis analysis = AnalyzeSequential(*function_.entry());
  return Check(analysis, {}, spv::StorageClass::Function);
}

CheckResult UninitializedReadAnalysis::RunPrivate(
    const std::unordered_map<uint32_t, CheckResult>& functions) const {
  const CFSequentialAnalysis analysis = AnalyzeSequential(*function_.entry());
  return Check(analysis, {}, spv::StorageClass::Private, functions);
}

CheckUninitializedResult ModuleAnalysis::Run() const {
  std::unordered_map<uint32_t, UninitializedReadAnalysis> functions;
  std::unordered_map<uint32_t, CheckResult> propagated;

  std::vector<LocalResult> local_results;
  for (const auto& function : *context_.module()) {
    UninitializedReadAnalysis analysis(context_, function);
    CheckResult result_local = analysis.RunLocal();
    if (!result_local.unmet_preconditions.empty()) {
      std::vector<BadLocalAccess> bad_accesses;
      for (UnmetPrecondition& pc : result_local.unmet_preconditions) {
        bad_accesses.push_back(
            BadLocalAccess{.var_id = pc.var_id,
                           .load = pc.inst,
                           .state_have = std::move(pc.state_have),
                           .state_missing = std::move(pc.state_need)});
      }
      local_results.push_back(LocalResult{.function_id = function.result_id(),
                                          .bad_accesses = bad_accesses});
    }
    functions.insert({function.result_id(), std::move(analysis)});
    propagated.insert({function.result_id(), {}});
  }

  const int SAFETY = 1000;
  for (int i = 0; i < SAFETY; i++) {
    bool any_change = false;
    auto results_erased_unmet = propagated;
    for (auto it : results_erased_unmet) {
      it.second.unmet_preconditions.clear();
    }
    for (const auto& it : functions) {
      const uint32_t function_id = it.first;
      const auto& function = it.second;
      const CheckResult result = function.RunPrivate(results_erased_unmet);
      CheckResult& old_result = propagated.at(function_id);
      if (!old_result.postconditions.GreaterOrEqual(result.postconditions)) {
        old_result.postconditions =
            old_result.postconditions.Union(result.postconditions);
        any_change = true;
      }
      propagated.at(function_id).unmet_preconditions =
          result.unmet_preconditions;
    }
    if (!any_change) {
      break;
    }
  }

  {
    auto results_erased_unmet = propagated;
    for (auto it : results_erased_unmet) {
      it.second.unmet_preconditions.clear();
    }
    for (const auto& it : functions) {
      const uint32_t function_id = it.first;
      const auto& function = it.second;
      const CheckResult result = function.RunPrivate(results_erased_unmet);
      assert(propagated.at(function_id)
                 .postconditions.Equals(result.postconditions));
      propagated.at(function_id).unmet_preconditions =
          result.unmet_preconditions;
    }
  }

  std::vector<GlobalResult> global_results;
  std::unordered_map<uint32_t, CheckResult> final_results;
  for (const auto& it : functions) {
    const uint32_t function_id = it.first;
    const auto& function = it.second;
    const CheckResult result = function.RunPrivate(propagated);
    final_results.insert({function_id, result});
  }

  for (opt::Instruction& ep : context_.module()->entry_points()) {
    const uint32_t function_id = ep.GetSingleWordInOperand(1);
    CheckResult& ep_result = final_results.at(function_id);
    if (!ep_result.unmet_preconditions.empty()) {
      std::vector<BadGlobalAccess> bad_accesses;
      for (UnmetPrecondition& pc : ep_result.unmet_preconditions) {
        bad_accesses.push_back(BadGlobalAccess{
            .var_id = pc.var_id,
            .op_load = pc.inst,
            .state_have = std::move(pc.state_have),
            .state_missing = std::move(pc.state_need),
            .call_sequence = std::move(pc.call_trace),
        });
      }
      global_results.push_back(
          GlobalResult{.op_entrypoint = &ep, .bad_accesses = bad_accesses});
    }
  }
  return CheckUninitializedResult{local_results, global_results};
}

}  // namespace uninitialized_variables

namespace {
struct DebugInfo {
  std::string file;
  uint32_t line;
  uint32_t col;
  std::optional<std::string> source_line;
};

std::string GetName(opt::IRContext& context, uint32_t id) {
  auto names = context.GetNames(id);
  if (!names.empty()) {
    return names.begin()->second->GetInOperand(1).AsString();
  } else {
    std::stringstream s;
    s << "%" << id;
    return s.str();
  }
}

std::optional<DebugInfo> GetDebugInfo(opt::IRContext& context, const opt::Instruction& def_inst) {
  uint32_t line;
  uint32_t col;
  std::string file;

  const opt::Instruction* line_inst = def_inst.dbg_line_inst();
  if (line_inst == nullptr) {
    return std::nullopt;
  }
  line = line_inst->GetSingleWordInOperand(1);
  col = line_inst->GetSingleWordInOperand(2);
  file = context.get_def_use_mgr()
             ->GetDef(line_inst->GetSingleWordInOperand(0))
             ->GetInOperand(0)
             .AsString();

  std::optional<std::string> source;
  context.module()->ForEachInst([&](const opt::Instruction* inst) {
    if (inst->opcode() == spv::Op::OpSource) {
      source = inst->GetInOperand(3).AsString();
    }
  });

  std::optional<std::string> source_line;
  if (source.has_value()) {
    std::stringstream ss(source.value());
    std::string sl;

    uint32_t l = 1;
    while (std::getline(ss, sl)) {
      if (sl.substr(0, 6).compare("#line ") == 0) {
        l = std::stoi(sl.substr(6));
        continue;
      }
      if (l == line) {
        source_line = sl;
        break;
      }
      ++l;
    }
  }
  return DebugInfo{file, line, col, source_line};
}

}  // namespace

namespace lints {

bool CheckUninitializedVariables(opt::IRContext* context) {
  using namespace uninitialized_variables;
  ModuleAnalysis module(*context);
  std::stringstream out;
  CheckUninitializedResult r = module.Run();
  for (const LocalResult& result : r.locals) {
    std::string fn_name = GetName(*context, result.function_id);
    std::optional<DebugInfo> fn = GetDebugInfo(*context, *context->get_def_use_mgr()->GetDef(result.function_id));
    if (fn.has_value()) {
      out << fn->file << ":" << fn->line << ":" << fn->col << ":\t";
    }
    out << "In function " << fn_name << ":\n";
    for (const auto& access : result.bad_accesses) {
      std::optional<DebugInfo> dbg =
          GetDebugInfo(*context, *access.load);
      std::string name = GetName(*context, access.var_id);
      if (dbg.has_value()) {
        out << dbg->file << ":" << dbg->line << ":" << dbg->col << ":\t";
      }
      out << "Load from potentially uninitialized local variable " << name
          << "\n";
      if (dbg.has_value() && dbg->source_line.has_value()) {
        out << "\t>" << dbg->source_line.value() << "\n";
      } else {
        out << "\t> " << access.load->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) << "\n";
      }
    }
    out << "\n";
  }

  if (!r.locals.empty() && !r.globals.empty()) {
    out << "\n";
  }
  for (const GlobalResult& result : r.globals) {
    const uint32_t ep_id = result.op_entrypoint->GetSingleWordOperand(1);
    std::string ep_name = GetName(*context, ep_id);
    std::optional<DebugInfo> ep = GetDebugInfo(*context, *result.op_entrypoint);
    if (ep.has_value()) {
      out << ep->file << ":" << ep->line << ":" << ep->col << ":\t";
    }
    out << "In entrypoint " << ep_name << ":\n";
    for (const auto& access : result.bad_accesses) {
      std::optional<DebugInfo> dbg =
          GetDebugInfo(*context, *access.op_load);
      std::string name = GetName(*context, access.var_id);
      out << "Load from potentially uninitialized module-private variable " << name
          << "\n";
      if (!access.call_sequence.empty()) {
        bool first = true;
        for (const auto& call : access.call_sequence) {
          uint32_t callee_id = call->GetSingleWordInOperand(0);
          std::string callee_name = GetName(*context, callee_id);

          std::optional<DebugInfo> call_dbg = GetDebugInfo(*context, *call);
          if (call_dbg.has_value()) {
            out << call_dbg->file << ":" << call_dbg->line << ":" << call_dbg->col << ":\t";
          }
          if (first) {
            out << "In call to " << callee_name << ":\n";
            first = false;
          } else {
            out << "Called by " << callee_name << ":\n";
          }

          if (call_dbg.has_value() && call_dbg->source_line.has_value()) {
            out << "\t> " << call_dbg->source_line.value() << "\n";
          } else {
            out << "\t> " << call->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) << "\n";
          }
        }
      }
      if (dbg.has_value()) {
        out << dbg->file << ":" << dbg->line << ":" << dbg->col << ":\n";
      }
      if (dbg.has_value() && dbg->source_line.has_value()) {
        out << "\t> " << dbg->source_line.value() << "\n";
      } else {
        out << "\t> " << access.op_load->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) << "\n";
      }
    }
  }

  std::cerr << out.str();
  return r.globals.empty() && r.locals.empty();
}

}  // namespace lints
}  // namespace lint
}  // namespace spvtools
