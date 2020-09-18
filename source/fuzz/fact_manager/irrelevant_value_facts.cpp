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

#include "source/fuzz/fact_manager/irrelevant_value_facts.h"

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fact_manager/data_synonym_and_id_equation_facts.h"
#include "source/fuzz/fact_manager/dead_block_facts.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

void IrrelevantValueFacts::AddFact(
    const protobufs::FactPointeeValueIsIrrelevant& fact,
    const DataSynonymAndIdEquationFacts& data_synonym_and_id_equation_facts,
    opt::IRContext* context) {
  (void)data_synonym_and_id_equation_facts;  // Keep release compilers happy.
  assert(data_synonym_and_id_equation_facts.GetSynonymsForId(fact.pointer_id())
             .empty() &&
         "The id cannot participate in DataSynonym facts.");
  auto pointer_def = context->get_def_use_mgr()->GetDef(fact.pointer_id());
  assert(pointer_def && "The id must exist in the module.");
  auto type = context->get_type_mgr()->GetType(pointer_def->type_id());
  (void)type;  // Keep release compilers happy.
  assert(type && type->AsPointer() && "The id must be a pointer.");

  pointers_to_irrelevant_pointees_ids_.insert(fact.pointer_id());
}

void IrrelevantValueFacts::AddFact(
    const protobufs::FactIdIsIrrelevant& fact,
    const DataSynonymAndIdEquationFacts& data_synonym_and_id_equation_facts,
    opt::IRContext* context) {
  (void)data_synonym_and_id_equation_facts;  // Keep release compilers happy.
  assert(data_synonym_and_id_equation_facts.GetSynonymsForId(fact.result_id())
             .empty() &&
         "The id cannot participate in DataSynonym facts.");
  auto pointer_def = context->get_def_use_mgr()->GetDef(fact.result_id());
  assert(pointer_def && "The id must exist in the module.");
  auto type = context->get_type_mgr()->GetType(pointer_def->type_id());
  (void)type;  // Keep release compilers happy.
  assert(type && !type->AsPointer() && "The id must not be a pointer.");

  irrelevant_ids_.insert(fact.result_id());
}

bool IrrelevantValueFacts::PointeeValueIsIrrelevant(uint32_t pointer_id) const {
  return pointers_to_irrelevant_pointees_ids_.count(pointer_id) != 0;
}

bool IrrelevantValueFacts::IdIsIrrelevant(
    uint32_t result_id, const DeadBlockFacts& dead_block_facts,
    opt::IRContext* context) const {
  // The id is irrelevant if it has been declared irrelevant.
  if (irrelevant_ids_.count(result_id)) {
    return true;
  }

  // The id must have a non-pointer type to be irrelevant.
  auto def = context->get_def_use_mgr()->GetDef(result_id);
  if (!def) {
    return false;
  }
  auto type = context->get_type_mgr()->GetType(def->type_id());
  if (!type || type->AsPointer()) {
    return false;
  }

  // The id is irrelevant if it is in a dead block.
  return context->get_instr_block(result_id) &&
         dead_block_facts.BlockIsDead(
             context->get_instr_block(result_id)->id());
}

std::unordered_set<uint32_t> IrrelevantValueFacts::GetIrrelevantIds(
    const DeadBlockFacts& dead_block_facts, opt::IRContext* context) const {
  // Get all the ids that have been declared irrelevant.
  auto irrelevant_ids = irrelevant_ids_;

  // Get all the non-pointer ids declared in dead blocks that have a type.
  for (uint32_t block_id : dead_block_facts.GetDeadBlocks()) {
    auto block = fuzzerutil::MaybeFindBlock(context, block_id);
    // It is possible and allowed for the block not to exist, e.g. it could have
    // been merged with another block.
    if (!block) {
      continue;
    }
    block->ForEachInst([context, &irrelevant_ids](opt::Instruction* inst) {
      // The instruction must have a result id and a type, and it must not be a
      // pointer.
      if (inst->HasResultId() && inst->type_id() &&
          !context->get_type_mgr()->GetType(inst->type_id())->AsPointer()) {
        irrelevant_ids.emplace(inst->result_id());
      }
    });
  }

  return irrelevant_ids;
}

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools
