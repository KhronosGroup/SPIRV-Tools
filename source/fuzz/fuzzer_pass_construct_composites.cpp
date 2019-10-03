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

#include "source/fuzz/fuzzer_pass_construct_composites.h"

#include <cmath>
#include <memory>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_construct_composite.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

FuzzerPassConstructComposites::FuzzerPassConstructComposites(
        opt::IRContext* ir_context, FactManager* fact_manager,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
        : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassConstructComposites::~FuzzerPassConstructComposites() = default;

void FuzzerPassConstructComposites::Apply() {

  std::vector<uint32_t> composite_type_ids;
  for (auto& inst : GetIRContext()->types_values()) {
    if (fuzzerutil::IsCompositeType(GetIRContext()->get_type_mgr()->GetType(inst.result_id()))) {
      composite_type_ids.push_back(inst.result_id());
    }
  }

  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // We now consider every instruction in the block, randomly deciding
      // whether to add a composite construction before the instruction.

      // In order to insert a new instruction, we need to be able to
      // identify the existing instruction the new instruction should be inserted before.
      // We do this
      // by tracking a base instruction, which must generate a result id, and an
      // offset (to allow us to identify instructions that do not generate
      // result ids).

      // The initial base instruction is the block label.
      uint32_t base = block.id();
      uint32_t offset = 0;
      // Consider every instruction in the block.
      for (auto inst_it = block.begin(); inst_it != block.end(); ++inst_it) {
        if (inst_it->HasResultId()) {
          // In the case that the instruction has a result id, we use the
          // instruction as its own base, with zero offset.
          base = inst_it->result_id();
          offset = 0;
        } else {
          // The instruction does not have a result id, so we need to identify
          // it via the latest instruction that did have a result id (base), and
          // an incremented offset.
          offset++;
        }

        // Check whether it is legitimate to insert a composite construction
        // before the instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeConstruct, inst_it)) {
          continue;
        }

        // Randomly decide whether to try inserting an object copy here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfConstructingComposite())) {
          continue;
        }

        // Populate list of potential instructions that can be used to make a composite.
        // TODO(afd) The following is (relatively) simple, but may end up being
        //  prohibitively inefficient, as it walks the whole dominator tree for
        //  every copy that is added.

        // Maps a type id to a sequence of instructions with that result type that are
        // available at this program point (i.e. they are global or their definition
        // strictly dominates the program point).
        TypeIdToInstructions type_id_to_available_instructions;

        for (auto instruction : FindAvailableInstructions(function, &block, inst_it,
                                                          fuzzerutil::CanMakeSynonymOf)) {
          RecordAvailableInstruction(instruction, &type_id_to_available_instructions);
        }

        // At this point, |composite_type_ids| captures all the composite types we could
        // try to create, while |type_id_to_available_instructions| captures all the available
        // result ids we might use, organized by type.

        auto composites_to_try_constructing = composite_type_ids;

        uint32_t chosen_composite_type = 0;
        std::unique_ptr<std::vector<uint32_t>> constructor_arguments = nullptr;

        while(!composites_to_try_constructing.empty()) {
          auto index = GetFuzzerContext()->RandomIndex(composites_to_try_constructing);
          auto next_composite_to_try_constructing = composites_to_try_constructing[index];
          composites_to_try_constructing.erase(composites_to_try_constructing.begin() + index);
          auto composite_type = GetIRContext()->get_type_mgr()->GetType(next_composite_to_try_constructing);
          if (auto array_type = composite_type->AsArray()) {
            constructor_arguments = TryConstructingArrayComposite(*array_type, type_id_to_available_instructions);
          } else if (auto matrix_type = composite_type->AsMatrix()) {
            constructor_arguments = TryConstructingMatrixComposite(*matrix_type, type_id_to_available_instructions);
          } else if (auto struct_type = composite_type->AsStruct()) {
            constructor_arguments = TryConstructingStructComposite(*struct_type, type_id_to_available_instructions);
          } else {
            auto vector_type = composite_type->AsVector();
            assert(vector_type && "The space of possible composite types should be covered by the above cases.");
            constructor_arguments = TryConstructingVectorComposite(*vector_type, type_id_to_available_instructions);
          }
          if (constructor_arguments != nullptr) {
            chosen_composite_type = next_composite_to_try_constructing;
            break;
          }
        }

        if (!chosen_composite_type) {
          continue;
        }
        assert(constructor_arguments != nullptr);

        TransformationConstructComposite transformation(chosen_composite_type,
                                                        *constructor_arguments,
                                                        base,
                                                        offset,
                                                        GetFuzzerContext()->GetFreshId());
        assert(
                transformation.IsApplicable(GetIRContext(), *GetFactManager()) &&
                "This transformation should be applicable by construction.");
        transformation.Apply(GetIRContext(), GetFactManager());
        *GetTransformations()->add_transformation() =
                transformation.ToMessage();

        if (!inst_it->HasResultId()) {
          // We have inserted a new instruction before the current
          // instruction, and we are tracking the current id-less instruction
          // via an offset (offset) from a previous instruction (base) that
          // has an id. We increment |offset| to reflect the newly-inserted
          // instruction.
          //
          // This is slightly preferable to the alternative of setting |base|
          // to be the result id of the new instruction, since on replay we
          // might end up eliminating this copy but keeping a subsequent copy.
          offset++;
        }

      }
    }
  }
}

void FuzzerPassConstructComposites::RecordAvailableInstruction(opt::Instruction* inst,
        TypeIdToInstructions* type_id_to_available_instructions) {
  if (type_id_to_available_instructions->count(inst->type_id()) == 0) {
    (*type_id_to_available_instructions)[inst->type_id()] = {};
  }
  type_id_to_available_instructions->at(inst->type_id()).push_back(inst);
}

std::unique_ptr<std::vector<uint32_t>> FuzzerPassConstructComposites::TryConstructingArrayComposite(const opt::analysis::Array& array_type,
                                                                                                    const TypeIdToInstructions& type_id_to_available_instructions) {
  // TODO make these be true by construction
  assert(array_type.length_info().words.size() == 2);
  assert(array_type.length_info().words[0] == opt::analysis::Array::LengthInfo::kConstant);

  auto result = MakeUnique<std::vector<uint32_t>>();
  auto element_type_id = GetIRContext()->get_type_mgr()->GetId(array_type.element_type());
  auto available_instructions = type_id_to_available_instructions.find(element_type_id);
  if (available_instructions == type_id_to_available_instructions.cend()) {
    // TODO comment infeasible
    return nullptr;
  }
  for (uint32_t index = 0; index < array_type.length_info().words[1]; index++) {
    result->push_back(available_instructions->second[GetFuzzerContext()->RandomIndex(available_instructions->second)]->result_id());
  }
  return result;
}

std::unique_ptr<std::vector<uint32_t>> FuzzerPassConstructComposites::TryConstructingMatrixComposite(const opt::analysis::Matrix& matrix_type,
                                                                                                     const TypeIdToInstructions& type_id_to_available_instructions) {
  (void)(matrix_type);
  (void)(type_id_to_available_instructions);
  assert(false);
  return nullptr;
}

std::unique_ptr<std::vector<uint32_t>> FuzzerPassConstructComposites::TryConstructingStructComposite(const opt::analysis::Struct& struct_type,
        const TypeIdToInstructions& type_id_to_available_instructions) {
  auto result = MakeUnique<std::vector<uint32_t>>();
  for (auto element_type : struct_type.element_types()) {
    auto element_type_id = GetIRContext()->get_type_mgr()->GetId(element_type);
    auto available_instructions = type_id_to_available_instructions.find(element_type_id);
    if (available_instructions == type_id_to_available_instructions.cend()) {
      // TODO comment infeasible
      return nullptr;
    }
    result->push_back(
            available_instructions->second[GetFuzzerContext()->RandomIndex(available_instructions->second)]->result_id());
  }
  return result;
}

std::unique_ptr<std::vector<uint32_t>> FuzzerPassConstructComposites::TryConstructingVectorComposite(const opt::analysis::Vector& vector_type,
        const TypeIdToInstructions& type_id_to_available_instructions) {
  // Get details of the type underlying the vector, and the width of the vector, for convenience.
  auto element_type = vector_type.element_type();
  auto element_count = vector_type.element_count();

  // Collect a mapping, from type id to width, for scalar/vector types that are smaller in width than |vector_type|,
  // but that have the same underlying type.  For example, if |vector_type| is vec4, the mapping will be
  // { float -> 1, vec2 -> 2, vec3 -> 3 }.  The mapping will have missing entries if some of these
  // types do not exist.

  // TODO comment why we have the list as well.
  std::vector<uint32_t> smaller_vector_type_ids;
  std::map<uint32_t, uint32_t> smaller_vector_type_id_to_width;
  // Add the underlying type.  This id must exist, in order for |vector_type| to exist.
  auto scalar_type_id = GetIRContext()->get_type_mgr()->
          GetId(element_type);
  smaller_vector_type_ids.push_back(scalar_type_id);
  smaller_vector_type_id_to_width[scalar_type_id] = 1;

  // Now add every vector type with width at least 2, and less than the width of |vector_type|.
  for (uint32_t width = 2; width < element_count; width++) {
    opt::analysis::Vector smaller_vector_type(vector_type.element_type(), width);
    auto smaller_vector_type_id = GetIRContext()->get_type_mgr()->GetId(&smaller_vector_type);
    // TODO recap why it might be 0
    if (smaller_vector_type_id) {
      smaller_vector_type_ids.push_back(smaller_vector_type_id);
      smaller_vector_type_id_to_width[smaller_vector_type_id] = width;
    }
  }

  // Now we know the types that are available to us, we set about populating a vector of the right
  // length.  We do this by deciding, with no order in mind, which instructions we will use to
  // populate the vector, and subsequently randomly choosing an order.  This is to avoid biasing
  // construction of vectors with smaller vectors to the left and scalars to the right.  That is
  // a concern because, e.g. in the case of populating a vec4, if we populate the constructor
  // instructions left-to-right, we can always choose a vec3 to construct the first three elements,
  // but can only choose a vec3 to construct the last three elements if we chose a float to construct
  // the first element (otherwise there will not be space left for a vec3).

  uint32_t vector_slots_used = 0;
  // The instructions we will use to construct the vector, in no particular order at this stage.
  std::vector<opt::Instruction*> instructions_to_use;

  while (vector_slots_used < vector_type.element_count()) {
    std::vector<opt::Instruction*> instructions_to_choose_from;
    for (auto& entry : smaller_vector_type_id_to_width) {
      if (entry.second > std::min(vector_type.element_count() - 1, vector_type.element_count() - vector_slots_used)) {
        continue;
      }
      auto available_instructions = type_id_to_available_instructions.find(entry.first);
      if (available_instructions == type_id_to_available_instructions.cend()) {
        continue;
      }
      instructions_to_choose_from.insert(instructions_to_choose_from.end(), available_instructions->second.begin(), available_instructions->second.end());
    }
    if (instructions_to_choose_from.empty()) {
      // TODO comment - like fuzzed into a corner
      return nullptr;
    }
    auto instruction_to_use = instructions_to_choose_from[GetFuzzerContext()->RandomIndex(instructions_to_choose_from)];
    instructions_to_use.push_back(instruction_to_use);
    auto chosen_type = GetIRContext()->get_type_mgr()->GetType(instruction_to_use->type_id());
    if (chosen_type->AsVector()) {
      assert(chosen_type->AsVector()->element_type() == element_type);
      assert(chosen_type->AsVector()->element_count() < element_count);
      assert(chosen_type->AsVector()->element_count() <= element_count - vector_slots_used);
      vector_slots_used += chosen_type->AsVector()->element_count();
    } else {
      assert(chosen_type == element_type);
      vector_slots_used += 1;
    }
  }
  assert(vector_slots_used == vector_type.element_count());

  auto result = MakeUnique<std::vector<uint32_t>>();
  std::vector<uint32_t> operands;
  while(!instructions_to_use.empty()) {
    auto index = GetFuzzerContext()->RandomIndex(instructions_to_use);
    result->push_back(instructions_to_use[index]->result_id());
    instructions_to_use.erase(instructions_to_use.begin() + index);
  }
  assert(result->size() > 1);
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
