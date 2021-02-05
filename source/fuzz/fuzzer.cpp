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

#include "source/fuzz/fuzzer.h"

#include <cassert>
#include <memory>
#include <numeric>

#include "source/fuzz/fact_manager/fact_manager.h"
#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass_add_access_chains.h"
#include "source/fuzz/fuzzer_pass_add_bit_instruction_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_composite_extract.h"
#include "source/fuzz/fuzzer_pass_add_composite_inserts.h"
#include "source/fuzz/fuzzer_pass_add_composite_types.h"
#include "source/fuzz/fuzzer_pass_add_copy_memory.h"
#include "source/fuzz/fuzzer_pass_add_dead_blocks.h"
#include "source/fuzz/fuzzer_pass_add_dead_breaks.h"
#include "source/fuzz/fuzzer_pass_add_dead_continues.h"
#include "source/fuzz/fuzzer_pass_add_equation_instructions.h"
#include "source/fuzz/fuzzer_pass_add_function_calls.h"
#include "source/fuzz/fuzzer_pass_add_global_variables.h"
#include "source/fuzz/fuzzer_pass_add_image_sample_unused_components.h"
#include "source/fuzz/fuzzer_pass_add_loads.h"
#include "source/fuzz/fuzzer_pass_add_local_variables.h"
#include "source/fuzz/fuzzer_pass_add_loop_preheaders.h"
#include "source/fuzz/fuzzer_pass_add_loops_to_create_int_constant_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_no_contraction_decorations.h"
#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_parameters.h"
#include "source/fuzz/fuzzer_pass_add_relaxed_decorations.h"
#include "source/fuzz/fuzzer_pass_add_stores.h"
#include "source/fuzz/fuzzer_pass_add_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_vector_shuffle_instructions.h"
#include "source/fuzz/fuzzer_pass_adjust_branch_weights.h"
#include "source/fuzz/fuzzer_pass_adjust_function_controls.h"
#include "source/fuzz/fuzzer_pass_adjust_loop_controls.h"
#include "source/fuzz/fuzzer_pass_adjust_memory_operands_masks.h"
#include "source/fuzz/fuzzer_pass_adjust_selection_controls.h"
#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"
#include "source/fuzz/fuzzer_pass_construct_composites.h"
#include "source/fuzz/fuzzer_pass_copy_objects.h"
#include "source/fuzz/fuzzer_pass_donate_modules.h"
#include "source/fuzz/fuzzer_pass_duplicate_regions_with_selections.h"
#include "source/fuzz/fuzzer_pass_expand_vector_reductions.h"
#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_inline_functions.h"
#include "source/fuzz/fuzzer_pass_interchange_signedness_of_integer_operands.h"
#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"
#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"
#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"
#include "source/fuzz/fuzzer_pass_merge_blocks.h"
#include "source/fuzz/fuzzer_pass_merge_function_returns.h"
#include "source/fuzz/fuzzer_pass_mutate_pointers.h"
#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"
#include "source/fuzz/fuzzer_pass_outline_functions.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_pass_permute_instructions.h"
#include "source/fuzz/fuzzer_pass_permute_phi_operands.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_down.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_up.h"
#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"
#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"
#include "source/fuzz/fuzzer_pass_replace_branches_from_dead_blocks_with_exits.h"
#include "source/fuzz/fuzzer_pass_replace_copy_memories_with_loads_stores.h"
#include "source/fuzz/fuzzer_pass_replace_copy_objects_with_stores_loads.h"
#include "source/fuzz/fuzzer_pass_replace_irrelevant_ids.h"
#include "source/fuzz/fuzzer_pass_replace_linear_algebra_instructions.h"
#include "source/fuzz/fuzzer_pass_replace_loads_stores_with_copy_memories.h"
#include "source/fuzz/fuzzer_pass_replace_opphi_ids_from_dead_predecessors.h"
#include "source/fuzz/fuzzer_pass_replace_opselects_with_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_replace_parameter_with_global.h"
#include "source/fuzz/fuzzer_pass_replace_params_with_struct.h"
#include "source/fuzz/fuzzer_pass_split_blocks.h"
#include "source/fuzz/fuzzer_pass_swap_commutable_operands.h"
#include "source/fuzz/fuzzer_pass_swap_conditional_branch_operands.h"
#include "source/fuzz/fuzzer_pass_toggle_access_chain_instruction.h"
#include "source/fuzz/fuzzer_pass_wrap_regions_in_selections.h"
#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_recommender_standard.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/build_module.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

Fuzzer::Fuzzer(spv_target_env target_env, MessageConsumer consumer,
               std::vector<uint32_t> binary_in,
               protobufs::FactSequence initial_facts,
               std::vector<fuzzerutil::ModuleSupplier> donor_suppliers,
               std::unique_ptr<RandomGenerator> random_generator,
               bool enable_all_passes,
               RepeatedPassStrategy repeated_pass_strategy,
               bool validate_after_each_fuzzer_pass,
               spv_validator_options validator_options,
               bool continue_fuzzing_probabilistically)
    : target_env_(target_env),
      consumer_(std::move(consumer)),
      binary_in_(std::move(binary_in)),
      initial_facts_(std::move(initial_facts)),
      donor_suppliers_(std::move(donor_suppliers)),
      random_generator_(std::move(random_generator)),
      enable_all_passes_(enable_all_passes),
      repeated_pass_strategy_(repeated_pass_strategy),
      validate_after_each_fuzzer_pass_(validate_after_each_fuzzer_pass),
      validator_options_(validator_options),
      continue_fuzzing_probabilistically_(continue_fuzzing_probabilistically),
      state_() {}

Fuzzer::State::State(
    std::unique_ptr<opt::IRContext> ir_context_,
    std::unique_ptr<FuzzerContext> fuzzer_context_,
    std::unique_ptr<TransformationContext> transformation_context_)
    : num_repeated_passes_applied(0),
      ir_context(std::move(ir_context_)),
      fuzzer_context(std::move(fuzzer_context_)),
      transformation_context(std::move(transformation_context_)),
      transformation_sequence_out(),
      pass_instances(),
      // These objects will be initialized in the |BuildState| method.
      repeated_pass_recommender(),
      repeated_pass_manager() {}

namespace {
const uint32_t kIdBoundGap = 100;

}  // namespace

Fuzzer::~Fuzzer() = default;

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddRepeatedPass(uint32_t percentage_chance_of_adding_pass,
                                  RepeatedPassInstances* pass_instances,
                                  Args&&... extra_args) {
  if (enable_all_passes_ || state_->fuzzer_context->ChoosePercentage(
                                percentage_chance_of_adding_pass)) {
    pass_instances->SetPass(MakeUnique<FuzzerPassT>(
        state_->ir_context.get(), state_->transformation_context.get(),
        state_->fuzzer_context.get(), &state_->transformation_sequence_out,
        std::forward<Args>(extra_args)...));
  }
}

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddFinalPass(std::vector<std::unique_ptr<FuzzerPass>>* passes,
                               Args&&... extra_args) {
  if (enable_all_passes_ || state_->fuzzer_context->ChooseEven()) {
    passes->push_back(MakeUnique<FuzzerPassT>(
        state_->ir_context.get(), state_->transformation_context.get(),
        state_->fuzzer_context.get(), &state_->transformation_sequence_out,
        std::forward<Args>(extra_args)...));
  }
}

bool Fuzzer::ApplyPassAndCheckValidity(FuzzerPass* pass) const {
  pass->Apply();
  return !validate_after_each_fuzzer_pass_ ||
         fuzzerutil::IsValidAndWellFormed(state_->ir_context.get(),
                                          validator_options_, consumer_);
}

const protobufs::TransformationSequence& Fuzzer::GetAppliedTransformations()
    const {
  assert(state_ && "Run method hasn't been called");
  return state_->transformation_sequence_out;
}

void Fuzzer::BuildState() {
  assert(!state_ && "state_ has already been initialized");

  {
    // Variables in this scope are unique pointers that will be invalidated when
    // the |state_| is created. Thus, it's better not to keep them around when
    // that happens.

    // Build the module from the input binary.
    auto ir_context = BuildModule(target_env_, consumer_, binary_in_.data(),
                                  binary_in_.size());
    assert(ir_context && "SPIR-V binary should've already been validated");

    // The fuzzer will introduce new ids into the module.  The module's id bound
    // gives the smallest id that can be used for this purpose.  We add an
    // offset to this so that there is a sizeable gap between the ids used in
    // the original module and the ids used for fuzzing, as a readability aid.
    //
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541) consider
    // the case where the maximum id bound is reached.
    auto minimum_fresh_id = ir_context->module()->id_bound() + kIdBoundGap;
    auto fuzzer_context =
        MakeUnique<FuzzerContext>(random_generator_.get(), minimum_fresh_id);

    auto transformation_context = MakeUnique<TransformationContext>(
        MakeUnique<FactManager>(ir_context.get()), validator_options_);
    transformation_context->GetFactManager()->AddInitialFacts(consumer_,
                                                              initial_facts_);

    state_ = MakeUnique<State>(std::move(ir_context), std::move(fuzzer_context),
                               std::move(transformation_context));
  }

  // The following passes are likely to be very useful: many other passes
  // introduce synonyms, irrelevant ids and constants that these passes can work
  // with.  We thus enable them with high probability.
  MaybeAddRepeatedPass<FuzzerPassObfuscateConstants>(90,
                                                     &state_->pass_instances);
  MaybeAddRepeatedPass<FuzzerPassApplyIdSynonyms>(90, &state_->pass_instances);
  MaybeAddRepeatedPass<FuzzerPassReplaceIrrelevantIds>(90,
                                                       &state_->pass_instances);

  do {
    // Each call to MaybeAddRepeatedPass randomly decides whether the given pass
    // should be enabled, and adds an instance of the pass to |pass_instances|
    // if it is enabled.
    MaybeAddRepeatedPass<FuzzerPassAddAccessChains>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddBitInstructionSynonyms>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeExtract>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeInserts>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeTypes>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddCopyMemory>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBlocks>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBreaks>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddDeadContinues>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddEquationInstructions>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddFunctionCalls>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddGlobalVariables>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddImageSampleUnusedComponents>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddLoads>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddLocalVariables>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddLoopPreheaders>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddLoopsToCreateIntConstantSynonyms>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddOpPhiSynonyms>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddParameters>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddRelaxedDecorations>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddStores>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddSynonyms>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassAddVectorShuffleInstructions>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassConstructComposites>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassCopyObjects>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassDonateModules>(&state_->pass_instances,
                                                  donor_suppliers_);
    MaybeAddRepeatedPass<FuzzerPassDuplicateRegionsWithSelections>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassExpandVectorReductions>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassFlattenConditionalBranches>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassInlineFunctions>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassInvertComparisonOperators>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassMakeVectorOperationsDynamic>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassMergeBlocks>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassMergeFunctionReturns>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassMutatePointers>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassOutlineFunctions>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPermuteBlocks>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPermuteFunctionParameters>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPermuteInstructions>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPropagateInstructionsDown>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPropagateInstructionsUp>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassPushIdsThroughVariables>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceAddsSubsMulsWithCarryingExtended>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceBranchesFromDeadBlocksWithExits>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyMemoriesWithLoadsStores>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyObjectsWithStoresLoads>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceLoadsStoresWithCopyMemories>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceParameterWithGlobal>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceLinearAlgebraInstructions>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpPhiIdsFromDeadPredecessors>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpSelectsWithConditionalBranches>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassReplaceParamsWithStruct>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassSplitBlocks>(&state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassSwapBranchConditionalOperands>(
        &state_->pass_instances);
    MaybeAddRepeatedPass<FuzzerPassWrapRegionsInSelections>(
        &state_->pass_instances);
    // There is a theoretical possibility that no pass instances were created
    // until now; loop again if so.
  } while (state_->pass_instances.GetPasses().empty());

  state_->repeated_pass_recommender =
      MakeUnique<RepeatedPassRecommenderStandard>(&state_->pass_instances,
                                                  state_->fuzzer_context.get());
  state_->repeated_pass_manager = RepeatedPassManager::Create(
      repeated_pass_strategy_, state_->fuzzer_context.get(),
      &state_->pass_instances, state_->repeated_pass_recommender.get());
}

Fuzzer::FuzzerResult Fuzzer::Run(uint32_t num_of_transformations_to_apply) {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (!state_) {
    spvtools::SpirvTools tools(target_env_);
    tools.SetMessageConsumer(consumer_);
    if (!tools.IsValid()) {
      consumer_(SPV_MSG_ERROR, nullptr, {},
                "Failed to create SPIRV-Tools interface; stopping.");
      return {FuzzerResultStatus::kFailedToCreateSpirvToolsInterface, {}};
    }

    // Initial binary should be valid.
    if (!tools.Validate(binary_in_.data(), binary_in_.size(),
                        validator_options_)) {
      consumer_(SPV_MSG_ERROR, nullptr, {},
                "Initial binary is invalid; stopping.");
      return {FuzzerResultStatus::kInitialBinaryInvalid, {}};
    }

    BuildState();
  }

  assert(state_ && "state_ is not initialized");

  const auto initial_num_of_transformations = static_cast<uint32_t>(
      state_->transformation_sequence_out.transformation_size());
  if (initial_num_of_transformations >=
      state_->fuzzer_context->GetTransformationLimit()) {
    return {FuzzerResultStatus::kTransformationLimitReached, {}};
  }

  if (state_->num_repeated_passes_applied >=
      state_->fuzzer_context->GetTransformationLimit()) {
    return {FuzzerResultStatus::kFuzzerStuck, {}};
  }

  do {
    if (!ApplyPassAndCheckValidity(state_->repeated_pass_manager->ChoosePass(
            state_->transformation_sequence_out))) {
      return {Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule, {}};
    }
  } while (ShouldContinueFuzzing(initial_num_of_transformations,
                                 num_of_transformations_to_apply));

  // Now apply some passes that it does not make sense to apply repeatedly,
  // as they do not unlock other passes.
  std::vector<std::unique_ptr<FuzzerPass>> final_passes;
  MaybeAddFinalPass<FuzzerPassAdjustBranchWeights>(&final_passes);
  MaybeAddFinalPass<FuzzerPassAdjustFunctionControls>(&final_passes);
  MaybeAddFinalPass<FuzzerPassAdjustLoopControls>(&final_passes);
  MaybeAddFinalPass<FuzzerPassAdjustMemoryOperandsMasks>(&final_passes);
  MaybeAddFinalPass<FuzzerPassAdjustSelectionControls>(&final_passes);
  MaybeAddFinalPass<FuzzerPassAddNoContractionDecorations>(&final_passes);
  MaybeAddFinalPass<FuzzerPassInterchangeSignednessOfIntegerOperands>(
      &final_passes);
  MaybeAddFinalPass<FuzzerPassInterchangeZeroLikeConstants>(&final_passes);
  MaybeAddFinalPass<FuzzerPassPermutePhiOperands>(&final_passes);
  MaybeAddFinalPass<FuzzerPassSwapCommutableOperands>(&final_passes);
  MaybeAddFinalPass<FuzzerPassToggleAccessChainInstruction>(&final_passes);
  for (auto& pass : final_passes) {
    if (!ApplyPassAndCheckValidity(pass.get())) {
      return {Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule, {}};
    }
  }
  // Encode the module as a binary.
  std::vector<uint32_t> binary_out;
  state_->ir_context->module()->ToBinary(&binary_out, false);

  return {Fuzzer::FuzzerResultStatus::kComplete, std::move(binary_out)};
}

bool Fuzzer::ShouldContinueFuzzing(uint32_t initial_num_of_transformations,
                                   uint32_t num_of_transformations_to_apply) {
  // There's a risk that fuzzing could get stuck, if none of the enabled fuzzer
  // passes are able to apply any transformations.  To guard against this we
  // count the number of times some repeated pass has been applied and ensure
  // that fuzzing stops if the number of repeated passes hits the limit on the
  // number of transformations that can be applied.
  assert(
      state_->num_repeated_passes_applied <=
          state_->fuzzer_context->GetTransformationLimit() &&
      "The number of repeated passes applied must not exceed its upper limit.");
  if (state_->ir_context->module()->id_bound() >=
      state_->fuzzer_context->GetIdBoundLimit()) {
    return false;
  }
  if (state_->num_repeated_passes_applied ==
      state_->fuzzer_context->GetTransformationLimit()) {
    // Stop because fuzzing has got stuck.
    return false;
  }
  auto transformations_applied_so_far = static_cast<uint32_t>(
      state_->transformation_sequence_out.transformation_size());
  if (transformations_applied_so_far >=
      state_->fuzzer_context->GetTransformationLimit()) {
    // Stop because we have reached the transformation limit.
    return false;
  }
  assert(transformations_applied_so_far >= initial_num_of_transformations &&
         "Number of transformations cannot decrease");
  if (num_of_transformations_to_apply != 0 &&
      transformations_applied_so_far - initial_num_of_transformations >=
          num_of_transformations_to_apply) {
    // Stop because we've applied the maximum number of transformations for a
    // single execution of a |Run| method.
    return false;
  }
  if (continue_fuzzing_probabilistically_) {
    // If we have applied T transformations so far, and the limit on the number
    // of transformations to apply is L (where T < L), the chance that we will
    // continue fuzzing is:
    //
    //     1 - T/(2*L)
    //
    // That is, the chance of continuing decreases as more transformations are
    // applied.  Using 2*L instead of L increases the number of transformations
    // that are applied on average.
    auto chance_of_continuing = static_cast<uint32_t>(
        100.0 *
        (1.0 -
         (static_cast<double>(transformations_applied_so_far) /
          (2.0 * static_cast<double>(
                     state_->fuzzer_context->GetTransformationLimit())))));
    if (!state_->fuzzer_context->ChoosePercentage(chance_of_continuing)) {
      // We have probabilistically decided to stop.
      return false;
    }
  }
  // Continue fuzzing!
  state_->num_repeated_passes_applied++;
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
