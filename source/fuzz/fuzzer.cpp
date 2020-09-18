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
#include "source/fuzz/fuzzer_pass_flatten_conditional_branches.h"
#include "source/fuzz/fuzzer_pass_inline_functions.h"
#include "source/fuzz/fuzzer_pass_interchange_signedness_of_integer_operands.h"
#include "source/fuzz/fuzzer_pass_interchange_zero_like_constants.h"
#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"
#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"
#include "source/fuzz/fuzzer_pass_merge_blocks.h"
#include "source/fuzz/fuzzer_pass_mutate_pointers.h"
#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"
#include "source/fuzz/fuzzer_pass_outline_functions.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_pass_permute_instructions.h"
#include "source/fuzz/fuzzer_pass_permute_phi_operands.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_up.h"
#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"
#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"
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
#include "source/fuzz/pass_management/repeated_pass_manager.h"
#include "source/fuzz/pass_management/repeated_pass_manager_looped_with_recommendations.h"
#include "source/fuzz/pass_management/repeated_pass_manager_random_with_recommendations.h"
#include "source/fuzz/pass_management/repeated_pass_manager_simple.h"
#include "source/fuzz/pass_management/repeated_pass_recommender_standard.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/pseudo_random_generator.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/build_module.h"
#include "source/spirv_fuzzer_options.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

namespace {
const uint32_t kIdBoundGap = 100;

const uint32_t kTransformationLimit = 2000;

}  // namespace

Fuzzer::Fuzzer(spv_target_env target_env, uint32_t seed, bool enable_all_passes,
               RepeatedPassStrategy repeated_pass_strategy,
               bool validate_after_each_fuzzer_pass,
               spv_validator_options validator_options)
    : target_env_(target_env),
      seed_(seed),
      enable_all_passes_(enable_all_passes),
      repeated_pass_strategy_(repeated_pass_strategy),
      validate_after_each_fuzzer_pass_(validate_after_each_fuzzer_pass),
      validator_options_(validator_options),
      num_repeated_passes_applied_(0) {}

Fuzzer::~Fuzzer() = default;

void Fuzzer::SetMessageConsumer(MessageConsumer consumer) {
  consumer_ = std::move(consumer);
}

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddRepeatedPass(
    RepeatedPassInstances* pass_instances, opt::IRContext* ir_context,
    TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformation_sequence_out,
    Args&&... extra_args) const {
  if (enable_all_passes_ || fuzzer_context->ChooseEven()) {
    pass_instances->SetPass(MakeUnique<FuzzerPassT>(
        ir_context, transformation_context, fuzzer_context,
        transformation_sequence_out, std::forward<Args>(extra_args)...));
  }
}

template <typename FuzzerPassT, typename... Args>
void Fuzzer::MaybeAddFinalPass(
    std::vector<std::unique_ptr<FuzzerPass>>* passes,
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformation_sequence_out,
    Args&&... extra_args) const {
  if (enable_all_passes_ || fuzzer_context->ChooseEven()) {
    passes->push_back(MakeUnique<FuzzerPassT>(
        ir_context, transformation_context, fuzzer_context,
        transformation_sequence_out, std::forward<Args>(extra_args)...));
  }
}

bool Fuzzer::ApplyPassAndCheckValidity(
    FuzzerPass* pass, const opt::IRContext& ir_context,
    const spvtools::SpirvTools& tools) const {
  pass->Apply();
  if (validate_after_each_fuzzer_pass_) {
    std::vector<uint32_t> binary_to_validate;
    ir_context.module()->ToBinary(&binary_to_validate, false);
    if (!tools.Validate(&binary_to_validate[0], binary_to_validate.size(),
                        validator_options_)) {
      consumer_(SPV_MSG_INFO, nullptr, {},
                "Binary became invalid during fuzzing (set a breakpoint to "
                "inspect); stopping.");
      return false;
    }
  }
  return true;
}

Fuzzer::FuzzerResultStatus Fuzzer::Run(
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers,
    std::vector<uint32_t>* binary_out,
    protobufs::TransformationSequence* transformation_sequence_out) {
  // Check compatibility between the library version being linked with and the
  // header files being used.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  spvtools::SpirvTools tools(target_env_);
  tools.SetMessageConsumer(consumer_);
  if (!tools.IsValid()) {
    consumer_(SPV_MSG_ERROR, nullptr, {},
              "Failed to create SPIRV-Tools interface; stopping.");
    return Fuzzer::FuzzerResultStatus::kFailedToCreateSpirvToolsInterface;
  }

  // Initial binary should be valid.
  if (!tools.Validate(&binary_in[0], binary_in.size(), validator_options_)) {
    consumer_(SPV_MSG_ERROR, nullptr, {},
              "Initial binary is invalid; stopping.");
    return Fuzzer::FuzzerResultStatus::kInitialBinaryInvalid;
  }

  // Build the module from the input binary.
  std::unique_ptr<opt::IRContext> ir_context =
      BuildModule(target_env_, consumer_, binary_in.data(), binary_in.size());
  assert(ir_context);

  // Make a PRNG from the seed passed to the fuzzer on creation.
  PseudoRandomGenerator random_generator(seed_);

  // The fuzzer will introduce new ids into the module.  The module's id bound
  // gives the smallest id that can be used for this purpose.  We add an offset
  // to this so that there is a sizeable gap between the ids used in the
  // original module and the ids used for fuzzing, as a readability aid.
  //
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541) consider the
  //  case where the maximum id bound is reached.
  auto minimum_fresh_id = ir_context->module()->id_bound() + kIdBoundGap;
  FuzzerContext fuzzer_context(&random_generator, minimum_fresh_id);

  FactManager fact_manager;
  fact_manager.AddFacts(consumer_, initial_facts, ir_context.get());
  TransformationContext transformation_context(&fact_manager,
                                               validator_options_);

  RepeatedPassInstances pass_instances{};
  do {
    // Each call to MaybeAddRepeatedPass randomly decides whether the given pass
    // should be enabled, and adds an instance of the pass to |pass_instances|
    // if it is enabled.
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3764): Consider
    //  enabling some passes always, or with higher probability.
    MaybeAddRepeatedPass<FuzzerPassAddAccessChains>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddBitInstructionSynonyms>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeInserts>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddCompositeTypes>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddCopyMemory>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBlocks>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddDeadBreaks>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddDeadContinues>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddEquationInstructions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddFunctionCalls>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddGlobalVariables>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddImageSampleUnusedComponents>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddLoads>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddLocalVariables>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddLoopPreheaders>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddOpPhiSynonyms>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddParameters>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddRelaxedDecorations>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddStores>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddSynonyms>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassAddVectorShuffleInstructions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassApplyIdSynonyms>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassConstructComposites>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassCopyObjects>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassDonateModules>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out, donor_suppliers);
    MaybeAddRepeatedPass<FuzzerPassDuplicateRegionsWithSelections>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassFlattenConditionalBranches>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassInlineFunctions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassInvertComparisonOperators>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassMakeVectorOperationsDynamic>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassMergeBlocks>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassMutatePointers>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassObfuscateConstants>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassOutlineFunctions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassPermuteBlocks>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassPermuteFunctionParameters>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassPermuteInstructions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassPropagateInstructionsUp>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassPushIdsThroughVariables>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceAddsSubsMulsWithCarryingExtended>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyMemoriesWithLoadsStores>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceCopyObjectsWithStoresLoads>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceLoadsStoresWithCopyMemories>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceParameterWithGlobal>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceLinearAlgebraInstructions>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceIrrelevantIds>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpPhiIdsFromDeadPredecessors>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceOpSelectsWithConditionalBranches>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassReplaceParamsWithStruct>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassSplitBlocks>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    MaybeAddRepeatedPass<FuzzerPassSwapBranchConditionalOperands>(
        &pass_instances, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    // There is a theoretical possibility that no pass instances were created
    // until now; loop again if so.
  } while (pass_instances.GetPasses().empty());

  RepeatedPassRecommenderStandard pass_recommender(&pass_instances,
                                                   &fuzzer_context);

  std::unique_ptr<RepeatedPassManager> repeated_pass_manager = nullptr;
  switch (repeated_pass_strategy_) {
    case RepeatedPassStrategy::kSimple:
      repeated_pass_manager = MakeUnique<RepeatedPassManagerSimple>(
          &fuzzer_context, &pass_instances);
      break;
    case RepeatedPassStrategy::kLoopedWithRecommendations:
      repeated_pass_manager =
          MakeUnique<RepeatedPassManagerLoopedWithRecommendations>(
              &fuzzer_context, &pass_instances, &pass_recommender);
      break;
    case RepeatedPassStrategy::kRandomWithRecommendations:
      repeated_pass_manager =
          MakeUnique<RepeatedPassManagerRandomWithRecommendations>(
              &fuzzer_context, &pass_instances, &pass_recommender);
      break;
  }

  do {
    if (!ApplyPassAndCheckValidity(repeated_pass_manager->ChoosePass(),
                                   *ir_context, tools)) {
      return Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule;
    }
  } while (
      ShouldContinueFuzzing(*transformation_sequence_out, &fuzzer_context));

  // Now apply some passes that it does not make sense to apply repeatedly,
  // as they do not unlock other passes.
  std::vector<std::unique_ptr<FuzzerPass>> final_passes;
  MaybeAddFinalPass<FuzzerPassAdjustBranchWeights>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassAdjustFunctionControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassAdjustLoopControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassAdjustMemoryOperandsMasks>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassAdjustSelectionControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassAddNoContractionDecorations>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassInterchangeSignednessOfIntegerOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassInterchangeZeroLikeConstants>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassPermutePhiOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassSwapCommutableOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddFinalPass<FuzzerPassToggleAccessChainInstruction>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  for (auto& pass : final_passes) {
    if (!ApplyPassAndCheckValidity(pass.get(), *ir_context, tools)) {
      return Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule;
    }
  }

  // Encode the module as a binary.
  ir_context->module()->ToBinary(binary_out, false);

  return Fuzzer::FuzzerResultStatus::kComplete;
}

bool Fuzzer::ShouldContinueFuzzing(
    const protobufs::TransformationSequence& transformation_sequence_out,
    FuzzerContext* fuzzer_context) {
  // There's a risk that fuzzing could get stuck, if none of the enabled fuzzer
  // passes are able to apply any transformations.  To guard against this we
  // count the number of times some repeated pass has been applied and ensure
  // that fuzzing stops if the number of repeated passes hits the limit on the
  // number of transformations that can be applied.
  assert(
      num_repeated_passes_applied_ <= kTransformationLimit &&
      "The number of repeated passes applied must not exceed its upper limit.");
  if (num_repeated_passes_applied_ == kTransformationLimit) {
    // Stop because fuzzing has got stuck.
    return false;
  }
  auto transformations_applied_so_far =
      static_cast<uint32_t>(transformation_sequence_out.transformation_size());
  if (transformations_applied_so_far >= kTransformationLimit) {
    // Stop because we have reached the transformation limit.
    return false;
  }
  auto chance_of_continuing = static_cast<uint32_t>(
      100.0 * (1.0 - (static_cast<double>(transformations_applied_so_far) /
                      static_cast<double>(kTransformationLimit))));
  if (!fuzzer_context->ChoosePercentage(chance_of_continuing)) {
    // We have probabilistically decided to stop.
    return false;
  }
  // Continue fuzzing!
  num_repeated_passes_applied_++;
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
