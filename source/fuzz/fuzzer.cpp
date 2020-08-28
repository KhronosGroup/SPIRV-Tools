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

const uint32_t kTransformationLimit = 500;

const uint32_t kChanceOfApplyingAnotherPass = 85;

const uint32_t kMaximumRecommendationAge = 5;

// A convenience method to add a fuzzer pass to |passes| with probability 0.5.
// All fuzzer passes take |ir_context|, |transformation_context|,
// |fuzzer_context| and |transformation_sequence_out| as parameters.  Extra
// arguments can be provided via |extra_args|.
//
// If the fuzzer pass is added, a pointer to the pass is returned.  Otherwise
// |nullptr| is returned.
template <typename T, typename... Args>
T* MaybeAddPass(std::vector<std::unique_ptr<FuzzerPass>>* passes,
                opt::IRContext* ir_context,
                TransformationContext* transformation_context,
                FuzzerContext* fuzzer_context,
                protobufs::TransformationSequence* transformation_sequence_out,
                Args&&... extra_args) {
  if (fuzzer_context->ChooseEven()) {
    auto pass = MakeUnique<T>(ir_context, transformation_context,
                              fuzzer_context, transformation_sequence_out,
                              std::forward<Args>(extra_args)...);
    auto result = pass.get();
    passes->push_back(std::move(pass));
    return result;
  }
  return nullptr;
}

}  // namespace

Fuzzer::Fuzzer(spv_target_env target_env, uint32_t seed,
               bool validate_after_each_fuzzer_pass,
               spv_validator_options validator_options)
    : target_env_(target_env),
      seed_(seed),
      validate_after_each_fuzzer_pass_(validate_after_each_fuzzer_pass),
      validator_options_(validator_options) {}

Fuzzer::~Fuzzer() = default;

void Fuzzer::SetMessageConsumer(MessageConsumer consumer) {
  consumer_ = std::move(consumer);
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

// This struct should have one field for every distinct fuzzer pass that can be
// applied repeatedly.  Fuzzer passes that are only intened to be run at most
// once, at the end of fuzzing, are not included.
struct Fuzzer::PassInstances {
  FuzzerPassAddAccessChains* add_access_chains;
  FuzzerPassAddCompositeInserts* add_composite_inserts;
  FuzzerPassAddCompositeTypes* add_composite_types;
  FuzzerPassAddCopyMemory* add_copy_memory;
  FuzzerPassAddDeadBlocks* add_dead_blocks;
  FuzzerPassAddDeadBreaks* add_dead_breaks;
  FuzzerPassAddDeadContinues* add_dead_continues;
  FuzzerPassAddEquationInstructions* add_equation_instructions;
  FuzzerPassAddFunctionCalls* add_function_calls;
  FuzzerPassAddGlobalVariables* add_global_variables;
  FuzzerPassAddImageSampleUnusedComponents* add_image_sample_unused_components;
  FuzzerPassAddLoads* add_loads;
  FuzzerPassAddLocalVariables* add_local_variables;
  FuzzerPassAddLoopPreheaders* add_loop_preheaders;
  FuzzerPassAddOpPhiSynonyms* add_op_phi_synonyms;
  FuzzerPassAddParameters* add_parameters;
  FuzzerPassAddRelaxedDecorations* add_relaxed_decorations;
  FuzzerPassAddStores* add_stores;
  FuzzerPassAddSynonyms* add_synonyms;
  FuzzerPassAddVectorShuffleInstructions* add_vector_shuffle_instructions;
  FuzzerPassApplyIdSynonyms* apply_id_synonyms;
  FuzzerPassConstructComposites* construct_composites;
  FuzzerPassCopyObjects* copy_objects;
  FuzzerPassDonateModules* donate_modules;
  FuzzerPassInlineFunctions* inline_functions;
  FuzzerPassInvertComparisonOperators* invert_comparison_operators;
  FuzzerPassMakeVectorOperationsDynamic* make_vector_operations_dynamic;
  FuzzerPassMergeBlocks* merge_blocks;
  FuzzerPassObfuscateConstants* obfuscate_constants;
  FuzzerPassOutlineFunctions* outline_functions;
  FuzzerPassPermuteBlocks* permute_blocks;
  FuzzerPassPermuteFunctionParameters* permute_function_parameters;
  FuzzerPassPermuteInstructions* permute_instructions;
  FuzzerPassPropagateInstructionsUp* propagate_instructions_up;
  FuzzerPassPushIdsThroughVariables* push_ids_through_variables;
  FuzzerPassReplaceAddsSubsMulsWithCarryingExtended*
      replace_adds_subs_muls_with_carrying_extended;
  FuzzerPassReplaceCopyMemoriesWithLoadsStores*
      replace_copy_memories_with_loads_stores;
  FuzzerPassReplaceCopyObjectsWithStoresLoads*
      replace_copy_objects_with_stores_loads;
  FuzzerPassReplaceLoadsStoresWithCopyMemories*
      replace_loads_stores_with_copy_memories;
  FuzzerPassReplaceParameterWithGlobal* replace_parameter_with_global;
  FuzzerPassReplaceLinearAlgebraInstructions*
      replace_linear_algebra_instructions;
  FuzzerPassReplaceParamsWithStruct* replace_params_with_struct;
  FuzzerPassSplitBlocks* split_blocks;
  FuzzerPassSwapBranchConditionalOperands* swap_branch_conditional_operands;
};

Fuzzer::FuzzerResultStatus Fuzzer::Run(
    const std::vector<uint32_t>& binary_in,
    const protobufs::FactSequence& initial_facts,
    const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers,
    std::vector<uint32_t>* binary_out,
    protobufs::TransformationSequence* transformation_sequence_out) const {
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

  // Each field will be set to |nullptr| if the corresponding fuzzer pass is
  // not available, and otherwise will be a pointer to an instance of the
  // fuzzer pass.
  PassInstances pass_instances = {};

  // Of the passes that can be applied repeatedly, decide which ones are
  // enabled.
  std::vector<std::unique_ptr<FuzzerPass>> passes;
  do {
    // Each call to MaybeAddPass randomly decides whether the given pass should
    // be enabled or not.
    //
    // Each pass that *is* enabled is added to |passes|, and a pointer to the
    // pass is set in the corresponding field of |pass_instances| to allow us
    // to refer to the instance by name.
    pass_instances.add_access_chains = MaybeAddPass<FuzzerPassAddAccessChains>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_composite_inserts =
        MaybeAddPass<FuzzerPassAddCompositeInserts>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_composite_types =
        MaybeAddPass<FuzzerPassAddCompositeTypes>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_copy_memory = MaybeAddPass<FuzzerPassAddCopyMemory>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_dead_blocks = MaybeAddPass<FuzzerPassAddDeadBlocks>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_dead_breaks = MaybeAddPass<FuzzerPassAddDeadBreaks>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_dead_continues =
        MaybeAddPass<FuzzerPassAddDeadContinues>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_equation_instructions =
        MaybeAddPass<FuzzerPassAddEquationInstructions>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_function_calls =
        MaybeAddPass<FuzzerPassAddFunctionCalls>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_global_variables =
        MaybeAddPass<FuzzerPassAddGlobalVariables>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_image_sample_unused_components =
        MaybeAddPass<FuzzerPassAddImageSampleUnusedComponents>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_loads = MaybeAddPass<FuzzerPassAddLoads>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_local_variables =
        MaybeAddPass<FuzzerPassAddLocalVariables>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_loop_preheaders =
        MaybeAddPass<FuzzerPassAddLoopPreheaders>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_op_phi_synonyms =
        MaybeAddPass<FuzzerPassAddOpPhiSynonyms>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_parameters = MaybeAddPass<FuzzerPassAddParameters>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_relaxed_decorations =
        MaybeAddPass<FuzzerPassAddRelaxedDecorations>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.add_stores = MaybeAddPass<FuzzerPassAddStores>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_synonyms = MaybeAddPass<FuzzerPassAddSynonyms>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.add_vector_shuffle_instructions =
        MaybeAddPass<FuzzerPassAddVectorShuffleInstructions>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.apply_id_synonyms = MaybeAddPass<FuzzerPassApplyIdSynonyms>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.construct_composites =
        MaybeAddPass<FuzzerPassConstructComposites>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.copy_objects = MaybeAddPass<FuzzerPassCopyObjects>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.donate_modules = MaybeAddPass<FuzzerPassDonateModules>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out, donor_suppliers);
    pass_instances.inline_functions = MaybeAddPass<FuzzerPassInlineFunctions>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.invert_comparison_operators =
        MaybeAddPass<FuzzerPassInvertComparisonOperators>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.make_vector_operations_dynamic =
        MaybeAddPass<FuzzerPassMakeVectorOperationsDynamic>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.merge_blocks = MaybeAddPass<FuzzerPassMergeBlocks>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.obfuscate_constants =
        MaybeAddPass<FuzzerPassObfuscateConstants>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.outline_functions = MaybeAddPass<FuzzerPassOutlineFunctions>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.permute_blocks = MaybeAddPass<FuzzerPassPermuteBlocks>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.permute_function_parameters =
        MaybeAddPass<FuzzerPassPermuteFunctionParameters>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.permute_instructions =
        MaybeAddPass<FuzzerPassPermuteInstructions>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.propagate_instructions_up =
        MaybeAddPass<FuzzerPassPropagateInstructionsUp>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.push_ids_through_variables =
        MaybeAddPass<FuzzerPassPushIdsThroughVariables>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_adds_subs_muls_with_carrying_extended =
        MaybeAddPass<FuzzerPassReplaceAddsSubsMulsWithCarryingExtended>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_copy_memories_with_loads_stores =
        MaybeAddPass<FuzzerPassReplaceCopyMemoriesWithLoadsStores>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_copy_objects_with_stores_loads =
        MaybeAddPass<FuzzerPassReplaceCopyObjectsWithStoresLoads>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_loads_stores_with_copy_memories =
        MaybeAddPass<FuzzerPassReplaceLoadsStoresWithCopyMemories>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_parameter_with_global =
        MaybeAddPass<FuzzerPassReplaceParameterWithGlobal>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_linear_algebra_instructions =
        MaybeAddPass<FuzzerPassReplaceLinearAlgebraInstructions>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.replace_params_with_struct =
        MaybeAddPass<FuzzerPassReplaceParamsWithStruct>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    pass_instances.split_blocks = MaybeAddPass<FuzzerPassSplitBlocks>(
        &passes, ir_context.get(), &transformation_context, &fuzzer_context,
        transformation_sequence_out);
    pass_instances.swap_branch_conditional_operands =
        MaybeAddPass<FuzzerPassSwapBranchConditionalOperands>(
            &passes, ir_context.get(), &transformation_context, &fuzzer_context,
            transformation_sequence_out);
    // There is a theoretical possibility that no pass instances were created
    // until now; loop again if so.
  } while (passes.empty());

  // Represents the current list of passes that are recommended to be run,
  // based on the passes that ran before them.
  // TODO(afd): The proposal implemented here is to have a list of
  //  recommendations, each with an age, and around half the time to randomly
  //  choose a recommendation, dropping recommendations that get too old.
  //   ...
  //  An alternative proposal is to have a *queue* of recommended passes, only
  //  push a pass to this queue with some probability (at recommendation time),
  //  and then half the time pop a pass from this queue and apply it - that
  //  might be simpler from an implementation point of view and is possibly
  //  more flexible, because passes can be pushed on to the recommendation
  //  queue with various probabilities.
  RecommendedPasses recommended_passes;

  // TODO(afd) Revisit the fuzzer's stopping condition.
  bool is_first = true;
  while (static_cast<uint32_t>(
             transformation_sequence_out->transformation_size()) <
             kTransformationLimit &&
         (is_first ||
          fuzzer_context.ChoosePercentage(kChanceOfApplyingAnotherPass))) {
    is_first = false;

    // We choose a pass to apply.  We either choose one uniformly at random
    // from the set of enabled passes, or uniformly at random from the set of
    // currently recommended passes.  This means that we have a higher chance
    // of choosing a recommended pass than a non-recommended pass, because
    // (a) we might choose the recommended pass from the list of all enabled
    // passes anyway, and (b) the list of recommendations will tend to be a lot
    // smaller than the list of all enabled passes.
    FuzzerPass* current_pass;

    if (recommended_passes.empty() || fuzzer_context.ChooseEven()) {
      // Don't use a recommendation; choose one of the enabled passes uniformly
      // at random.
      current_pass = passes[fuzzer_context.RandomIndex(passes)].get();
    } else {
      // TODO(afd): If we used the recommendation queue approach described
      //  above, then we would pop the head of the queue here and apply that
      //  pass.  The recommendations would have been pushed on to the queue in
      //  a randomized fashion.

      // Use a recommendation: choose from the recommended passes uniformly at
      // random.
      uint32_t recommendation_index =
          fuzzer_context.RandomIndex(recommended_passes);
      current_pass = recommended_passes[recommendation_index].first;
      // Remove the chosen recommendation (to reduce the chances of choosing
      // this recommendation again and again).
      recommended_passes.erase(recommended_passes.begin() +
                               recommendation_index);
    }
    if (!ApplyPassAndCheckValidity(current_pass, *ir_context, tools)) {
      return Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule;
    }
    // Make some new recommendations based on the pass that was just executed.
    UpdateRecommendedPasses(current_pass, pass_instances, &recommended_passes);
  }

  // Now apply some passes that it does not make sense to apply repeatedly,
  // as they do not unlock other passes.
  std::vector<std::unique_ptr<FuzzerPass>> final_passes;
  MaybeAddPass<FuzzerPassAdjustBranchWeights>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassAdjustFunctionControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassAdjustLoopControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassAdjustMemoryOperandsMasks>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassAdjustSelectionControls>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassAddNoContractionDecorations>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassInterchangeSignednessOfIntegerOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassInterchangeZeroLikeConstants>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassPermutePhiOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassReplaceOpPhiIdsFromDeadPredecessors>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassReplaceIrrelevantIds>(
      &passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassSwapCommutableOperands>(
      &final_passes, ir_context.get(), &transformation_context, &fuzzer_context,
      transformation_sequence_out);
  MaybeAddPass<FuzzerPassToggleAccessChainInstruction>(
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

void Fuzzer::UpdateRecommendedPasses(FuzzerPass* completed_pass,
                                     const PassInstances& pass_instances,
                                     RecommendedPasses* recommended_passes) {
  {
    // Increase the age of all existing recommendations, dropping those that get
    // too old.
    RecommendedPasses temp;
    for (auto& recommendation : *recommended_passes) {
      if (recommendation.second < kMaximumRecommendationAge) {
        temp.push_back({recommendation.first, recommendation.second + 1});
      }
    }
    *recommended_passes = temp;
  }

  // TODO(afd): This should have a case for every fuzzer pass for which
  //  |pass_instances| has a field.
  //    ...
  //  If we used the recommendation queue approach then for each recommendation
  //  we would push it to the queue with some probability, and this could be
  //  controlled per pass, and could even be dependent on the current contents
  //  of the recommendation queue.  If multiple recommendations are possible
  //  after one fuzzer pass then they should probably be processed in a random
  //  order.
  if (completed_pass == pass_instances.donate_modules) {
    RecommendPass(pass_instances.add_function_calls, recommended_passes);
  } else if (completed_pass == pass_instances.add_function_calls) {
    RecommendPass(pass_instances.inline_functions, recommended_passes);
  } else if (completed_pass == pass_instances.inline_functions) {
    RecommendPass(pass_instances.outline_functions, recommended_passes);
  }
}

void Fuzzer::RecommendPass(FuzzerPass* pass,
                           RecommendedPasses* recommended_passes) {
  if (pass == nullptr) {
    // The pass being recommended is not available.
    return;
  }
  // Check whether this pass has already been recommended.  If so, reset its
  // age.
  for (auto& recommendation : *recommended_passes) {
    if (recommendation.first == pass) {
      recommendation.second = 0;
      return;
    }
  }
  // This pass has not been recommended recently; add it to the list of
  // recommendations.
  recommended_passes->push_back({pass, 0});
}

}  // namespace fuzz
}  // namespace spvtools
