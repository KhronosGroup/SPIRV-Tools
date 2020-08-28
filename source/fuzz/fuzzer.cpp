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
#include <deque>
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

const uint32_t kChanceOfAddingAnotherPassToPassLoop = 90;

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

enum FuzzerStrategy {
  kSimple,
  kRandomWithRecommendations,
  kLoopedWithRecommendations
};

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

  // Of the passes that can be applied repeatedly, this captures those that are
  // enabled.
  std::vector<std::unique_ptr<FuzzerPass>> enabled_passes;

  do {
    // Each call to MaybeAddPass randomly decides whether the given pass should
    // be enabled or not.
    //
    // Each pass that *is* enabled is added to |passes|, and a pointer to the
    // pass is set in the corresponding field of |pass_instances| to allow us
    // to refer to the instance by name.
    pass_instances.add_access_chains = MaybeAddPass<FuzzerPassAddAccessChains>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_composite_inserts =
        MaybeAddPass<FuzzerPassAddCompositeInserts>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_composite_types =
        MaybeAddPass<FuzzerPassAddCompositeTypes>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_copy_memory = MaybeAddPass<FuzzerPassAddCopyMemory>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_dead_blocks = MaybeAddPass<FuzzerPassAddDeadBlocks>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_dead_breaks = MaybeAddPass<FuzzerPassAddDeadBreaks>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_dead_continues =
        MaybeAddPass<FuzzerPassAddDeadContinues>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_equation_instructions =
        MaybeAddPass<FuzzerPassAddEquationInstructions>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_function_calls =
        MaybeAddPass<FuzzerPassAddFunctionCalls>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_global_variables =
        MaybeAddPass<FuzzerPassAddGlobalVariables>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_image_sample_unused_components =
        MaybeAddPass<FuzzerPassAddImageSampleUnusedComponents>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_loads = MaybeAddPass<FuzzerPassAddLoads>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_local_variables =
        MaybeAddPass<FuzzerPassAddLocalVariables>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_loop_preheaders =
        MaybeAddPass<FuzzerPassAddLoopPreheaders>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_op_phi_synonyms =
        MaybeAddPass<FuzzerPassAddOpPhiSynonyms>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_parameters = MaybeAddPass<FuzzerPassAddParameters>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_relaxed_decorations =
        MaybeAddPass<FuzzerPassAddRelaxedDecorations>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.add_stores = MaybeAddPass<FuzzerPassAddStores>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_synonyms = MaybeAddPass<FuzzerPassAddSynonyms>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.add_vector_shuffle_instructions =
        MaybeAddPass<FuzzerPassAddVectorShuffleInstructions>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.apply_id_synonyms = MaybeAddPass<FuzzerPassApplyIdSynonyms>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.construct_composites =
        MaybeAddPass<FuzzerPassConstructComposites>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.copy_objects = MaybeAddPass<FuzzerPassCopyObjects>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.donate_modules = MaybeAddPass<FuzzerPassDonateModules>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out, donor_suppliers);
    pass_instances.inline_functions = MaybeAddPass<FuzzerPassInlineFunctions>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.invert_comparison_operators =
        MaybeAddPass<FuzzerPassInvertComparisonOperators>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.make_vector_operations_dynamic =
        MaybeAddPass<FuzzerPassMakeVectorOperationsDynamic>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.merge_blocks = MaybeAddPass<FuzzerPassMergeBlocks>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.obfuscate_constants =
        MaybeAddPass<FuzzerPassObfuscateConstants>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.outline_functions = MaybeAddPass<FuzzerPassOutlineFunctions>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.permute_blocks = MaybeAddPass<FuzzerPassPermuteBlocks>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.permute_function_parameters =
        MaybeAddPass<FuzzerPassPermuteFunctionParameters>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.permute_instructions =
        MaybeAddPass<FuzzerPassPermuteInstructions>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.propagate_instructions_up =
        MaybeAddPass<FuzzerPassPropagateInstructionsUp>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.push_ids_through_variables =
        MaybeAddPass<FuzzerPassPushIdsThroughVariables>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_adds_subs_muls_with_carrying_extended =
        MaybeAddPass<FuzzerPassReplaceAddsSubsMulsWithCarryingExtended>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_copy_memories_with_loads_stores =
        MaybeAddPass<FuzzerPassReplaceCopyMemoriesWithLoadsStores>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_copy_objects_with_stores_loads =
        MaybeAddPass<FuzzerPassReplaceCopyObjectsWithStoresLoads>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_loads_stores_with_copy_memories =
        MaybeAddPass<FuzzerPassReplaceLoadsStoresWithCopyMemories>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_parameter_with_global =
        MaybeAddPass<FuzzerPassReplaceParameterWithGlobal>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_linear_algebra_instructions =
        MaybeAddPass<FuzzerPassReplaceLinearAlgebraInstructions>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.replace_params_with_struct =
        MaybeAddPass<FuzzerPassReplaceParamsWithStruct>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    pass_instances.split_blocks = MaybeAddPass<FuzzerPassSplitBlocks>(
        &enabled_passes, ir_context.get(), &transformation_context,
        &fuzzer_context, transformation_sequence_out);
    pass_instances.swap_branch_conditional_operands =
        MaybeAddPass<FuzzerPassSwapBranchConditionalOperands>(
            &enabled_passes, ir_context.get(), &transformation_context,
            &fuzzer_context, transformation_sequence_out);
    // There is a theoretical possibility that no pass instances were created
    // until now; loop again if so.
  } while (enabled_passes.empty());

  FuzzerStrategy strategy = kRandomWithRecommendations;

  bool fuzzing_succeeded;
  switch (strategy) {
    case kSimple:
      fuzzing_succeeded = ApplyFuzzerPassesSimple(
          tools, enabled_passes, &fuzzer_context, ir_context.get(),
          transformation_sequence_out);
      break;
    case kRandomWithRecommendations:
      fuzzing_succeeded = ApplyFuzzerPassesRandomlyWithRecommendations(
          tools, pass_instances, enabled_passes, &fuzzer_context,
          ir_context.get(), transformation_sequence_out);
      break;
    case kLoopedWithRecommendations:
      fuzzing_succeeded = ApplyFuzzerPassesLoopedWithRecommendations(
          tools, pass_instances, enabled_passes, &fuzzer_context,
          ir_context.get(), transformation_sequence_out);
      break;
    default:
      assert(false && "Unreachable.");
      fuzzing_succeeded = false;
      break;
  }
  if (!fuzzing_succeeded) {
    return Fuzzer::FuzzerResultStatus::kFuzzerPassLedToInvalidModule;
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

std::vector<FuzzerPass*> Fuzzer::GetRecommendedFuturePasses(
    FuzzerPass* completed_pass, const PassInstances& pass_instances,
    FuzzerContext* fuzzer_context) const {
  if (completed_pass == pass_instances.add_access_chains) {
    // - Adding access chains means there is more scope for loading and storing
    // - It could be worth making more access chains from the recently-added
    //   access chains.
    return RandomOrderAndNonNull(
        {pass_instances.add_loads, pass_instances.add_stores,
         pass_instances.add_access_chains},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_composite_inserts) {
    // - Having added inserts we will have more vectors, so there is scope for
    //   vector shuffling.
    // - Adding inserts creates synonyms, which we should try to use.
    // - Vector inserts can be made dynamic.
    return RandomOrderAndNonNull(
        {pass_instances.add_vector_shuffle_instructions,
         pass_instances.apply_id_synonyms,
         pass_instances.make_vector_operations_dynamic},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_composite_types) {
    // - More composite types gives more scope for constructing composites.
    return RandomOrderAndNonNull({pass_instances.construct_composites},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_copy_memory) {
    // - Recently-added copy memories could be replace with load-store pairs.
    return RandomOrderAndNonNull(
        {pass_instances.replace_copy_memories_with_loads_stores},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_dead_blocks) {
    // - Dead blocks are great for adding function calls
    // - Dead blocks are also great for adding loads and stores
    // - The guard associated with a dead block can be obfuscated
    return RandomOrderAndNonNull(
        {pass_instances.add_function_calls, pass_instances.add_loads,
         pass_instances.add_stores, pass_instances.obfuscate_constants},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_dead_breaks) {
    // - The guard of the dead break is a good candidate for obfuscation.
    return RandomOrderAndNonNull({pass_instances.obfuscate_constants},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_dead_continues) {
    // - The guard of the dead continue is a good candidate for obfuscation.
    return RandomOrderAndNonNull({pass_instances.obfuscate_constants},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_equation_instructions) {
    // - Equation instructions can create synonyms, which we can apply
    // - Equation instructions collaborate with one another to make synonyms, so
    //   having added some it is worth adding more
    return RandomOrderAndNonNull({pass_instances.apply_id_synonyms,
                                  pass_instances.add_equation_instructions},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_function_calls) {
    // - Called functions can be inlined
    // - TODO: Called functions use irrelevant ids, which can be replaced
    return RandomOrderAndNonNull({pass_instances.inline_functions},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_global_variables) {
    // - New globals provide new possibilities for making access chains
    // - We can load from and store to new globals
    return RandomOrderAndNonNull(
        {pass_instances.add_access_chains, pass_instances.add_loads,
         pass_instances.add_stores},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_image_sample_unused_components) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.add_loads) {
    // - Loads might end up with corresponding stores, so that pairs can be
    //   replaced with memory copies.
    return RandomOrderAndNonNull(
        {pass_instances.replace_loads_stores_with_copy_memories},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_local_variables) {
    // - New locals provide new possibilities for making access chains
    // - We can load from and store to new locals
    return RandomOrderAndNonNull(
        {pass_instances.add_access_chains, pass_instances.add_loads,
         pass_instances.add_stores},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_loop_preheaders) {
    // - No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.add_op_phi_synonyms) {
    // - New synonyms can be applied
    // - TODO: the one about dead blocks and op phis.
    return RandomOrderAndNonNull({pass_instances.apply_id_synonyms},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.add_parameters) {
    // - TODO: irrelevant id replacement.
    return RandomOrderAndNonNull({}, fuzzer_context);
  }
  if (completed_pass == pass_instances.add_relaxed_decorations) {
    // - No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.add_stores) {
    // - Stores might end up with corresponding loads, so that pairs can be
    //   replaced with memory copies.
    return RandomOrderAndNonNull(
        {pass_instances.replace_loads_stores_with_copy_memories},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_synonyms) {
    // - New synonyms can be applied
    // - Synonym instructions use constants, which can be obfuscated
    // - Synonym instructions introduce addition/subtraction, which can be
    //   replaced with carrying/extended versions
    return RandomOrderAndNonNull(
        {pass_instances.apply_id_synonyms, pass_instances.obfuscate_constants,
         pass_instances.replace_adds_subs_muls_with_carrying_extended},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.add_vector_shuffle_instructions) {
    // - Vector shuffles create synonyms that can be applied
    // - TODO: extract from composites
    return RandomOrderAndNonNull({pass_instances.apply_id_synonyms},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.apply_id_synonyms) {
    // - No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.construct_composites) {
    // TODO: extract from composites,
    return RandomOrderAndNonNull({}, fuzzer_context);
  }
  if (completed_pass == pass_instances.copy_objects) {
    // - Object copies create synonyms that can be applied
    // - OpCopyObject can be replaced with a store/load pair
    return RandomOrderAndNonNull(
        {pass_instances.apply_id_synonyms,
         pass_instances.replace_copy_objects_with_stores_loads},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.donate_modules) {
    // - New functions in the module can be called
    return RandomOrderAndNonNull({pass_instances.add_function_calls},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.inline_functions) {
    // - Parts of inlined functions can be outlined again
    return RandomOrderAndNonNull({pass_instances.outline_functions},
                                 fuzzer_context);
  }
  if (completed_pass == pass_instances.invert_comparison_operators) {
    // - No obvious follow-on passes
    return {};
  }
  if (completed_pass == pass_instances.make_vector_operations_dynamic) {
    // - No obvious follow-on passes
    return {};
  }
  if (completed_pass == pass_instances.merge_blocks) {
    // - Having merged some blocks it may be interesting to split them in a
    //   different way
    return RandomOrderAndNonNull({pass_instances.split_blocks}, fuzzer_context);
  }
  if (completed_pass == pass_instances.obfuscate_constants) {
    // - No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.outline_functions) {
    // - This creates more functions, which can be called
    // - Inlining the function for the region that was outlined might also be
    //   fruitful; it will be inlined in a different form
    return RandomOrderAndNonNull(
        {pass_instances.add_function_calls, pass_instances.inline_functions},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.permute_blocks) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.permute_function_parameters) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.permute_instructions) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.propagate_instructions_up) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.push_ids_through_variables) {
    // - This pass creates synonyms, so it is worth applying them.
    return RandomOrderAndNonNull({pass_instances.apply_id_synonyms},
                                 fuzzer_context);
  }
  if (completed_pass ==
      pass_instances.replace_adds_subs_muls_with_carrying_extended) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass ==
      pass_instances.replace_copy_memories_with_loads_stores) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.replace_copy_objects_with_stores_loads) {
    // - We may end up with load/store pairs that could be used to create memory
    //   copies.
    return RandomOrderAndNonNull(
        {pass_instances.replace_loads_stores_with_copy_memories},
        fuzzer_context);
  }
  if (completed_pass ==
      pass_instances.replace_loads_stores_with_copy_memories) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.replace_parameter_with_global) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.replace_linear_algebra_instructions) {
    // No obvious follow-on passes.
    return {};
  }
  if (completed_pass == pass_instances.replace_params_with_struct) {
    // TODO: possibly some passes related to composites
    return {};
  }
  if (completed_pass == pass_instances.split_blocks) {
    // - More blocks means more chances for adding dead breaks/continues, and
    //   for adding dead blocks
    return RandomOrderAndNonNull(
        {pass_instances.add_dead_breaks, pass_instances.add_dead_continues,
         pass_instances.add_dead_blocks},
        fuzzer_context);
  }
  if (completed_pass == pass_instances.swap_branch_conditional_operands) {
    // No obvious follow-on passes.
    return {};
  }
  assert(false && "Unreachable: every fuzzer pass should be dealt with.");
}

std::vector<FuzzerPass*> Fuzzer::RandomOrderAndNonNull(
    const std::vector<FuzzerPass*>& passes,
    FuzzerContext* fuzzer_context) const {
  std::vector<uint32_t> indices(passes.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<FuzzerPass*> result;
  while (!indices.empty()) {
    FuzzerPass* maybe_pass =
        passes[fuzzer_context->RemoveAtRandomIndex(&indices)];
    if (maybe_pass) {
      result.push_back(maybe_pass);
    }
  }
  return result;
}

bool Fuzzer::ContinueFuzzing(
    const protobufs::TransformationSequence& transformation_sequence_out,
    FuzzerContext* fuzzer_context) const {
  auto transformations_applied_so_far =
      static_cast<uint32_t>(transformation_sequence_out.transformation_size());
  if (transformations_applied_so_far >= kTransformationLimit) {
    return false;
  }
  auto chance_of_continuing = static_cast<uint32_t>(
      100.0 * (1.0 - (static_cast<double>(transformations_applied_so_far) /
                      static_cast<double>(kTransformationLimit))));
  return fuzzer_context->ChoosePercentage(chance_of_continuing);
}

bool Fuzzer::ApplyFuzzerPassesSimple(
    const spvtools::SpirvTools& tools,
    const std::vector<std::unique_ptr<FuzzerPass>>& enabled_passes,
    FuzzerContext* fuzzer_context, opt::IRContext* ir_context,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  do {
    FuzzerPass* current_pass =
        enabled_passes[fuzzer_context->RandomIndex(enabled_passes)].get();
    if (!ApplyPassAndCheckValidity(current_pass, *ir_context, tools)) {
      return false;
    }
  } while (ContinueFuzzing(*transformation_sequence_out, fuzzer_context));
  return true;
}

bool Fuzzer::ApplyFuzzerPassesRandomlyWithRecommendations(
    const spvtools::SpirvTools& tools, const PassInstances& pass_instances,
    const std::vector<std::unique_ptr<FuzzerPass>>& enabled_passes,
    FuzzerContext* fuzzer_context, opt::IRContext* ir_context,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  std::deque<FuzzerPass*> recommended_passes;
  do {
    FuzzerPass* current_pass;
    if (recommended_passes.empty() || fuzzer_context->ChooseEven()) {
      current_pass =
          enabled_passes[fuzzer_context->RandomIndex(enabled_passes)].get();
    } else {
      current_pass = recommended_passes.front();
      recommended_passes.pop_front();
    }
    if (!ApplyPassAndCheckValidity(current_pass, *ir_context, tools)) {
      return false;
    }
    for (auto future_pass : GetRecommendedFuturePasses(
             current_pass, pass_instances, fuzzer_context)) {
      recommended_passes.push_back(future_pass);
    }
  } while (ContinueFuzzing(*transformation_sequence_out, fuzzer_context));
  return true;
}

bool Fuzzer::ApplyFuzzerPassesLoopedWithRecommendations(
    const spvtools::SpirvTools& tools, const PassInstances& pass_instances,
    const std::vector<std::unique_ptr<FuzzerPass>>& enabled_passes,
    FuzzerContext* fuzzer_context, opt::IRContext* ir_context,
    protobufs::TransformationSequence* transformation_sequence_out) const {
  std::vector<FuzzerPass*> pass_loop;
  do {
    FuzzerPass* current_pass =
        enabled_passes[fuzzer_context->RandomIndex(enabled_passes)].get();
    pass_loop.push_back(current_pass);
    for (auto future_pass : GetRecommendedFuturePasses(
             current_pass, pass_instances, fuzzer_context)) {
      pass_loop.push_back(future_pass);
    }
  } while (
      fuzzer_context->ChoosePercentage(kChanceOfAddingAnotherPassToPassLoop));

  for (uint32_t index = 0;;
       index = (index + 1) % static_cast<uint32_t>(pass_loop.size())) {
    if (!ApplyPassAndCheckValidity(pass_loop[index], *ir_context, tools)) {
      return false;
    }
    if (!ContinueFuzzing(*transformation_sequence_out, fuzzer_context)) {
      break;
    }
  }
  return true;
}

}  // namespace fuzz
}  // namespace spvtools
