// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_PASS_INSTANCES_
#define SOURCE_FUZZ_PASS_INSTANCES_

#include "source/fuzz/fuzzer_pass_add_access_chains.h"
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
#include "source/fuzz/fuzzer_pass_add_opphi_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_parameters.h"
#include "source/fuzz/fuzzer_pass_add_relaxed_decorations.h"
#include "source/fuzz/fuzzer_pass_add_stores.h"
#include "source/fuzz/fuzzer_pass_add_synonyms.h"
#include "source/fuzz/fuzzer_pass_add_vector_shuffle_instructions.h"
#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"
#include "source/fuzz/fuzzer_pass_construct_composites.h"
#include "source/fuzz/fuzzer_pass_copy_objects.h"
#include "source/fuzz/fuzzer_pass_donate_modules.h"
#include "source/fuzz/fuzzer_pass_inline_functions.h"
#include "source/fuzz/fuzzer_pass_invert_comparison_operators.h"
#include "source/fuzz/fuzzer_pass_make_vector_operations_dynamic.h"
#include "source/fuzz/fuzzer_pass_merge_blocks.h"
#include "source/fuzz/fuzzer_pass_obfuscate_constants.h"
#include "source/fuzz/fuzzer_pass_outline_functions.h"
#include "source/fuzz/fuzzer_pass_permute_blocks.h"
#include "source/fuzz/fuzzer_pass_permute_function_parameters.h"
#include "source/fuzz/fuzzer_pass_permute_instructions.h"
#include "source/fuzz/fuzzer_pass_propagate_instructions_up.h"
#include "source/fuzz/fuzzer_pass_push_ids_through_variables.h"
#include "source/fuzz/fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h"
#include "source/fuzz/fuzzer_pass_replace_copy_memories_with_loads_stores.h"
#include "source/fuzz/fuzzer_pass_replace_copy_objects_with_stores_loads.h"
#include "source/fuzz/fuzzer_pass_replace_linear_algebra_instructions.h"
#include "source/fuzz/fuzzer_pass_replace_loads_stores_with_copy_memories.h"
#include "source/fuzz/fuzzer_pass_replace_parameter_with_global.h"
#include "source/fuzz/fuzzer_pass_replace_params_with_struct.h"
#include "source/fuzz/fuzzer_pass_split_blocks.h"
#include "source/fuzz/fuzzer_pass_swap_conditional_branch_operands.h"

namespace spvtools {
namespace fuzz {

// This struct should have one field for every distinct fuzzer pass that can be
// applied repeatedly.  Fuzzer passes that are only intended to be run at most
// once, at the end of fuzzing, are not included.
struct PassInstances {
#define PASS_INSTANCE(NAME) \
  public:                   \
    FuzzerPass##NAME* Get##NAME() { \
      return NAME##_; \
    }                       \
    void SetPass(std::unique_ptr<FuzzerPass##NAME> pass) {   \
      assert(NAME##_ == nullptr && "Attempt to set pass multiple times."); \
      NAME##_ = pass.get(); \
      passes_.push_back(std::move(pass)); \
    }                       \
  private:                  \
    FuzzerPass##NAME* NAME##_

  PASS_INSTANCE(AddAccessChains);
  PASS_INSTANCE(AddCompositeInserts);
  PASS_INSTANCE(AddCompositeTypes);
  PASS_INSTANCE(AddCopyMemory);
  PASS_INSTANCE(AddDeadBlocks);
  PASS_INSTANCE(AddDeadBreaks);
  PASS_INSTANCE(AddDeadContinues);
  PASS_INSTANCE(AddEquationInstructions);
  PASS_INSTANCE(AddFunctionCalls);
  PASS_INSTANCE(AddGlobalVariables);
  PASS_INSTANCE(AddImageSampleUnusedComponents);
  PASS_INSTANCE(AddLoads);
  PASS_INSTANCE(AddLocalVariables);
  PASS_INSTANCE(AddLoopPreheaders);
  PASS_INSTANCE(AddOpPhiSynonyms);
  PASS_INSTANCE(AddParameters);
  PASS_INSTANCE(AddRelaxedDecorations);
  PASS_INSTANCE(AddStores);
  PASS_INSTANCE(AddSynonyms);
  PASS_INSTANCE(AddVectorShuffleInstructions);
  PASS_INSTANCE(ApplyIdSynonyms);
  PASS_INSTANCE(ConstructComposites);
  PASS_INSTANCE(CopyObjects);
  PASS_INSTANCE(DonateModules);
  PASS_INSTANCE(InlineFunctions);
  PASS_INSTANCE(InvertComparisonOperators);
  PASS_INSTANCE(MakeVectorOperationsDynamic);
  PASS_INSTANCE(MergeBlocks);
  PASS_INSTANCE(ObfuscateConstants);
  PASS_INSTANCE(OutlineFunctions);
  PASS_INSTANCE(PermuteBlocks);
  PASS_INSTANCE(PermuteFunctionParameters);
  PASS_INSTANCE(PermuteInstructions);
  PASS_INSTANCE(PropagateInstructionsUp);
  PASS_INSTANCE(PushIdsThroughVariables);
  PASS_INSTANCE(ReplaceAddsSubsMulsWithCarryingExtended);
  PASS_INSTANCE(ReplaceCopyMemoriesWithLoadsStores);
  PASS_INSTANCE(ReplaceCopyObjectsWithStoresLoads);
  PASS_INSTANCE(ReplaceLoadsStoresWithCopyMemories);
  PASS_INSTANCE(ReplaceParameterWithGlobal);
  PASS_INSTANCE(ReplaceLinearAlgebraInstructions);
  PASS_INSTANCE(ReplaceParamsWithStruct);
  PASS_INSTANCE(SplitBlocks);
  PASS_INSTANCE(SwapBranchConditionalOperands);

 public:
  const std::vector<std::unique_ptr<FuzzerPass>>& GetPasses() const {
    return passes_;
  }

 private:
  std::vector<std::unique_ptr<FuzzerPass>> passes_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_PASS_RECOMMENDER_
