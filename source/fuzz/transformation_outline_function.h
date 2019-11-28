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

#ifndef SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_

#include <map>
#include <set>
#include <vector>

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationOutlineFunction : public Transformation {
 public:
  explicit TransformationOutlineFunction(
      const protobufs::TransformationOutlineFunction& message);

  TransformationOutlineFunction(
      uint32_t entry_block, uint32_t exit_block,
      uint32_t new_function_struct_return_type_id,
      uint32_t new_function_type_id, uint32_t new_function_id,
      uint32_t new_function_first_block,
      uint32_t new_function_region_entry_block, uint32_t new_caller_result_id,
      uint32_t new_callee_result_id,
      std::map<uint32_t, uint32_t>&& input_id_to_fresh_id,
      std::map<uint32_t, uint32_t>&& output_id_to_fresh_id);

  // - All the fresh ids occurring in the transformation must be distinct and
  //   fresh.
  // - |message_.entry_block| and |message_.exit_block| must form a single-entry
  //   single-exit control flow graph region.
  // - |message_.entry_block| must not start with OpVariable
  // - A structured control flow construct must lie either completely within the
  //   region or completely outside it, with the exception that
  //   |message_.exit_block| can be a merge or continue block even if the
  //   associated header is not in the region.
  // - If |message_.entry_block| is a loop header, it must not contain OpPhi
  //   instructions.
  // - |message_.input_id_to_fresh_id| must contain an entry for every id
  //   defined outside the region but used in the region.
  // - |message_.output_id_to_fresh_id| must contain an entry for every id
  //   defined in the region but used outside the region.
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // - A new function with id |message_.new_function_id| is added to the module.
  // - If the region generates output ids, the return type of this function is
  //   a new struct type with one field per output id, and with type id
  //   |message_.new_function_struct_return_type|, otherwise the function return
  //   types is void and |message_.new_function_struct_return_type| is not used.
  // - If the region generates input ids, the new function has one parameter per
  //   input id.  Fresh ids for these parameters are provided by
  //   |message_.input_id_to_fresh_id|.
  // - Unless the type required for the new function is already known,
  //   |message_.new_function_type_id| is used as the type id for a new function
  //   type, and the new function uses this type.
  // - The new function starts with a dummy block with id
  //   |message_.new_function_first_block|, which jumps straight to a successor
  //   block, to avoid violating rules on what the first block in a function may
  //   look like.
  // - The outlined region is replaced with a single block, with the same id
  //   as |message_.entry_block|, and which calls the new function, passing the
  //   region's input ids as parameters.  The result is  stored in
  //   |message_.new_caller_result_id|, which has type
  //   |message_.new_function_struct_return_type| (unless there are
  //   no output ids, in which case the return type is void).  The components
  //   of this returned struct are then copied out into the region's output ids.
  //   The block ends with the merge instruction (if any) and terminator of
  //   |message_.exit_block|.
  // - The body of the new function is identical to the outlined region, except
  //   that (a) the region's entry block has id
  //   |message_.new_function_region_entry_block|, (b) input id uses are
  //   replaced with parameter accesses, (c) and definitions of output ids are
  //   replaced with definitions of corresponding fresh ids provided by
  //   |message_.output_id_to_fresh_id|, and (d) the block of the function
  //   ends by returning a composite of type
  //   |message_.new_function_struct_return_type| comprised of all the fresh
  //   output ids (unless the return type is void, in which case no value is
  //   returned.
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

  // TODO comment
  static std::set<opt::BasicBlock*> GetRegionBlocks(
      opt::IRContext* context, opt::BasicBlock* entry_block,
      opt::BasicBlock* exit_block);

  // TODO comment
  static std::vector<uint32_t> GetRegionInputIds(
      opt::IRContext* context, const std::set<opt::BasicBlock*>& region_set,
      opt::BasicBlock* region_entry_block, opt::BasicBlock* region_exit_block);

  // TODO comment
  static std::vector<uint32_t> GetRegionOutputIds(
      opt::IRContext* context, const std::set<opt::BasicBlock*>& region_set,
      opt::BasicBlock* region_entry_block, opt::BasicBlock* region_exit_block);

 private:
  // A helper method for the applicability check.  Returns true if and only if
  // |id| is (a) a fresh id for the module, and (b) an id that has not
  // previously been subject to this check.  We use this to check whether the
  // ids given for the transformation are not only fresh but also different from
  // one another.
  bool CheckIdIsFreshAndNotUsedByThisTransformation(
      uint32_t id, opt::IRContext* context,
      std::set<uint32_t>* ids_used_by_this_transformation) const;

  // TODO comment
  void UpdateModuleIdBoundForFreshIds(
      opt::IRContext* context,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const;

  // TODO comment
  void RemapInputAndOutputIdsInRegion(
      opt::IRContext* context,
      const opt::BasicBlock& original_region_entry_block,
      const opt::BasicBlock& original_region_exit_block,
      const std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map) const;

  // TODO comment
  std::unique_ptr<opt::Function> PrepareFunctionPrototype(
      opt::IRContext* context, const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& input_id_to_fresh_id_map,
      const std::map<uint32_t, uint32_t>& output_id_to_type_id) const;

  void PopulateOutlinedFunction(
      opt::IRContext* context,
      const opt::BasicBlock& original_region_entry_block,
      const opt::BasicBlock& original_region_exit_block,
      const std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& output_id_to_fresh_id_map,
      opt::Function* outlined_function) const;

  void ContractOriginalRegion(
      opt::IRContext* context, std::set<opt::BasicBlock*>& region_blocks,
      const std::vector<uint32_t>& region_input_ids,
      const std::vector<uint32_t>& region_output_ids,
      const std::map<uint32_t, uint32_t>& output_id_to_type_id,
      uint32_t return_type_id,
      std::unique_ptr<opt::Instruction> cloned_exit_block_merge,
      std::unique_ptr<opt::Instruction> cloned_exit_block_terminator,
      opt::BasicBlock* original_region_entry_block) const;

  protobufs::TransformationOutlineFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_OUTLINE_FUNCTION_H_
