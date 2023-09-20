// Copyright (c) 2023 Google Inc.
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

#ifndef SOURCE_OPT_DEDUPE_INTERLOCK_INVOCATION_PASS_H_
#define SOURCE_OPT_DEDUPE_INTERLOCK_INVOCATION_PASS_H_

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "source/enum_set.h"
#include "source/extensions.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/spirv_target_env.h"

namespace spvtools {
namespace opt {

// This pass will ensure that an entry point will only have at most one
// OpBeginInterlockInvocationEXT and one OpEndInterlockInvocationEXT, in that
// order
class InvocationInterlockPlacementPass : public Pass {
 public:
  InvocationInterlockPlacementPass() {}
  InvocationInterlockPlacementPass(const InvocationInterlockPlacementPass&) =
      delete;
  InvocationInterlockPlacementPass(InvocationInterlockPlacementPass&&) = delete;

  const char* name() const override { return "dedupe-interlock-invocation"; }
  Status Process() override;

 private:
  using BlockSet = std::unordered_set<uint32_t>;

  // Specifies whether a function originally had a begin or end instruction.
  struct ExtractionResult {
    bool had_begin : 1;
    bool had_end : 2;
  };

  // Check if a block has only a single next block, depending on the directing
  // that we are traversing the CFG. If forward_flow is true, we are walking
  // forward through the CFG, and will return if the block has only one
  // successor. Otherwise, we are walking backward through the CFG, and will
  // return if the block has only one predecessor.
  bool hasSingleNextBlock(uint32_t block_id, bool forward_flow);

  // Iterate over each of a block's predecessors or successors, depending on
  // direction. If forward_flow is true, we are walking forward through the CFG,
  // and need to iterate over the successors. Otherwise, we are walking backward
  // through the CFG, and need to iterate over the predecessors.
  void forEachNext(uint32_t block_id, bool forward_flow,
                   std::function<void(uint32_t)> f);

  // Add either a begin or end instruction to the edge of the basic block. If
  // forward_flow is true, we are walking forward through the CFG, and want to
  // add a begin instruction to the end of the block. Otherwise, we are walking
  // backward through the CFG, and want to add an end instruction to the
  // beginning of the basic block.
  void addInstructionAtEdge(BasicBlock* block, bool forward_flow);

  // Populates begin_ with the set of blocks that
  // contain OpBeginInvocationInterlockEXT, and end_ with the set of blocks that
  // contain OpEndInvocationInterlockEXT
  // void computeBeginAndEnd();

  // Remove every OpBeginInvocationInterlockEXT instruction in block after the
  // first. Returns whether any instructions were removed.
  bool killDuplicateBegin(BasicBlock* block);
  // Remove every OpBeginInvocationInterlockEXT instruction in block before the
  // last. Returns whether any instructions were removed.
  bool killDuplicateEnd(BasicBlock* block);

  // Recursively removes any begin or end instructions from func and any
  // function func calls. Returns whether any begin or end instructions were
  // removed.
  ExtractionResult removeInstructionsFromFunction(Function* func);

  // For every function call in any of the passed blocks, move any begin or end
  // instructions outside of the function call. Returns whether any extractions
  // occurred.
  bool extractInstructionsFromCalls(std::vector<BasicBlock*> blocks);

  // Finds the sets of blocks begin_ and end_ that contain
  // OpBeginInvocationInterlockEXT and OpEndInvocationInterlockEXT,
  // respectively.
  void computeBeginAndEnd(std::vector<BasicBlock*> blocks);

  // Compute the set of blocks including or after the barrier instruction, and
  // the set of blocks with any previous blocks inside the barrier instruction.
  // If forward_flow is true, move forward through the CFG, computing
  // after_begin_ and predecessors_after_begin_computing after_begin_ and
  // predecessors_after_begin_, otherwise, move backward through the CFG,
  // computing before_end_ and successors_before_end_.
  void computeReachableBlocks(BlockSet& out_set, BlockSet& in_set,
                              bool forwardFlow);

  // Remove unneeded begin and end instructions in block.
  bool removeUnneededInstructions(BasicBlock* block);

  // Given a block which branches to multiple successors, and a specific
  // successor, creates a new empty block, and update the branch instruction to
  // branch to the new block instead.
  BasicBlock* splitEdge(BasicBlock* block, uint32_t succ_id);

  // For the edge from block to next_id, places a begin or end instruction on
  // the edge, based on the direction we are walking the CFG, specified in
  // forward_flow.
  bool placeInstructionsForEdge(BasicBlock* block, uint32_t next_id,
                                bool forward_flow);
  // Calls placeInstructionsForEdge for each edge in block.
  bool placeInstructions(BasicBlock* block);

  // Processes a single fragment shader entry function.
  bool processFragmentShaderEntry(Function* entry_func);

  // Maps a function to whether that function originally held a begin or end
  // instruction.
  std::unordered_map<Function*, ExtractionResult> extracted_functions_;

  // The set of blocks which have an OpBeginInvocationInterlockEXT instruction.
  BlockSet begin_;
  // The set of blocks which have an OpEndInvocationInterlockEXT instruction.
  BlockSet end_;
  // The set of blocks which either have a begin instruction, or have a
  // predecessor which has a begin instruction.
  BlockSet after_begin_;
  // The set of blocks which either have an end instruction, or have a successor
  // which have an end instruction.
  BlockSet before_end_;
  // The set of blocks which have a predecessor in after_begin_.
  BlockSet predecessors_after_begin_;
  // The set of blocks which have a successor in before_end_.
  BlockSet successors_before_end_;
};

}  // namespace opt
}  // namespace spvtools
#endif  // SOURCE_OPT_DEDUPE_INTERLOCK_INVOCATION_PASS_H_
