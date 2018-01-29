// Copyright (c) 2017 Google Inc.
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

#ifndef LIBSPIRV_OPT_LOOP_DESCRIPTORS_H_
#define LIBSPIRV_OPT_LOOP_DESCRIPTORS_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "opt/basic_block.h"
#include "opt/tree_iterator.h"

namespace spvtools {
namespace opt {
class DominatorAnalysis;
struct DominatorTreeNode;
}  // namespace opt
namespace ir {
class IRContext;
class CFG;
class LoopDescriptor;

// A class to represent and manipulate a loop in structured control flow.
class Loop {
  // The type used to represent nested child loops.
  using ChildrenList = std::vector<Loop*>;

 public:
  using iterator = ChildrenList::iterator;
  using const_iterator = ChildrenList::const_iterator;
  using BasicBlockListTy = std::unordered_set<uint32_t>;

  Loop()
      : context_(nullptr),
        loop_header_(nullptr),
        loop_continue_(nullptr),
        loop_merge_(nullptr),
        loop_preheader_(nullptr),
        parent_(nullptr) {}

  Loop(IRContext* context, opt::DominatorAnalysis* analysis, BasicBlock* header,
       BasicBlock* continue_target, BasicBlock* merge_target);

  // Iterators over the immediate sub-loops.
  inline iterator begin() { return nested_loops_.begin(); }
  inline iterator end() { return nested_loops_.end(); }
  inline const_iterator begin() const { return cbegin(); }
  inline const_iterator end() const { return cend(); }
  inline const_iterator cbegin() const { return nested_loops_.begin(); }
  inline const_iterator cend() const { return nested_loops_.end(); }

  // Returns the header (first basic block of the loop). This block contains the
  // OpLoopMerge instruction.
  inline BasicBlock* GetHeaderBlock() { return loop_header_; }
  inline const BasicBlock* GetHeaderBlock() const { return loop_header_; }
  inline void SetHeaderBlock(BasicBlock* header) { loop_header_ = header; }

  // Updates the OpLoopMerge instruction to reflect the current state of the
  // loop.
  inline void UpdateLoopMergeInst() {
    assert(GetHeaderBlock()->GetLoopMergeInst() &&
           "The loop is not structured");
    ir::Instruction* merge_inst = GetHeaderBlock()->GetLoopMergeInst();
    merge_inst->SetInOperand(0, {GetMergeBlock()->id()});
  }

  // Returns the latch basic block (basic block that holds the back-edge).
  // These functions return nullptr if the loop is not structured (i.e. if it
  // has more than one backedge).
  inline BasicBlock* GetLatchBlock() { return loop_continue_; }
  inline const BasicBlock* GetLatchBlock() const { return loop_continue_; }
  // Sets |latch| as the loop unique block branching back to the header.
  // A latch block must have the following properties:
  //  - |latch| must be in the loop;
  //  - must be the only block branching back to the header block.
  void SetLatchBlock(BasicBlock* latch);

  // Returns the basic block which marks the end of the loop.
  // These functions return nullptr if the loop is not structured.
  inline BasicBlock* GetMergeBlock() { return loop_merge_; }
  inline const BasicBlock* GetMergeBlock() const { return loop_merge_; }
  // Sets |merge| as the loop merge block. A merge block must have the following
  // properties:
  //  - |merge| must not be in the loop;
  //  - all its predecessors must be in the loop.
  //  - it must not be already used as merge block.
  // If the loop has an OpLoopMerge in its header, this instruction is also
  // updated.
  void SetMergeBlock(BasicBlock* merge);

  // Returns the loop pre-header, nullptr means that the loop predecessor does
  // not qualify as a preheader.
  // The preheader is the unique predecessor that:
  //   - Dominates the loop header;
  //   - Has only the loop header as successor.
  inline BasicBlock* GetPreHeaderBlock() { return loop_preheader_; }

  // Returns the loop pre-header.
  inline const BasicBlock* GetPreHeaderBlock() const { return loop_preheader_; }

  // Returns the loop pre-header, if there is no suitable preheader it will be
  // created.
  BasicBlock* GetOrCreatePreHeaderBlock();

  // Returns true if this loop contains any nested loops.
  inline bool HasNestedLoops() const { return nested_loops_.size() != 0; }

  // Clears and fills |exit_blocks| with all basic blocks that are not in the
  // loop and has at least one predecessor in the loop.
  void GetExitBlocks(std::unordered_set<uint32_t>* exit_blocks) const;

  // Clears and fills |merging_blocks| with all basic blocks that are
  // post-dominated by the merge block. The merge block must exist.
  // The set |merging_blocks| will only contain the merge block if it is
  // unreachable.
  void GetMergingBlocks(std::unordered_set<uint32_t>* merging_blocks) const;

  // Returns true if the loop is in a Loop Closed SSA form.
  // In LCSSA form, all in-loop definitions are used in the loop or in phi
  // instructions in the loop exit blocks.
  bool IsLCSSA() const;

  // Returns the depth of this loop in the loop nest.
  // The outer-most loop has a depth of 1.
  inline size_t GetDepth() const {
    size_t lvl = 1;
    for (const Loop* loop = GetParent(); loop; loop = loop->GetParent()) lvl++;
    return lvl;
  }

  // Adds |nested| as a nested loop of this loop. Automatically register |this|
  // as the parent of |nested|.
  inline void AddNestedLoop(Loop* nested) {
    assert(!nested->GetParent() && "The loop has another parent.");
    nested_loops_.push_back(nested);
    nested->SetParent(this);
  }

  inline Loop* GetParent() { return parent_; }
  inline const Loop* GetParent() const { return parent_; }

  inline bool HasParent() const { return parent_; }

  // Returns true if this loop is itself nested within another loop.
  inline bool IsNested() const { return parent_ != nullptr; }

  // Returns the set of all basic blocks contained within the loop. Will be all
  // BasicBlocks dominated by the header which are not also dominated by the
  // loop merge block.
  inline const BasicBlockListTy& GetBlocks() const {
    return loop_basic_blocks_;
  }

  // Returns true if the basic block |bb| is inside this loop.
  inline bool IsInsideLoop(const BasicBlock* bb) const {
    return IsInsideLoop(bb->id());
  }

  // Returns true if the basic block id |bb_id| is inside this loop.
  inline bool IsInsideLoop(uint32_t bb_id) const {
    return loop_basic_blocks_.count(bb_id);
  }

  // Returns true if the instruction |inst| is inside this loop.
  bool IsInsideLoop(Instruction* inst) const;

  // Adds the Basic Block |bb| this loop and its parents.
  void AddBasicBlockToLoop(const BasicBlock* bb) {
    assert(IsBasicBlockInLoopSlow(bb) &&
           "Basic block does not belong to the loop");

    AddBasicBlock(bb);
  }

  // Adds the Basic Block |bb| this loop and its parents.
  void AddBasicBlock(const BasicBlock* bb) {
    for (Loop* loop = this; loop != nullptr; loop = loop->parent_) {
      loop_basic_blocks_.insert(bb->id());
    }
  }

  // Sets the parent loop of this loop, that is, a loop which contains this loop
  // as a nested child loop.
  inline void SetParent(Loop* parent) { parent_ = parent; }

  // Returns true is the instruction is invariant and safe to move wrt loop
  bool ShouldHoistInstruction(IRContext* context, Instruction* inst);

  // Returns true if all operands of inst are in basic blocks not contained in
  // loop
  bool AreAllOperandsOutsideLoop(IRContext* context, Instruction* inst);

 private:
  IRContext* context_;
  // The block which marks the start of the loop.
  BasicBlock* loop_header_;

  // The block which begins the body of the loop.
  BasicBlock* loop_continue_;

  // The block which marks the end of the loop.
  BasicBlock* loop_merge_;

  // The block immediately before the loop header.
  BasicBlock* loop_preheader_;

  // A parent of a loop is the loop which contains it as a nested child loop.
  Loop* parent_;

  // Nested child loops of this loop.
  ChildrenList nested_loops_;

  // A set of all the basic blocks which comprise the loop structure. Will be
  // computed only when needed on demand.
  BasicBlockListTy loop_basic_blocks_;

  // Check that |bb| is inside the loop using domination property.
  // Note: this is for assertion purposes only, IsInsideLoop should be used
  // instead.
  bool IsBasicBlockInLoopSlow(const BasicBlock* bb);

  // Returns the loop preheader if it exists, returns nullptr otherwise.
  BasicBlock* FindLoopPreheader(opt::DominatorAnalysis* dom_analysis);

  // Sets |latch| as the loop unique continue block. No checks are performed
  // here.
  inline void SetLatchBlockImpl(BasicBlock* latch) { loop_continue_ = latch; }
  // Sets |merge| as the loop merge block. No checks are performed here.
  inline void SetMergeBlockImpl(BasicBlock* merge) { loop_merge_ = merge; }

  // This is only to allow LoopDescriptor::dummy_top_loop_ to add top level
  // loops as child.
  friend class LoopDescriptor;
  friend class LoopUtils;
};

// Loop descriptions class for a given function.
// For a given function, the class builds loop nests information.
// The analysis expects a structured control flow.
class LoopDescriptor {
 public:
  // Iterator interface (depth first postorder traversal).
  using iterator = opt::PostOrderTreeDFIterator<Loop>;
  using const_iterator = opt::PostOrderTreeDFIterator<const Loop>;

  // Creates a loop object for all loops found in |f|.
  explicit LoopDescriptor(const Function* f);

  // Disable copy constructor, to avoid double-free on destruction.
  LoopDescriptor(const LoopDescriptor&) = delete;
  // Move constructor.
  LoopDescriptor(LoopDescriptor&& other) {
    // We need to take ownership of the Loop objects in the other
    // LoopDescriptor, to avoid double-free.
    loops_ = std::move(other.loops_);
    other.loops_.clear();
    basic_block_to_loop_ = std::move(other.basic_block_to_loop_);
    other.basic_block_to_loop_.clear();
    dummy_top_loop_ = std::move(other.dummy_top_loop_);
  }

  // Destructor
  ~LoopDescriptor();

  // Returns the number of loops found in the function.
  inline size_t NumLoops() const { return loops_.size(); }

  // Returns the loop at a particular |index|. The |index| must be in bounds,
  // check with NumLoops before calling.
  inline Loop& GetLoopByIndex(size_t index) const {
    assert(loops_.size() > index &&
           "Index out of range (larger than loop count)");
    return *loops_[index];
  }

  // Returns the inner most loop that contains the basic block id |block_id|.
  inline Loop* operator[](uint32_t block_id) const {
    return FindLoopForBasicBlock(block_id);
  }

  // Returns the inner most loop that contains the basic block |bb|.
  inline Loop* operator[](const BasicBlock* bb) const {
    return (*this)[bb->id()];
  }

  // Iterators for post order depth first traversal of the loops.
  // Inner most loops will be visited first.
  inline iterator begin() { return iterator::begin(&dummy_top_loop_); }
  inline iterator end() { return iterator::end(&dummy_top_loop_); }
  inline const_iterator begin() const { return cbegin(); }
  inline const_iterator end() const { return cend(); }
  inline const_iterator cbegin() const {
    return const_iterator::begin(&dummy_top_loop_);
  }
  inline const_iterator cend() const {
    return const_iterator::end(&dummy_top_loop_);
  }

  // Returns the inner most loop that contains the basic block |bb|.
  inline void SetBasicBlockToLoop(uint32_t bb_id, Loop* loop) {
    basic_block_to_loop_[bb_id] = loop;
  }

 private:
  // TODO(dneto): This should be a vector of unique_ptr.  But VisualStudio 2013
  // is unable to compile it.
  using LoopContainerType = std::vector<Loop*>;

  // Creates loop descriptors for the function |f|.
  void PopulateList(const Function* f);

  // Returns the inner most loop that contains the basic block id |block_id|.
  inline Loop* FindLoopForBasicBlock(uint32_t block_id) const {
    std::unordered_map<uint32_t, Loop*>::const_iterator it =
        basic_block_to_loop_.find(block_id);
    return it != basic_block_to_loop_.end() ? it->second : nullptr;
  }

  // Erase all the loop information.
  void ClearLoops();

  // A list of all the loops in the function.  This variable owns the Loop
  // objects.
  LoopContainerType loops_;
  // Dummy root: this "loop" is only there to help iterators creation.
  Loop dummy_top_loop_;
  std::unordered_map<uint32_t, Loop*> basic_block_to_loop_;
};

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOOP_DESCRIPTORS_H_
