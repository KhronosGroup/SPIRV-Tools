// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_PASS_H_
#define LIBSPIRV_OPT_PASS_H_

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "def_use_manager.h"
#include "module.h"
#include "spirv-tools/libspirv.hpp"
#include "basic_block.h"

#include "spirv-tools-opt_export.h"

namespace spvtools {
namespace opt {

// Abstract class of a pass. All passes should implement this abstract class
// and all analysis and transformation is done via the Process() method.
class Pass {
 public:
  // The status of processing a module using a pass.
  //
  // The numbers for the cases are assigned to make sure that Failure & anything
  // is Failure, SuccessWithChange & any success is SuccessWithChange.
  enum class Status {
    Failure = 0x00,
    SuccessWithChange = 0x10,
    SuccessWithoutChange = 0x11,
  };

  using ProcessFunction = std::function<bool(ir::Function*)>;

  // Constructs a new pass.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  SPIRV_TOOLS_OPT_EXPORT Pass();

  // Destructs the pass.
  SPIRV_TOOLS_OPT_EXPORT virtual ~Pass() = default;

  // Returns a descriptive name for this pass.
  //
  // NOTE: When deriving a new pass class, make sure you make the name
  // compatible with the corresponding spirv-opt command-line flag. For example,
  // if you add the flag --my-pass to spirv-opt, make this function return
  // "my-pass" (no leading hyphens).
  SPIRV_TOOLS_OPT_EXPORT virtual const char* name() const = 0;

  // Sets the message consumer to the given |consumer|. |consumer| which will be
  // invoked every time there is a message to be communicated to the outside.
  inline void SetMessageConsumer(MessageConsumer c) { consumer_ = std::move(c); }

  // Returns the reference to the message consumer for this pass.
  inline const MessageConsumer& consumer() const { return consumer_; }

  // Returns the def-use manager used for this pass. TODO(dnovillo): This should
  // be handled by the pass manager.
  inline analysis::DefUseManager* get_def_use_mgr() const {
    return def_use_mgr_.get();
  }

  // Returns a pointer to the current module for this pass.
  inline ir::Module* get_module() const { return module_; }

  // Add to |todo| all ids of functions called in |func|.
  SPIRV_TOOLS_OPT_EXPORT void AddCalls(ir::Function* func, std::queue<uint32_t>* todo);

  // Applies |pfn| to every function in the call trees that are rooted at the
  // entry points.  Returns true if any call |pfn| returns true.  By convention
  // |pfn| should return true if it modified the module.
  SPIRV_TOOLS_OPT_EXPORT bool ProcessEntryPointCallTree(ProcessFunction& pfn, ir::Module* module);

  // Applies |pfn| to every function in the call trees rooted at the entry
  // points and exported functions.  Returns true if any call |pfn| returns
  // true.  By convention |pfn| should return true if it modified the module.
  SPIRV_TOOLS_OPT_EXPORT bool ProcessReachableCallTree(ProcessFunction& pfn, ir::Module* module);

  // Applies |pfn| to every function in the call trees rooted at the elements of
  // |roots|.  Returns true if any call to |pfn| returns true.  By convention
  // |pfn| should return true if it modified the module.  After returning
  // |roots| will be empty.
  SPIRV_TOOLS_OPT_EXPORT bool ProcessCallTreeFromRoots(
      ProcessFunction& pfn,
      const std::unordered_map<uint32_t, ir::Function*>& id2function,
      std::queue<uint32_t>* roots);

  // Processes the given |module|. Returns Status::Failure if errors occur when
  // processing. Returns the corresponding Status::Success if processing is
  // succesful to indicate whether changes are made to the module.
  SPIRV_TOOLS_OPT_EXPORT virtual Status Process(ir::Module* module) = 0;

 protected:
  // Initialize basic data structures for the pass. This sets up the def-use
  // manager, module and other attributes. TODO(dnovillo): Some of this should
  // be done during pass instantiation. Other things should be outside the pass
  // altogether (e.g., def-use manager).
  inline
  void InitializeProcessing(ir::Module* module) {
    module_ = module;
    next_id_ = module_->IdBound();
    def_use_mgr_.reset(new analysis::DefUseManager(consumer(), get_module()));
    block2structured_succs_.clear();
    label2preds_.clear();
    id2block_.clear();
    for (auto& fn : *module_) {
      for (auto& blk : fn) {
        id2block_[blk.id()] = &blk;
      }
    }
  }

  // Return true if |block_ptr| points to a loop header block. TODO(dnovillo)
  // This belongs in a CFG class.
  SPIRV_TOOLS_OPT_EXPORT bool IsLoopHeader(ir::BasicBlock* block_ptr) const;

  // Compute structured successors for function |func|. A block's structured
  // successors are the blocks it branches to together with its declared merge
  // block if it has one. When order matters, the merge block always appears
  // first. This assures correct depth first search in the presence of early
  // returns and kills. If the successor vector contain duplicates if the merge
  // block, they are safely ignored by DFS. TODO(dnovillo): This belongs in a
  // CFG class.
  SPIRV_TOOLS_OPT_EXPORT void ComputeStructuredSuccessors(ir::Function* func);

  // Compute structured block order for |func| into |structuredOrder|. This
  // order has the property that dominators come before all blocks they
  // dominate and merge blocks come after all blocks that are in the control
  // constructs of their header. TODO(dnovillo): This belongs in a CFG class.
  SPIRV_TOOLS_OPT_EXPORT void ComputeStructuredOrder(ir::Function* func,
                              std::list<ir::BasicBlock*>* order);

  // Return type id for |ptrInst|'s pointee
  SPIRV_TOOLS_OPT_EXPORT uint32_t GetPointeeTypeId(const ir::Instruction* ptrInst) const;

  // Return the next available Id and increment it.
  inline uint32_t TakeNextId() { return next_id_++; }

  // Write the next available Id back to the module.
  inline void FinalizeNextId() {
    assert(module_);
    module_->SetIdBound(next_id_);
  }

  // Returns the id of the merge block declared by a merge instruction in this
  // block, if any.  If none, returns zero.
  SPIRV_TOOLS_OPT_EXPORT uint32_t MergeBlockIdIfAny(const ir::BasicBlock& blk, uint32_t* cbid);

  // Map from block to its structured successor blocks. See
  // ComputeStructuredSuccessors() for definition.
  std::unordered_map<const ir::BasicBlock*, std::vector<ir::BasicBlock*>>
      block2structured_succs_;

  // Extra block whose successors are all blocks with no predecessors
  // in function.
  ir::BasicBlock pseudo_entry_block_;

  // Augmented CFG Exit Block.
  ir::BasicBlock pseudo_exit_block_;

  // Map from block's label id to its predecessor blocks ids
  std::unordered_map<uint32_t, std::vector<uint32_t>> label2preds_;

  // Map from block's label id to block.
  std::unordered_map<uint32_t, ir::BasicBlock*> id2block_;

 private:
  using cbb_ptr = const ir::BasicBlock*;

  MessageConsumer consumer_;  // Message consumer.

  // Def-Uses for the module we are processing
  std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Next unused ID
  uint32_t next_id_;

  // The module that the pass is being applied to.
  ir::Module* module_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_H_
