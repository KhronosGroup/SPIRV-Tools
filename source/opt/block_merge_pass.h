// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef SOURCE_OPT_BLOCK_MERGE_PASS_H_
#define SOURCE_OPT_BLOCK_MERGE_PASS_H_

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class BlockMergePass : public Pass {
 public:
  BlockMergePass();

  Status Process(opt::IRContext*) override;
  virtual opt::IRContext::Analysis GetPreservedAnalyses() override {
    return opt::IRContext::kAnalysisDefUse |
           opt::IRContext::kAnalysisInstrToBlockMapping |
           opt::IRContext::kAnalysisDecorations |
           opt::IRContext::kAnalysisCombinators |
           opt::IRContext::kAnalysisNameMap;
  }

 private:
  // Kill any OpName instruction referencing |inst|, then kill |inst|.
  void KillInstAndName(opt::Instruction* inst);

  // Search |func| for blocks which have a single Branch to a block
  // with no other predecessors. Merge these blocks into a single block.
  bool MergeBlocks(opt::Function* func);

  // Returns true if |block| (or |id|) contains a merge instruction.
  bool IsHeader(opt::BasicBlock* block);
  bool IsHeader(uint32_t id);

  // Returns true if |block| (or |id|) is the merge target of a merge
  // instruction.
  bool IsMerge(opt::BasicBlock* block);
  bool IsMerge(uint32_t id);

  void Initialize(opt::IRContext* c);
  Pass::Status ProcessImpl();
};

class BlockMergePassToken : public PassToken {
 public:
  BlockMergePassToken() = default;
  ~BlockMergePassToken() override = default;

  const char* name() const override { return "merge-blocks"; }

  std::unique_ptr<Pass> CreatePass() const override {
    return MakeUnique<BlockMergePass>();
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_BLOCK_MERGE_PASS_H_
