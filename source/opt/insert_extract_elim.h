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

#ifndef LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_
#define LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "ir_context.h"
#include "mem_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InsertExtractElimPass : public MemPass {
 public:
  InsertExtractElimPass();
  const char* name() const override { return "eliminate-insert-extract"; }
  Status Process(ir::IRContext*) override;

 private:
  // Return true if the extract indices in |extIndices| starting at |extOffset|
  // match indices of insert |insInst|.
  bool ExtInsMatch(const std::vector<uint32_t>& extIndices,
                   const ir::Instruction* insInst,
                   const uint32_t extOffset) const;

  // Return true if indices in |extIndices| starting at |extOffset| and
  // indices of insert |insInst| conflict, specifically, if the insert
  // changes bits specified by the extract, but changes either more bits
  // or less bits than the extract specifies, meaning the exact value being
  // inserted cannot be used to replace the extract.
  bool ExtInsConflict(const std::vector<uint32_t>& extIndices,
                      const ir::Instruction* insInst,
                      const uint32_t extOffset) const;

  // Return true if |typeId| is a vector type
  bool IsVectorType(uint32_t typeId);

  // Return true if |typeId| is composite.
  bool IsComposite(uint32_t typeId);

  // Return the number of subcomponents in the composite type |typeId|.
  // Return 0 if not a composite type or number of components is not a
  // 32-bit constant.
  uint32_t NumComponents(uint32_t typeId);

  // Mark all inserts in instruction chain ending at |insertChain| with
  // indices that intersect with extract indices |extIndices| starting with
  // index at |extOffset|. Chains are composed solely of Inserts and Phis.
  // Mark all inserts in chain if |extIndices| is nullptr.
  void MarkInsertChain(ir::Instruction* insertChain,
                       std::vector<uint32_t>* extIndices, uint32_t extOffset);

  // Perform EliminateDeadInsertsOnePass(|func|) until no modification is
  // made. Return true if modified.
  bool EliminateDeadInserts(ir::Function* func);

  // DCE all dead struct, matrix and vector inserts in |func|. An insert is
  // dead if the value it inserts is never used. Replace any reference to the
  // insert with its original composite. Return true if modified. Dead inserts
  // in dependence cycles are not currently eliminated. Dead inserts into
  // arrays are not currently eliminated.
  bool EliminateDeadInsertsOnePass(ir::Function* func);

  // Return id of component of |cinst| specified by |extIndices| starting with
  // index at |extOffset|. Return 0 if indices cannot be matched exactly.
  uint32_t DoExtract(ir::Instruction* cinst, std::vector<uint32_t>* extIndices,
                     uint32_t extOffset);

  // Look for OpExtract on sequence of OpInserts in |func|. If there is a
  // reaching insert which corresponds to the indices of the extract, replace
  // the extract with the value that is inserted. Also resolve extracts from
  // CompositeConstruct or ConstantComposite.
  bool EliminateInsertExtract(ir::Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  void Initialize(ir::IRContext* c);
  Pass::Status ProcessImpl();

  // Live inserts
  std::unordered_set<uint32_t> liveInserts_;

  // Visited phis as insert chain is traversed; used to avoid infinite loop
  std::unordered_set<uint32_t> visitedPhis_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSERT_EXTRACT_ELIM_PASS_H_
