// Copyright (c) 2016 The Khronos Group Inc.
// Copyright (c) 2016 Valve Corporation
// Copyright (c) 2016 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_SSAMEM_PASS_H_
#define LIBSPIRV_OPT_SSAMEM_PASS_H_


#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <utility>
#include <queue>

#include "def_use_manager.h"
#include "module.h"
#include "basic_block.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalAccessChainConvertPass : public Pass {
 public:
  LocalAccessChainConvertPass();
  const char* name() const override { return "sroa"; }
  Status Process(ir::Module*) override;

 private:
   // Module this pass is processing
   ir::Module* module_;

   // Def-Uses for the module we are processing
   std::unique_ptr<analysis::DefUseManager> def_use_mgr_;

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Set of verified target types
  std::unordered_set<uint32_t> seen_target_vars_;

  // Set of verified non-target types
  std::unordered_set<uint32_t> seen_non_target_vars_;

  // Next unused ID
  uint32_t next_id_;

  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  // Returns true if type is a scalar type
  // or a vector or matrix
  bool IsMathType(const ir::Instruction* typeInst);

  // Returns true if type is a math type or a struct or array
  // of a math type.
  bool IsTargetType(const ir::Instruction* typeInst);

  // Given a load or store pointed at by |ip|, return the pointer
  // instruction. Also return the variable's id in |varId|.
  ir::Instruction* GetPtr(ir::Instruction* ip, uint32_t* varId);

  // Return true if variable is function scope variable of targeted type.
  bool IsTargetVar(uint32_t varId);

  // Delete inst if it has no uses. Assumes inst has a resultId.
  void DeleteIfUseless(ir::Instruction* inst);

  // Replace all instances of load's id with replId and delete load
  // and its access chain, if any
  void ReplaceAndDeleteLoad(ir::Instruction* loadInst,
    uint32_t replId,
    ir::Instruction* ptrInst);

  // Return type id for pointer's pointee
  uint32_t GetPteTypeId(const ir::Instruction* ptrInst);

  // Build instruction from |opcode|, |typeId|, |resultId|, and |in_opnds|.
  // Append to |newInsts|.
  void BuildAndAppendInst(SpvOp opcode, uint32_t typeId, uint32_t resultId,
    const std::vector<ir::Operand>& in_opnds,
    std::vector<std::unique_ptr<ir::Instruction>>& newInsts);

  // Build load of variable in |ptrInst| and append to |newInsts|.
  // Return var in |varId| and its pointee type in |varPteTypeId|.
  uint32_t BuildAndAppendVarLoad(const ir::Instruction* ptrInst,
    uint32_t* varId, uint32_t* varPteTypeId,
    std::vector<std::unique_ptr<ir::Instruction>>& newInsts);

  // Append constant operands from access chain |ptrInst| to
  // |in_opnds|. Assumes all indices in access chain are constants.
  void AppendConstantOperands( const ir::Instruction* ptrInst,
    std::vector<ir::Operand>* in_opnds);

  // Create a load/insert/store equivalent to a store of
  // valId through ptrInst.
  void GenAccessChainStoreReplacement(const ir::Instruction* ptrInst,
      uint32_t valId,
      std::vector<std::unique_ptr<ir::Instruction>>& newInsts);

  // For the (constant index) access chain ptrInst, create an
  // equivalent load and extract
  void GenAccessChainLoadReplacement(const ir::Instruction* ptrInst,
      std::vector<std::unique_ptr<ir::Instruction>>& newInsts,
      uint32_t& resultId);

  // Return true if all indices are constant
  bool IsConstantIndexAccessChain(ir::Instruction* acp);

  // Identify all function scope variables of target type which are 
  // accessed only with loads, stores and access chains with constant
  // indices. Convert all loads and stores of such variables into equivalent
  // loads, stores, extracts and inserts. This unifies access to these
  // variables to a single mode and simplifies analysis and optimization.
  // See IsTargetType() for targeted types.
  bool LocalAccessChainConvert(ir::Function* func);

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SSAMEM_PASS_H_

