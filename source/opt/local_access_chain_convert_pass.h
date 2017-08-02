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

#ifndef LIBSPIRV_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_
#define LIBSPIRV_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_


#include <algorithm>
#include <map>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "basic_block.h"
#include "def_use_manager.h"
#include "module.h"
#include "mem_pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LocalAccessChainConvertPass : public MemPass {
 public:
  LocalAccessChainConvertPass();
  const char* name() const override { return "convert-local-access-chains"; }
  Status Process(ir::Module*) override;

 private:
  // Search |func| and cache function scope variables of target type that are
  // not accessed with non-constant-index access chains. Also cache non-target
  // variables.
  void FindTargetVars(ir::Function* func);

  // Delete |inst| if it has no uses. Assumes |inst| has a non-zero resultId.
  void DeleteIfUseless(ir::Instruction* inst);

  // Return type id for |ptrInst|'s pointee
  uint32_t GetPointeeTypeId(const ir::Instruction* ptrInst) const;

  // Build instruction from |opcode|, |typeId|, |resultId|, and |in_opnds|.
  // Append to |newInsts|.
  void BuildAndAppendInst(SpvOp opcode, uint32_t typeId, uint32_t resultId,
    const std::vector<ir::Operand>& in_opnds,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts);

  // Build load of variable in |ptrInst| and append to |newInsts|.
  // Return var in |varId| and its pointee type in |varPteTypeId|.
  uint32_t BuildAndAppendVarLoad(const ir::Instruction* ptrInst,
    uint32_t* varId, uint32_t* varPteTypeId,
    std::vector<std::unique_ptr<ir::Instruction>>* newInsts);

  // Append literal integer operands to |in_opnds| corresponding to constant
  // integer operands from access chain |ptrInst|. Assumes all indices in
  // access chains are OpConstant.
  void AppendConstantOperands( const ir::Instruction* ptrInst,
    std::vector<ir::Operand>* in_opnds);

  // Create a load/insert/store equivalent to a store of
  // |valId| through (constant index) access chaing |ptrInst|.
  // Append to |newInsts|.
  void GenAccessChainStoreReplacement(const ir::Instruction* ptrInst,
      uint32_t valId,
      std::vector<std::unique_ptr<ir::Instruction>>* newInsts);

  // For the (constant index) access chain |ptrInst|, create an
  // equivalent load and extract. Append to |newInsts|.
  uint32_t GenAccessChainLoadReplacement(const ir::Instruction* ptrInst,
      std::vector<std::unique_ptr<ir::Instruction>>* newInsts);

  // Return true if all indices of access chain |acp| are OpConstant integers
  bool IsConstantIndexAccessChain(const ir::Instruction* acp) const;

  // Identify all function scope variables of target type which are 
  // accessed only with loads, stores and access chains with constant
  // indices. Convert all loads and stores of such variables into equivalent
  // loads, stores, extracts and inserts. This unifies access to these
  // variables to a single mode and simplifies analysis and optimization.
  // See IsTargetType() for targeted types.
  //
  // Nested access chains and pointer access chains are not currently
  // converted.
  bool ConvertLocalAccessChains(ir::Function* func);

  // Initialize extensions whitelist
  void InitExtensions();

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  // Save next available id into |module|.
  inline void FinalizeNextId(ir::Module* module) {
    module->SetIdBound(next_id_);
  }

  // Return next available id and calculate next.
  inline uint32_t TakeNextId() {
    return next_id_++;
  }

  void Initialize(ir::Module* module);
  Pass::Status ProcessImpl();

  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_whitelist_;

  // Next unused ID
  uint32_t next_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_LOCAL_ACCESS_CHAIN_CONVERT_PASS_H_

