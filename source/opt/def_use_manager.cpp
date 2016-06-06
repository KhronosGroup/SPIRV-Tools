// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include <cassert>
#include <functional>

#include "def_use_manager.h"

#include "constructs.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DefUseManager::AnalyzeDefUse(ir::Module* module) {
  module->ForEachInst(std::bind(&DefUseManager::AnalyzeInstDefUse, this,
                                std::placeholders::_1));
}

void DefUseManager::ReplaceAllUsesWith(uint32_t before, uint32_t after) {
  assert(id_to_defs_.count(before) != 0 && "id to be replace unknown");
  id_to_defs_[before]->ToNop();
  if (!id_to_uses_.count(before)) return;
  for (const auto& p : id_to_uses_[before]) {
    p.first->SetPayload(p.second, {after});
  }
}

void DefUseManager::AnalyzeInstDefUse(ir::Inst* inst) {
  if (inst->result_id() != 0) id_to_defs_[inst->result_id()] = inst;

  for (uint32_t i = 0; i < inst->NumPayloads(); ++i) {
    const auto type = inst->GetPayload(i).type;
    if (type == SPV_OPERAND_TYPE_ID || type == SPV_OPERAND_TYPE_TYPE_ID ||
        type == SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID ||
        type == SPV_OPERAND_TYPE_SCOPE_ID) {
      id_to_uses_[inst->GetSingleWordPayload(i)].emplace_back(inst, i);
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
