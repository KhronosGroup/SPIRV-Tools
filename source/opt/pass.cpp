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

#include "pass.h"

#include "iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kEntryPointFunctionIdInIdx = 1;

}  // namespace anonymous

void Pass::AddCalls(ir::Function* func,
    std::unordered_set<uint32_t>* next) {
  for (auto bi = func->begin(); bi != func->end(); ++bi)
    for (auto ii = bi->begin(); ii != bi->end(); ++ii)
      if (ii->opcode() == SpvOpFunctionCall)
        next->insert(ii->GetSingleWordInOperand(0));
}

bool Pass::ProcessEntryPointCallTree(
    ProcessFunction& pfn, ir::Module* module) {
  // Map from function's result id to function
  std::unordered_map<uint32_t, ir::Function*> id2function;
  id2function.clear();
  for (auto& fn : *module)
    id2function[fn.result_id()] = &fn;
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> todo;
  std::unordered_set<uint32_t> next;
  std::unordered_set<uint32_t> done;
  for (auto& e : module->entry_points())
    todo.insert(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  while (!todo.empty()) {
    for (auto& fi : todo) {
      if (done.find(fi) != done.end())
        continue;
      ir::Function* fn = id2function[fi];
      modified = pfn(fn) || modified;
      done.insert(fi);
      AddCalls(fn, &next);
    }
    todo.clear();
    todo.insert(next.begin(), next.end());
    next.clear();
  }
  return modified;
}

}  // namespace opt
}  // namespace spvtools

