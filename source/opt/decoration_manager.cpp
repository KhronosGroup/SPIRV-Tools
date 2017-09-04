// Copyright (c) 2017 Pierre Moreau
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

#include "decoration_manager.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DecorationManager::AnalyzeDecorations(ir::Module* module) {
  if (!module) return;

  // Collect all ids decorated at least once, and all group ids.
  for (auto i = module->annotation_begin(); i != module->annotation_end();
       ++i) {
    switch (i->opcode()) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpMemberDecorate:
        id_to_decoration_insts_.insert({ i->GetSingleWordInOperand(0u), {} });
        break;
      case SpvOpGroupDecorate:
        for (uint32_t j = 1u; j < i->NumInOperands(); ++j)
          id_to_decoration_insts_.insert({ i->GetSingleWordInOperand(j), {} });
        break;
      case SpvOpGroupMemberDecorate:
        for (uint32_t j = 1u; j < i->NumInOperands(); j += 2u)
          id_to_decoration_insts_.insert({ i->GetSingleWordInOperand(j), {} });
        break;
      case SpvOpDecorationGroup:
        group_to_decoration_insts_.insert({ i->GetSingleWordInOperand(0u), {} });
        break;
      default:
        break;
    }
  }

  // For each group, collect all its decoration instructions.
  for (const ir::Instruction& inst : module->annotations()) {
    switch (inst.opcode()) {
      case SpvOpDecorate:
      case SpvOpDecorateId:
      case SpvOpMemberDecorate:
        id_to_decoration_insts_[inst.GetSingleWordInOperand(0u)].push_back(inst);
        break;
      case SpvOpGroupDecorate:
        for (uint32_t j = 1u; j < i->NumInOperands(); ++j)
          id_to_decoration_insts_.insert({ i->GetSingleWordInOperand(j), {} });
        break;
      case SpvOpGroupMemberDecorate:
        for (uint32_t j = 1u; j < i->NumInOperands(); j += 2u)
          id_to_decoration_insts_.insert({ i->GetSingleWordInOperand(j), {} });
        break;
      default:
        break;
    }
  }

  // For each id in |id_to_decoration_insts_|, collect its decorations.
  for (auto i = module->annotation_begin(); i != module->annotation_end();
       ++i) {
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
