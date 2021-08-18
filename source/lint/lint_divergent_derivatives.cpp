// Copyright (c) 2021 Google LLC.
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

#include <cassert>
#include <sstream>
#include <string>

#include "source/diagnostic.h"
#include "source/lint/divergence_analysis.h"
#include "source/lint/lints.h"
#include "source/opt/basic_block.h"
#include "source/opt/cfg.h"
#include "source/opt/control_dependence.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.h"

namespace spvtools {
namespace lint {
namespace lints {
namespace {
// Returns the %name[id], where `name` is the first name associated with the
// given id, or just %id if one is not found.
std::string FriendlyName(opt::IRContext* context, uint32_t id) {
  auto names = context->GetNames(id);
  std::stringstream ss;
  ss << "%";
  if (names.empty()) {
    ss << id;
  } else {
    opt::Instruction* inst_name = names.begin()->second;
    if (inst_name->opcode() == SpvOpName) {
      ss << names.begin()->second->GetInOperand(0).AsString();
      ss << "[" << id << "]";
    } else {
      ss << id;
    }
  }
  return ss.str();
}

bool InstructionHasDerivative(const opt::Instruction& inst) {
  static const SpvOp derivative_opcodes[] = {
      // implicit derivatives
      SpvOpImageSampleImplicitLod,
      SpvOpImageSampleDrefImplicitLod,
      SpvOpImageSampleProjImplicitLod,
      SpvOpImageSampleProjDrefImplicitLod,
      SpvOpImageSparseSampleImplicitLod,
      SpvOpImageSparseSampleDrefImplicitLod,
      SpvOpImageSparseSampleProjImplicitLod,
      SpvOpImageSparseSampleProjDrefImplicitLod,
      // explicit derivatives
      SpvOpDPdx,
      SpvOpDPdy,
      SpvOpFwidth,
      SpvOpDPdxFine,
      SpvOpDPdyFine,
      SpvOpFwidthFine,
      SpvOpDPdxCoarse,
      SpvOpDPdyCoarse,
      SpvOpFwidthCoarse,
  };
  return std::find(std::begin(derivative_opcodes), std::end(derivative_opcodes),
                   inst.opcode()) != std::end(derivative_opcodes);
}

spvtools::DiagnosticStream Warn(opt::IRContext* context,
                                opt::Instruction* inst) {
  if (inst == nullptr) {
    return DiagnosticStream({0, 0, 0}, context->consumer(), "", SPV_WARNING);
  } else {
    // TODO(dongja): line numbers based on debug info
    return DiagnosticStream(
        {0, 0, 0}, context->consumer(),
        inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES),
        SPV_WARNING);
  }
}

void PrintDivergenceFlow(opt::IRContext* context, DivergenceAnalysis div,
                         uint32_t id) {
  opt::analysis::DefUseManager* def_use = context->get_def_use_mgr();
  opt::CFG* cfg = context->cfg();
  while (id != 0) {
    bool is_block = def_use->GetDef(id)->opcode() == SpvOpLabel;
    if (is_block) {
      Warn(context, nullptr)
          << "block " << FriendlyName(context, id) << " is divergent";
      uint32_t source = div.GetDivergenceSource(id);
      // Skip intermediate blocks.
      while (source != 0 && def_use->GetDef(source)->opcode() == SpvOpLabel) {
        id = source;
        source = div.GetDivergenceSource(id);
      }
      if (source == 0) break;
      spvtools::opt::Instruction* branch =
          cfg->block(div.GetDivergenceDependenceSource(id))->terminator();
      Warn(context, branch)
          << "because it depends on a conditional branch on divergent value "
          << FriendlyName(context, source) << "";
      id = source;
    } else {
      Warn(context, nullptr)
          << "value " << FriendlyName(context, id) << " is divergent";
      uint32_t source = div.GetDivergenceSource(id);
      opt::Instruction* def = def_use->GetDef(id);
      opt::Instruction* source_def =
          source == 0 ? nullptr : def_use->GetDef(source);
      // First print data -> data dependencies.
      while (source != 0 && source_def->opcode() != SpvOpLabel) {
        Warn(context, def_use->GetDef(id))
            << "because " << FriendlyName(context, id) << " uses value "
            << FriendlyName(context, source)
            << "in its definition, which is divergent";
        id = source;
        def = source_def;
        source = div.GetDivergenceSource(id);
        source_def = def_use->GetDef(source);
      }
      if (source == 0) {
        Warn(context, def) << "because it has a divergent definition";
        break;
      }
      Warn(context, def) << "because it is conditionally set in block "
                         << FriendlyName(context, source);
      id = source;
    }
  }
}
}  // namespace

bool CheckDivergentDerivatives(opt::IRContext* context) {
  DivergenceAnalysis div(*context);
  for (opt::Function& func : *context->module()) {
    div.Run(&func);
    for (const opt::BasicBlock& bb : func) {
      for (const opt::Instruction& inst : bb) {
        if (InstructionHasDerivative(inst) &&
            div.GetDivergenceLevel(bb.id()) >
                DivergenceAnalysis::DivergenceLevel::kPartiallyUniform) {
          Warn(context, nullptr)
              << "derivative with divergent control flow"
              << " located in block " << FriendlyName(context, bb.id());
          PrintDivergenceFlow(context, div, bb.id());
        }
      }
    }
  }
  return true;
}

}  // namespace lints
}  // namespace lint
}  // namespace spvtools
