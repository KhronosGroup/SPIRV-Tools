// Copyright (c) 2022-2025 Arm Ltd.
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

#include "source/opt/graph.h"

namespace spvtools {
namespace opt {

Graph* Graph::Clone(IRContext* ctx) const {
  Graph* clone = new Graph(std::unique_ptr<Instruction>(DefInst().Clone(ctx)));

  clone->inputs_.reserve(inputs_.size());
  for (const auto& i : inputs()) {
    clone->AddInput(std::unique_ptr<Instruction>(i->Clone(ctx)));
  }

  clone->insts_.reserve(insts_.size());
  for (const auto& i : instructions()) {
    clone->AddInstruction(std::unique_ptr<Instruction>(i->Clone(ctx)));
  }

  clone->outputs_.reserve(outputs_.size());
  for (const auto& i : outputs()) {
    clone->AddOutput(std::unique_ptr<Instruction>(i->Clone(ctx)));
  }

  clone->SetGraphEnd(std::unique_ptr<Instruction>(EndInst()->Clone(ctx)));

  clone->non_semantic_.reserve(non_semantic_.size());
  for (const auto& i : non_semantic_) {
    clone->AddNonSemanticInstruction(
        std::unique_ptr<Instruction>(i->Clone(ctx)));
  }

  return clone;
}

void Graph::ForEachInst(const std::function<void(Instruction*)>& f,
                        bool run_on_debug_line_insts,
                        bool run_on_non_semantic_insts) {
  WhileEachInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts, run_on_non_semantic_insts);
}

void Graph::ForEachInst(const std::function<void(const Instruction*)>& f,
                        bool run_on_debug_line_insts,
                        bool run_on_non_semantic_insts) const {
  WhileEachInst(
      [&f](const Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts, run_on_non_semantic_insts);
}

bool Graph::WhileEachInst(const std::function<bool(Instruction*)>& f,
                          bool run_on_debug_line_insts,
                          bool run_on_non_semantic_insts) {
  (void)run_on_debug_line_insts;

  if (!f(def_inst_.get())) {
    return false;
  }

  for (auto& inst : inputs_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  for (auto& inst : insts_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  for (auto& inst : outputs_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  if (!f(end_inst_.get())) {
    return false;
  }

  if (run_on_non_semantic_insts) {
    for (auto& inst : non_semantic_) {
      if (!f(inst.get())) {
        return false;
      }
    }
  }

  return true;
}

bool Graph::WhileEachInst(const std::function<bool(const Instruction*)>& f,
                          bool run_on_debug_line_insts,
                          bool run_on_non_semantic_insts) const {
  (void)run_on_debug_line_insts;

  if (!f(def_inst_.get())) {
    return false;
  }

  for (auto& inst : inputs_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  for (auto& inst : insts_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  for (auto& inst : outputs_) {
    if (!f(inst.get())) {
      return false;
    }
  }

  if (!f(end_inst_.get())) {
    return false;
  }

  if (run_on_non_semantic_insts) {
    for (auto& inst : non_semantic_) {
      if (!f(inst.get())) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace opt
}  // namespace spvtools
