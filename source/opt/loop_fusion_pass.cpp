// Copyright (c) 2018 Google LLC.
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

#include "opt/loop_fusion_pass.h"

#include "opt/ir_context.h"
#include "opt/loop_descriptor.h"
#include "opt/loop_fusion.h"
#include "opt/register_pressure.h"

namespace spvtools {
namespace opt {

Pass::Status LoopFusionPass::Process(opt::IRContext* c) {
  bool modified = false;
  opt::Module* module = c->module();

  // Process each function in the module
  for (opt::Function& f : *module) {
    modified |= ProcessFunction(&f);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool LoopFusionPass::ProcessFunction(opt::Function* function) {
  auto c = function->context();
  opt::LoopDescriptor& ld = *c->GetLoopDescriptor(function);

  // If a loop doesn't have a preheader needs then it needs to be created. Make
  // sure to return Status::SuccessWithChange in that case.
  auto modified = ld.CreatePreHeaderBlocksIfMissing();

  // TODO(tremmelg): Could the only loop that |loop| could possibly be fused be
  // picked out so don't have to check every loop
  for (auto& loop_0 : ld) {
    for (auto& loop_1 : ld) {
      LoopFusion fusion(c, &loop_0, &loop_1);

      if (fusion.AreCompatible() && fusion.IsLegal()) {
        RegisterLiveness liveness(c, function);
        RegisterLiveness::RegionRegisterLiveness reg_pressure{};
        liveness.SimulateFusion(loop_0, loop_1, &reg_pressure);

        if (reg_pressure.used_registers_ <= max_registers_per_loop_) {
          fusion.Fuse();
          // Recurse, as the current iterators will have been invalidated.
          ProcessFunction(function);
          return true;
        }
      }
    }
  }

  return modified;
}

}  // namespace opt
}  // namespace spvtools
