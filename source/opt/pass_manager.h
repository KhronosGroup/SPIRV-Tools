// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_PASS_MANAGER_H_
#define LIBSPIRV_OPT_PASS_MANAGER_H_

#include <cassert>
#include <memory>
#include <vector>

#include "module.h"
#include "passes.h"

namespace spvtools {
namespace opt {

// The pass manager, responsible for tracking and running passes.
// Clients should first call AddPass() to add passes and then call Run()
// to run on a module. Passes are executed in the exact order of added.
//
// TODO(antiagainst): The pass manager is fairly simple right now. Eventually it
// should support pass dependency, common functionality (like def-use analysis)
// sharing, etc.
class PassManager {
 public:
  // Adds a pass.
  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
  }
  // Uses the argument to construct a pass instance of type PassT, and adds the
  // pass instance to this pass manger.
  template <typename PassT, typename... Args>
  void AddPass(Args&&... args) {
    passes_.emplace_back(new PassT(std::forward<Args>(args)...));
  }

  // Returns the number of passes added.
  uint32_t NumPasses() const { return static_cast<uint32_t>(passes_.size()); }
  // Returns a pointer to the |index|th pass added.
  Pass* GetPass(uint32_t index) const {
    assert(index < passes_.size() && "index out of bound");
    return passes_[index].get();
  }

  // Runs all passes on the given |module|.
  void Run(ir::Module* module) {
    bool modified = false;
    for (const auto& pass : passes_) {
      modified |= pass->Process(module);
    }
    // Set the Id bound in the header in case a pass forgot to do so.
    if (modified) module->SetIdBound(module->ComputeIdBound());
  }

 private:
  // A vector of passes. Order matters.
  std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_MANAGER_H_
