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

#include <memory>
#include <vector>

#include "log.h"
#include "message.h"
#include "module.h"
#include "passes.h"

namespace spvtools {
namespace opt {

// The pass manager, responsible for tracking and running passes.
// Clients should first call AddPass() to add passes and then call Run()
// to run on a module. Passes are executed in the exact order of addition.
class PassManager {
 public:
  // Constructs a pass manager with the given message consumer.
  explicit PassManager(MessageConsumer c) : consumer_(std::move(c)) {}

  // Adds an externally constructed pass.
  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
  }
  // Uses the argument |args| to construct a pass instance of type |T|, and adds
  // the pass instance to this pass manger. The pass added will use this pass
  // manager's message consumer.
  template <typename T, typename... Args>
  void AddPass(Args&&... args) {
    passes_.emplace_back(new T(consumer_, std::forward<Args>(args)...));
  }

  // Returns the number of passes added.
  uint32_t NumPasses() const { return static_cast<uint32_t>(passes_.size()); }
  // Returns a pointer to the |index|th pass added.
  Pass* GetPass(uint32_t index) const {
    SPIRV_ASSERT(consumer_, index < passes_.size(), "index out of bound");
    return passes_[index].get();
  }

  // Returns the message consumer.
  const MessageConsumer& consumer() const { return consumer_; }

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
  // Consumer for messages.
  MessageConsumer consumer_;
  // A vector of passes. Order matters.
  std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_MANAGER_H_
