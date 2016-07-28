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
  // Adds a pass
  void AddPass(std::unique_ptr<Pass> pass) {
    Pass* raw_ptr = pass.release();
    AddPass(raw_ptr);
  }
  void AddPass(std::unique_ptr<Pass, void (*)(Pass*)> pass) {
    passes_.push_back(std::move(pass));
  }
  template <typename PassT>
  void AddPass() {
    AddPass(new PassT);
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
    for (const auto& pass : passes_) {
      // TODO(antiagainst): Currently we ignore the return value of the pass,
      // which indicates whether the module has been modified, since there is
      // nothing shared between passes right now.
      pass->Process(module);
    }
  }

 private:
  // Make a unique_ptr for a given pass pointer, then add the unqiue_ptr to the
  // passes_ list.
  void AddPass(Pass* pass) {
    auto deleter = [](Pass* p) { delete p;};
    auto pass_ptr = std::unique_ptr<Pass, void (*)(Pass*)>(pass, deleter);
    passes_.push_back(std::move(pass_ptr));
  }
  // A vector of passes. Order matters.
  std::vector<std::unique_ptr<Pass, void (*)(Pass*)>> passes_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_MANAGER_H_
