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

#ifndef LIBSPIRV_OPT_PASSES_H_
#define LIBSPIRV_OPT_PASSES_H_

#include <memory>

#include "module.h"

namespace spvtools {
namespace opt {

// A pass. All analysis and transformation is done via the Process() method.
class Pass {
 public:
  // Returns a descriptive name for this pass.
  virtual const char* name() const = 0;
  // Processes the given |module| and returns true if the given |module| is
  // modified for optimization.
  virtual bool Process(ir::Module* module) = 0;
};

// A null pass that does nothing.
class NullPass : public Pass {
  const char* name() const override { return "null"; }
  bool Process(ir::Module*) override { return false; }
};

// The optimization pass for removing debug instructions (as documented in
// Section 3.32.2 of the SPIR-V spec).
class StripDebugInfoPass : public Pass {
 public:
  const char* name() const override { return "strip-debug"; }
  bool Process(ir::Module* module) override;
};

class FreezeSpecConstantValuePass : public Pass {
 public:
  const char* name() const override { return "freeze-spec-const"; }
  bool Process(ir::Module*) override;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASSES_H_
