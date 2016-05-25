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

#ifndef LIBSPIRV_OPT_SPVBUILDER_H_
#define LIBSPIRV_OPT_SPVBUILDER_H_

#include <memory>

#include "constructs.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {
namespace ir {

// A class for building Module.
class SpvBuilder {
 public:
  SpvBuilder(Module* module) : module_(module) {}

  void SetModuleHeader(uint32_t magic, uint32_t version, uint32_t generator,
                       uint32_t bound, uint32_t reserved) {
    module_->SetHeader({magic, version, generator, bound, reserved});
  }
  void AddInstruction(const spv_parsed_instruction_t* inst);

 private:
  // The module to be built.
  Module* module_;
  // The current Function under construction.
  std::unique_ptr<Function> function_;
  // The current BasicBlock under construction.
  std::unique_ptr<BasicBlock> block_;
  // Line related deug instructions accumulated thus far.
  std::vector<Inst> dbg_line_info_;
};

}  // namespace ir
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_SPVBUILDER_H_
