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

#ifndef LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_
#define LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_

#include <memory>
#include <string>
#include <vector>

#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// TODO(antiagainst): expand and export these functions as the C++ interface in
// libspirv.hpp.

// Assembles the given assembly |text| and returns the binary.
std::vector<uint32_t> Assemble(const std::string& text);

// Disassembles the given SPIR-V |binary| and returns the assembly.
std::string Disassemble(const std::vector<uint32_t>& binary);

// Builds and returns a Module for the given SPIR-V |binary|.
std::unique_ptr<ir::Module> BuildModule(const std::vector<uint32_t>& binary);

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_TEST_OPT_OPT_TEST_COMMON_H_
