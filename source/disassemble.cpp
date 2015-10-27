// Copyright (c) 2015 The Khronos Group Inc.
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

// This file contains a disassembler:  It converts a SPIR-V binary
// to text.
// TODO(dneto): Right now it's just a stub, but should evolve to using
// the binary decoder in binary.cpp to semantically decompose the binary
// for us.  All this module has to do is convert the result to text.

#include "libspirv/libspirv.h"

spv_result_t spvBinaryToText(uint32_t* code, const uint64_t wordCount,
                             const uint32_t options,
                             const spv_opcode_table opcodeTable,
                             const spv_operand_table operandTable,
                             const spv_ext_inst_table extInstTable,
                             spv_text* pText, spv_diagnostic* pDiagnostic) {
  return spvBinaryToTextWithFormat(
      code, wordCount, options, opcodeTable, operandTable, extInstTable,
      SPV_ASSEMBLY_SYNTAX_FORMAT_DEFAULT, pText, pDiagnostic);
}
