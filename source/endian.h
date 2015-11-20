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

#ifndef LIBSPIRV_ENDIAN_H_
#define LIBSPIRV_ENDIAN_H_

#include "libspirv/libspirv.h"

// Converts a word in the specified endianness to the host native endianness.
uint32_t spvFixWord(const uint32_t word, const spv_endianness_t endianness);

// Converts a pair of words in the specified endianness to the host native
// endianness.
uint64_t spvFixDoubleWord(const uint32_t low, const uint32_t high,
                          const spv_endianness_t endianness);

// Gets the endianness of the SPIR-V module given in the binary parameter.
// Returns SPV_ENDIANNESS_UNKNOWN if the SPIR-V magic number is invalid,
// otherwise writes the determined endianness into *endian.
spv_result_t spvBinaryEndianness(const spv_const_binary binary,
                                 spv_endianness_t* endian);

// Returns true if the given endianness matches the host's native endiannes.
bool spvIsHostEndian(spv_endianness_t endian);

#endif  // LIBSPIRV_ENDIAN_H_
