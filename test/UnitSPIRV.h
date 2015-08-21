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

#ifndef _CODEPLAY_UNITBIL_H_
#define _CODEPLAY_UNITBIL_H_

#include <libspirv/libspirv.h>
#include "../source/binary.h"
#include "../source/diagnostic.h"
#include "../source/opcode.h"
#include "../source/text.h"
#include "../source/validate.h"

#include <iomanip>

#ifdef __ANDROID__
#include <sstream>
namespace std {
template<typename T>
std::string to_string(const T& val) {
  std::ostringstream os;
  os << val;
  return os.str();
}
}
#endif

#include <gtest/gtest.h>

#include <stdint.h>

// Determine endianness & predicate tests on it
enum {
  I32_ENDIAN_LITTLE = 0x03020100ul,
  I32_ENDIAN_BIG = 0x00010203ul,
};

static const union {
  unsigned char bytes[4];
  uint32_t value;
} o32_host_order = {{0, 1, 2, 3}};

inline ::std::ostream& operator<<(::std::ostream& os,
                                  const spv_binary_t& binary) {
  for (size_t i = 0; i < binary.wordCount; ++i) {
    os << "0x" << std::setw(8) << std::setfill('0') << std::hex
       << binary.code[i] << " ";
    if (i % 8 == 7) {
      os << std::endl;
    }
  }
  os << std::endl;
  return os;
}

namespace std {
inline ::std::ostream& operator<<(::std::ostream& os,
                                  const std::vector<uint32_t>& value) {
  size_t count = 0;
  for (size_t i : value) {
    os << "0x" << std::setw(8) << std::setfill('0') << std::hex << i << " ";
    if (count++ % 8 == 7) {
      os << std::endl;
    }
  }
  os << std::endl;
  return os;
}
}

#define I32_ENDIAN_HOST (o32_host_order.value)

#endif
