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


// A namespace for utilities used in SPIR-V Tools unit tests.
// TODO(dneto): Move other type declarations into this namespace.
namespace spvtest {

class WordVector;

// Emits the given word vector to the given stream.
// This function can be used by the gtest value printer.
void PrintTo(const WordVector& words, ::std::ostream* os);

// A proxy class to allow us to easily write out vectors of SPIR-V words.
class WordVector {
 public:
  explicit WordVector(const std::vector<uint32_t>& value) : value_(value) {}
  explicit WordVector(const spv_binary_t& binary)
      : value_(binary.code, binary.code + binary.wordCount) {}

  // Returns the underlying vector.
  const std::vector<uint32_t>& value() const { return value_; }

  // Returns the string representation of this word vector.
  std::string str() const {
    std::ostringstream os;
    PrintTo(*this, &os);
    return os.str();
  }

 private:
  const std::vector<uint32_t> value_;
};

inline void PrintTo(const WordVector& words, ::std::ostream* os) {
  size_t count = 0;
  for (uint32_t value : words.value()) {
    *os << "0x" << std::setw(8) << std::setfill('0') << std::hex << value << " ";
    if (count++ % 8 == 7) {
      *os << std::endl;
    }
  }
  *os << std::endl;
}

} // namespace spvtest

// A type for easily creating spv_text_t values, with an implicit conversion to
// spv_text.
struct AutoText {
  AutoText(std::string value) : str(value), text({str.data(), str.size()}) {}
  operator spv_text() { return &text; }
  std::string str;
  spv_text_t text;
};

#define I32_ENDIAN_HOST (o32_host_order.value)

#endif
