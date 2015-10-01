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
#include "../source/text_handler.h"
#include "../source/validate.h"

#include <iomanip>

#ifdef __ANDROID__
#include <sstream>
namespace std {
template <typename T>
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

// TODO(dneto): Using a union this way relies on undefined behaviour.
// Replace this with uses of BitwiseCast from source/bitwisecast.h
static const union {
  unsigned char bytes[4];
  uint32_t value;
} o32_host_order = {{0, 1, 2, 3}};
#define I32_ENDIAN_HOST (o32_host_order.value)

// A namespace for utilities used in SPIR-V Tools unit tests.
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
    *os << "0x" << std::setw(8) << std::setfill('0') << std::hex << value
        << " ";
    if (count++ % 8 == 7) {
      *os << std::endl;
    }
  }
  *os << std::endl;
}

// Returns a vector of words representing a single instruction with the
// given opcode and operand words as a vector.
inline std::vector<uint32_t> MakeInstruction(
    spv::Op opcode, const std::vector<uint32_t>& args) {
  std::vector<uint32_t> result{
      spvOpcodeMake(uint16_t(args.size() + 1), opcode)};
  result.insert(result.end(), args.begin(), args.end());
  return result;
}

// Returns a vector of words representing a single instruction with the
// given opcode and whose operands are the concatenation of the two given
// argument lists.
inline std::vector<uint32_t> MakeInstruction(
    spv::Op opcode, std::vector<uint32_t> args,
    const std::vector<uint32_t>& extra_args) {
  args.insert(args.end(), extra_args.begin(), extra_args.end());
  return MakeInstruction(opcode, args);
}

// Encodes a string as a sequence of words, using the SPIR-V encoding.
inline std::vector<uint32_t> MakeVector(std::string input) {
  std::vector<uint32_t> result;
  uint32_t word = 0;
  size_t num_bytes = input.size();
  // SPIR-V strings are null-terminated.  The byte_index == num_bytes
  // case is used to push the terminating null byte.
  for (size_t byte_index = 0; byte_index <= num_bytes; byte_index++) {
    const auto new_byte =
        (byte_index < num_bytes ? uint8_t(input[byte_index]) : uint8_t(0));
    word |= (new_byte << (8 * (byte_index % sizeof(uint32_t))));
    if (3 == (byte_index % sizeof(uint32_t))) {
      result.push_back(word);
      word = 0;
    }
  }
  // Emit a trailing partial word.
  if ((num_bytes + 1) % sizeof(uint32_t)) {
    result.push_back(word);
  }
  return result;
}

// A type for easily creating spv_text_t values, with an implicit conversion to
// spv_text.
struct AutoText {
  AutoText(std::string value) : str(value), text({str.data(), str.size()}) {}
  operator spv_text() { return &text; }
  std::string str;
  spv_text_t text;
};

// An example case for an enumerated value, optionally with operands.
template <typename E>
class EnumCase {
 public:
  EnumCase(E value, std::string name, std::vector<uint32_t> operands = {})
      : enum_value_(value), name_(name), operands_(operands) {}
  // Returns the enum value as a uint32_t.
  uint32_t value() const { return static_cast<uint32_t>(enum_value_); }
  // Returns the name of the enumerant.
  const std::string& name() const { return name_; }
  // Returns a reference to the operands.
  const std::vector<uint32_t>& operands() const { return operands_; }

 private:
  E enum_value_;
  std::string name_;
  std::vector<uint32_t> operands_;
};

}  // namespace spvtest

#endif
