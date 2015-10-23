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

#ifndef _LIBSPIRV_UTIL_HEX_FLOAT_H_
#define _LIBSPIRV_UTIL_HEX_FLOAT_H_

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "bitutils.h"

namespace spvutils {

// This is an example traits. It is not meant to be used in practice, but will
// be the default for any non-specialized type.
template <typename T>
struct HexFloatTraits {
  // Integer type that can store this hex-float.
  typedef void uint_type;
  // Signed integer type that can store this hex-float.
  typedef void int_type;
  // The number of bits that are actually relevant in the uint_type.
  // This allows us to deal with, for example, 24-bit values in a 32-bit
  // integer.
  static const uint32_t num_used_bits = 0;
  // Number of bits that represent the exponent.
  static const uint32_t num_exponent_bits = 0;
  // Number of bits that represent the fractional part.
  static const uint32_t num_fraction_bits = 0;
  // The bias of the exponent. (How much we need to subtract from the stored
  // value to get the correct value.)
  static const uint32_t exponent_bias = 0;
};

// Traits for IEEE float.
// 1 sign bit, 8 exponent bits, 23 fractional bits.
template <>
struct HexFloatTraits<float> {
  typedef uint32_t uint_type;
  typedef int32_t int_type;
  static const uint_type num_used_bits = 32;
  static const uint_type num_exponent_bits = 8;
  static const uint_type num_fraction_bits = 23;
  static const uint_type exponent_bias = 127;
};

// Traits for IEEE double.
// 1 sign bit, 11 exponent bits, 52 fractional bits.
template <>
struct HexFloatTraits<double> {
  typedef uint64_t uint_type;
  typedef int64_t int_type;
  static const uint_type num_used_bits = 64;
  static const uint_type num_exponent_bits = 11;
  static const uint_type num_fraction_bits = 52;
  static const uint_type exponent_bias = 1023;
};

// Template class that houses a floating pointer number.
// It exposes a number of constants based on the provided traits to
// assist in interpreting the bits of the value.
template <typename T, typename Traits = HexFloatTraits<T>>
class HexFloat {
 public:
  using uint_type = typename Traits::uint_type;
  using int_type = typename Traits::int_type;

  explicit HexFloat(T f) : value_(f) {}
  T value() const { return value_; }
  void set_value(T f) { value_ = f; }

  // These are all written like this because it is convenient to have
  // compile-time constants for all of these values.

  // Pass-through values to save typing.
  static const uint32_t num_used_bits = Traits::num_used_bits;
  static const uint32_t exponent_bias = Traits::exponent_bias;
  static const uint32_t num_exponent_bits = Traits::num_exponent_bits;
  static const uint32_t num_fraction_bits = Traits::num_fraction_bits;

  // Number of bits to shift left to set the highest relevant bit.
  static const uint32_t top_bit_left_shift = num_used_bits - 1;
  // How many nibbles (hex characters) the fractional part takes up.
  static const uint32_t fraction_nibbles = (num_fraction_bits + 3) / 4;
  // If the fractional part does not fit evenly into a hex character (4-bits)
  // then we have to left-shift to get rid of leading 0s. This is the amount
  // we have to shift (might be 0).
  static const uint32_t num_overflow_bits =
      fraction_nibbles * 4 -  num_fraction_bits;

  // The representation of the fraction, not the actual bits. This
  // includes the leading bit that is usually implicit.
  static const uint_type fraction_represent_mask =
      spvutils::SetBits<uint_type, 0, num_fraction_bits + 1>::get;

  // The topmost bit in the fraction. (The first non-implicit bit).
  static const uint_type fraction_top_bit =
      uint_type(1) << num_fraction_bits + num_overflow_bits - 1;

  // The mask for the encoded fraction. It does not include the
  // implicit bit.
  static const uint_type fraction_encode_mask =
      spvutils::SetBits<uint_type, 0, num_fraction_bits>::get;

  // The bit that is used as a sign.
  static const uint_type sign_mask = uint_type(1) << top_bit_left_shift;

  // The bits that represent the exponent.
  static const uint_type exponent_mask =
      spvutils::SetBits<uint_type, num_fraction_bits, num_exponent_bits>::get;

  // How far left the exponent is shifted.
  static const uint32_t exponent_left_shift = num_fraction_bits;

  // How far from the right edge the fraction is shifted.
  static const uint32_t fraction_right_shift =
      (sizeof(uint_type)*8) - num_fraction_bits;

 private:
  T value_;

  static_assert(num_used_bits ==
                    Traits::num_exponent_bits + Traits::num_fraction_bits + 1,
                "The number of bits do not fit");
};

// Outputs the given HexFloat to the stream.
template <typename T, typename Traits>
std::ostream& operator<<(std::ostream& os, const HexFloat<T, Traits>& value) {
  using HF = HexFloat<T, Traits>;
  using uint_type = typename HF::uint_type;
  using int_type = typename HF::int_type;

  static_assert(HF::num_used_bits != 0,
                "num_used_bits must be non-zero for a valid float");
  static_assert(HF::num_exponent_bits != 0,
                "num_exponent_bits must be non-zero for a valid float");
  static_assert(HF::num_fraction_bits != 0,
                "num_fractin_bits must be non-zero for a valid float");
  static_assert(HF::num_overflow_bits != 0,
                "num_exponent_bits must be non-zero for a valid float");

  const uint_type bits = spvutils::BitwiseCast<uint_type>(value.value());
  const char* const sign = (bits & HF::sign_mask) ? "-" : "";
  const uint_type exponent = (bits & HF::exponent_mask) >> HF::num_fraction_bits;

  uint_type fraction = (bits & HF::fraction_encode_mask)
                       << HF::num_overflow_bits;

  const bool is_zero = exponent == 0 && fraction == 0;
  const bool is_denorm = exponent == 0 && !is_zero;

  // exponent contains the biased exponent we have to convert it back into
  // the normal range.
  int_type int_exponent = static_cast<int_type>(exponent) - HF::exponent_bias;
  // If the number is all zeros, then we actually have to NOT shift the
  // exponent.
  int_exponent = is_zero ? 0 : int_exponent;

  // If we are denorm, then start shifting, and decreasing the exponent until
  // our leading bit is 1.

  if (is_denorm) {
    while ((fraction & HF::fraction_top_bit) == 0) {
      fraction <<= 1;
      int_exponent -= 1;
    }
    // Since this is denormalized, we have to consume the leading 1 since it
    // will end up being implicit.
    fraction <<= 1;  // eat the leading 1
    fraction &= HF::fraction_represent_mask;
  }

  uint_type fraction_nibbles = HF::fraction_nibbles;
  // We do not have to display any trailing 0s, since this represents the
  // fractional part.
  while (fraction_nibbles > 0 && (fraction & 0xF) == 0) {
    // Shift off any trailing values;
    fraction >>= 4;
    --fraction_nibbles;
  }

  os << sign << "0x" << (is_zero ? '0' : '1');
  if (fraction_nibbles) {
    // Make sure to keep the leading 0s in place, since this is the fractional
    // part.
    os << "." << std::setw(fraction_nibbles) << std::setfill('0') << std::hex
       << fraction;
  }
  os << "p" << std::dec << (int_exponent >= 0 ? "+" : "") << int_exponent;
  return os;
}
}

#endif  // _LIBSPIRV_UTIL_HEX_FLOAT_H_
