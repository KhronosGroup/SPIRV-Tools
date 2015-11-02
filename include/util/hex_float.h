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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "bitutils.h"

namespace spvutils {

template <typename T>
struct FloatProxyTraits {
  typedef void uint_type;
};

template <>
struct FloatProxyTraits<float> {
  typedef uint32_t uint_type;
};

template <>
struct FloatProxyTraits<double> {
  typedef uint64_t uint_type;
};

// Since copying a floating point number (especially if it is NaN)
// does not guarantee that bits are preserved, this class lets us
// store the type and use it as a float when necessary.
template <typename T>
class FloatProxy {
 public:
  using uint_type = typename FloatProxyTraits<T>::uint_type;

  // Since this is to act similar to the normal floats,
  // do not initialize the data by default.
  FloatProxy() = default;

  // Intentionally non-explicit. This is a proxy type so
  // implicit conversions allow us to use it more transparently.
  FloatProxy(T val) { data_ = BitwiseCast<uint_type>(val); }

  // Intentionally non-explicit. This is a proxy type so
  // implicit conversions allow us to use it more transparently.
  FloatProxy(uint_type val) { data_ = val; }

  // This is helpful to have and is guaranteed not to stomp bits.
  FloatProxy<T> operator-() const {
    return data_ ^ (uint_type(0x1) << (sizeof(T) * 8 - 1));
  }

  // Returns the data as a floating point value.
  T getAsFloat() const { return BitwiseCast<T>(data_); }

  // Returns the raw data.
  uint_type data() const { return data_; }

  // Returns true if the value represents any type of NaN.
  bool isNan() { return std::isnan(getAsFloat()); }

 private:
  uint_type data_;
};

template <typename T>
bool operator==(const FloatProxy<T>& first, const FloatProxy<T>& second) {
  return first.data() == second.data();
}

// Convenience to read the value as a normal float.
template <typename T>
std::istream& operator>>(std::istream& is, FloatProxy<T>& value) {
  T float_val;
  is >> float_val;
  value = FloatProxy<T>(float_val);
  return is;
}

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
struct HexFloatTraits<FloatProxy<float>> {
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
struct HexFloatTraits<FloatProxy<double>> {
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
      fraction_nibbles * 4 - num_fraction_bits;

  // The representation of the fraction, not the actual bits. This
  // includes the leading bit that is usually implicit.
  static const uint_type fraction_represent_mask =
      spvutils::SetBits<uint_type, 0,
                        num_fraction_bits + num_overflow_bits>::get;

  // The topmost bit in the fraction. (The first non-implicit bit).
  static const uint_type fraction_top_bit =
      uint_type(1) << (num_fraction_bits + num_overflow_bits - 1);

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
      (sizeof(uint_type) * 8) - num_fraction_bits;

 private:
  T value_;

  static_assert(num_used_bits ==
                    Traits::num_exponent_bits + Traits::num_fraction_bits + 1,
                "The number of bits do not fit");
};

// Returns 4 bits represented by the hex character.
inline uint8_t get_nibble_from_character(char character) {
  const char* dec = "0123456789";
  const char* lower = "abcdef";
  const char* upper = "ABCDEF";
  if (auto p = strchr(dec, character)) return p - dec;
  if (auto p = strchr(lower, character)) return p - lower + 0xa;
  if (auto p = strchr(upper, character)) return p - upper + 0xa;

  assert(false && "This was called with a non-hex character");
  return 0;
}

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

  const uint_type bits = spvutils::BitwiseCast<uint_type>(value.value());
  const char* const sign = (bits & HF::sign_mask) ? "-" : "";
  const uint_type exponent =
      (bits & HF::exponent_mask) >> HF::num_fraction_bits;

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

template <typename T, typename Traits>
inline std::istream& ParseNormalFloat(std::istream& is, bool negate_value,
                                      HexFloat<T, Traits>& value) {
  T val;
  is >> val;
  if (negate_value) {
    val = -val;
  }
  value.set_value(val);
  return is;
}

// Reads a HexFloat from the given stream.
// If the float is not encoded as a hex-float then it will be parsed
// as a regular float.
// This may fail if your stream does not support at least one unget.
// Nan values can be encoded with "0x1.<not zero>p+exponent_bias".
// This would normally overflow a float and round to
// infinity but this special pattern is the exact representation for a NaN,
// and therefore is actually encoded as the correct NaN. To encode inf,
// either 0x0p+exponent_bias can be spcified or any exponent greater than
// exponent_bias.
// Examples using IEEE 32-bit float encoding.
//    0x1.0p+128 (+inf)
//    -0x1.0p-128 (-inf)
//
//    0x1.1p+128 (+Nan)
//    -0x1.1p+128 (-Nan)
//
//    0x1p+129 (+inf)
//    -0x1p+129 (-inf)
template <typename T, typename Traits>
std::istream& operator>>(std::istream& is, HexFloat<T, Traits>& value) {
  using HF = HexFloat<T, Traits>;
  using uint_type = typename HF::uint_type;
  using int_type = typename HF::int_type;

  value.set_value(T(0.f));

  if (is.flags() & std::ios::skipws) {
    // If the user wants to skip whitespace , then we should obey that.
    while (std::isspace(is.peek())) {
      is.get();
    }
  }

  char next_char = is.peek();
  bool negate_value = false;

  if (next_char != '-' && next_char != '0') {
    return ParseNormalFloat(is, negate_value, value);
  }

  if (next_char == '-') {
    negate_value = true;
    is.get();
    next_char = is.peek();
  }

  if (next_char == '0') {
    is.get();  // We may have to unget this.
    char maybe_hex_start = is.peek();
    if (maybe_hex_start != 'x' && maybe_hex_start != 'X') {
      is.unget();
      return ParseNormalFloat(is, negate_value, value);
    } else {
      is.get();  // Throw away the 'x';
    }
  } else {
    return ParseNormalFloat(is, negate_value, value);
  }

  // This "looks" like a hex-float so treat it as one.
  bool seen_p = false;
  bool seen_dot = false;
  uint_type fraction_index = 0;

  uint_type fraction = 0;
  int_type exponent = HF::exponent_bias;

  // Strip off leading zeros so we don't have to special-case them later.
  while ((next_char = is.peek()) == '0') {
    is.get();
  }

  bool is_denorm =
      true;  // Assume denorm "representation" until we hear otherwise.
             // NB: This does not mean the value is actually denorm,
             // it just means that it was written 0.
  bool bits_written = false;  // Stays false until we write a bit.
  while (!seen_p && !seen_dot) {
    // Handle characters that are left of the fractional part.
    if (next_char == '.') {
      seen_dot = true;
    } else if (next_char == 'p') {
      seen_p = true;
    } else if (::isxdigit(next_char)) {
      // We know this is not denormalized since we have stripped all leading
      // zeroes and we are not a ".".
      is_denorm = false;
      uint8_t number = get_nibble_from_character(next_char);
      for (int i = 0; i < 4; ++i, number <<= 1) {
        uint_type write_bit = (number & 0x8) ? 0x1 : 0x0;
        if (bits_written) {
          // If we are here the bits represented belong in the fractional
          // part of the float, and we have to adjust the exponent accordingly.
          fraction |= write_bit << (HF::top_bit_left_shift - fraction_index++);
          exponent += 1;
        }
        bits_written |= write_bit != 0;
      }
    } else {
      // We have not found our exponent yet, so we have to fail.
      is.setstate(std::ios::failbit);
      return is;
    }
    is.get();
    next_char = is.peek();
  }
  bits_written = false;
  while (seen_dot && !seen_p) {
    // Handle only fractional parts now.
    if (next_char == 'p') {
      seen_p = true;
    } else if (::isxdigit(next_char)) {
      int number = get_nibble_from_character(next_char);
      for (int i = 0; i < 4; ++i, number <<= 1) {
        uint_type write_bit = (number & 0x8) ? 0x01 : 0x00;
        bits_written |= write_bit != 0;
        if (is_denorm && !bits_written) {
          // Handle modifying the exponent here this way we can handle
          // an arbitrary number of hex values without overflowing our
          // integer.
          exponent -= 1;
        } else {
          fraction |= write_bit << (HF::top_bit_left_shift - fraction_index++);
        }
      }
    } else {
      // We still have not found our 'p' exponent yet, so this is not a valid
      // hex-float.
      is.setstate(std::ios::failbit);
      return is;
    }
    is.get();
    next_char = is.peek();
  }

  bool seen_sign = false;
  int8_t exponent_sign = 1;
  int_type written_exponent = 0;
  while (true) {
    if ((next_char == '-' || next_char == '+')) {
      if (seen_sign) {
        is.setstate(std::ios::failbit);
        return is;
      }
      seen_sign = true;
      exponent_sign = (next_char == '-') ? -1 : 1;
    } else if (::isdigit(next_char)) {
      // Hex-floats express their exponent as decimal.
      written_exponent *= 10;
      written_exponent += next_char - '0';
    } else {
      break;
    }
    is.get();
    next_char = is.peek();
  }

  written_exponent *= exponent_sign;
  exponent += written_exponent;

  bool is_zero = is_denorm && (fraction == 0);
  if (is_denorm && !is_zero) {
    fraction <<= 1;
    exponent -= 1;
  } else if (is_zero) {
    exponent = 0;
  }

  if (exponent <= 0 && !is_zero) {
    fraction >>= 1;
    fraction |= static_cast<uint_type>(1) << HF::top_bit_left_shift;
  }

  fraction = (fraction >> HF::fraction_right_shift) & HF::fraction_encode_mask;

  const uint_type max_exponent =
      SetBits<uint_type, 0, HF::num_exponent_bits>::get;

  // Handle actual denorm numbers
  while (exponent < 0 && !is_zero) {
    fraction >>= 1;
    exponent += 1;

    fraction &= HF::fraction_encode_mask;
    if (fraction == 0) {
      // We have underflowed our fraction. We should clamp to zero.
      is_zero = true;
      exponent = 0;
    }
  }

  // We have overflowed so we should be inf/-inf.
  if (exponent > max_exponent) {
    exponent = max_exponent;
    fraction = 0;
  }

  uint_type output_bits = static_cast<uint_type>(negate_value ? 1 : 0)
                          << HF::top_bit_left_shift;
  output_bits |= fraction;
  output_bits |= (exponent << HF::exponent_left_shift) & HF::exponent_mask;

  T output_float = spvutils::BitwiseCast<T>(output_bits);
  value.set_value(output_float);

  return is;
}
}

#endif  // _LIBSPIRV_UTIL_HEX_FLOAT_H_
