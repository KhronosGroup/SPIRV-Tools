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

#ifndef LIBSPIRV_UTIL_BITUTILS_H_
#define LIBSPIRV_UTIL_BITUTILS_H_

#include <cstring>

namespace spvutils {

// Performs a bitwise copy of source to the destination type Dest.
template <typename Dest, typename Src>
Dest BitwiseCast(Src source) {
  Dest dest;
  static_assert(sizeof(source) == sizeof(dest),
                "BitwiseCast: Source and destination must have the same size");
  std::memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// SetBits<T, First, Last> returns an integer of type <T> with the bits
// between First and Last inclusive set and all other bits 0. This is indexed
// from left to right So SetBits<unsigned long, 0, 1> would have the leftmost
// two bits set.
template<typename T, T First = 0, T Last = 0>
struct SetBits {
  static_assert(First < Last, "The first bit must be before the last bit");
  const static T get = (T(1) << ((sizeof(T) * 8) - First - 1)) |
                       SetBits<T, First + 1, Last>::get;
};

template<typename T, T Last>
struct SetBits<T, Last, Last> {
  const static T get = T(1) << ((sizeof(T) * 8) - Last - 1);
};

// This is all compile-time so we can put our tests right here.
static_assert(SetBits<uint32_t, 0, 0>::get == uint32_t(0x80000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 0, 1>::get == uint32_t(0xc0000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 1, 2>::get == uint32_t(0x60000000),
              "SetBits failed");
static_assert(SetBits<uint32_t, 31, 31>::get == uint32_t(0x00000001),
              "SetBits failed");
static_assert(SetBits<uint32_t, 31, 32>::get == uint32_t(0x00000001),
              "SetBits failed");
static_assert(SetBits<uint32_t, 30, 31>::get == uint32_t(0x00000003),
              "SetBits failed");
static_assert(SetBits<uint32_t, 0, 31>::get == uint32_t(0xFFFFFFFF),
              "SetBits failed");
static_assert(SetBits<uint32_t, 16, 31>::get == uint32_t(0x0000FFFF),
              "SetBits failed");

static_assert(SetBits<uint64_t, 0, 0>::get == uint64_t(0x8000000000000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 0, 1>::get == uint64_t(0xc000000000000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 32, 32>::get == uint64_t(0x0000000080000000LL),
              "SetBits failed");
static_assert(SetBits<uint64_t, 16, 31>::get == uint64_t(0x0000FFFF00000000LL),
              "SetBits failed");

}  // namespace spvutils

#endif  // LIBSPIRV_UTIL_BITUTILS_H_
