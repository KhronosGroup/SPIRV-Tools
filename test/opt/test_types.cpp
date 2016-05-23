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

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "source/opt/types.h"

namespace {

using namespace spvtools::opt::type;

TEST(Types, Equality) {
  std::vector<std::unique_ptr<Type>> types;

  types.emplace_back(new Void());
  auto* voidt = types.back().get();
  types.emplace_back(new Bool());
  auto* boolt = types.back().get();

  // Integer
  types.emplace_back(new Integer(32, true));
  auto* s32 = types.back().get();
  types.emplace_back(new Integer(32, false));
  types.emplace_back(new Integer(64, true));
  types.emplace_back(new Integer(64, false));
  auto* u64 = types.back().get();

  // Float
  types.emplace_back(new Float(32));
  auto* f32 = types.back().get();
  types.emplace_back(new Float(64));

  // Vector
  types.emplace_back(new Vector(s32, 2));
  types.emplace_back(new Vector(s32, 3));
  auto* v3s32 = types.back().get();
  types.emplace_back(new Vector(u64, 4));
  types.emplace_back(new Vector(f32, 3));
  auto* v3f32 = types.back().get();

  // Matrix
  types.emplace_back(new Matrix(v3s32, 3));
  types.emplace_back(new Matrix(v3s32, 4));
  types.emplace_back(new Matrix(v3f32, 4));

  // Array
  types.emplace_back(new Array(f32, 100));
  types.emplace_back(new Array(f32, 42));
  auto* a42f32 = types.back().get();
  types.emplace_back(new Array(u64, 24));

  // RuntimeArray
  types.emplace_back(new RuntimeArray(v3f32));
  types.emplace_back(new RuntimeArray(v3s32));
  auto* rav3s32 = types.back().get();

  // Struct
  types.emplace_back(new Struct({s32}));
  types.emplace_back(new Struct({s32, f32}));
  auto* sts32f32 = types.back().get();
  types.emplace_back(new Struct({u64, a42f32, rav3s32}));

  // Pointer
  types.emplace_back(new Pointer(f32, SpvStorageClassInput));
  types.emplace_back(new Pointer(sts32f32, SpvStorageClassFunction));
  types.emplace_back(new Pointer(a42f32, SpvStorageClassFunction));

  // Function
  types.emplace_back(new Function(voidt, {}));
  types.emplace_back(new Function(voidt, {boolt}));
  types.emplace_back(new Function(voidt, {boolt, s32}));
  types.emplace_back(new Function(s32, {boolt, s32}));

  for (size_t i = 0; i < types.size(); ++i) {
    for (size_t j = 0; j < types.size(); ++j) {
      if (i == j) {
        EXPECT_TRUE(types[i]->IsSame(types[j].get()))
            << "expected '" << types[i]->str() << "' is the same as '"
            << types[j]->str() << "'";
      } else {
        EXPECT_FALSE(types[i]->IsSame(types[j].get()))
            << "expected '" << types[i]->str() << "' is different to '"
            << types[j]->str() << "'";
      }
    }
  }
}

}  // anonymous namespace
