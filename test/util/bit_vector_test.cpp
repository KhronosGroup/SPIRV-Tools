// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "gmock/gmock.h"

#include "util/bit_vector.h"

namespace {

using spvtools::utils::BitVector;
using BitVectorTest = ::testing::Test;

TEST(BitVectorTest, Initialize) {
  BitVector bvec;

  // Checks that all values are 0.  Also tests checking a  bit past the end of
  // the vector containing the bits.
  for (int i = 1; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }
}

TEST(BitVectorTest, Set) {
  BitVector bvec;

  // Since 10,000 is larger than the initial size, this tests the resizing
  // code.
  for (int i = 3; i < 10000; i *= 2) {
    bvec.Set(i);
  }

  // Check that bits that were not set are 0.
  for (int i = 1; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }

  // Check that bits that were set are 1.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Get(i));
  }
}

TEST(BitVectorTest, Clear) {
  BitVector bvec;
  for (int i = 3; i < 10000; i *= 2) {
    bvec.Set(i);
  }

  // Check that the bits were properly set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Get(i));
  }

  // Clear all of the bits except for bit 3.
  for (int i = 6; i < 10000; i *= 2) {
    bvec.Clear(i);
  }

  // Make sure bit 3 was not cleared.
  EXPECT_TRUE(bvec.Get(3));

  // Make sure all of the other bits that were set have been cleared.
  for (int i = 6; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }
}

}  // namespace
