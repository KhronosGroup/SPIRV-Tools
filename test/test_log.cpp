// Copyright (c) 2016 Google Inc.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "message.h"
#include "opt/log.h"

namespace {

using namespace spvtools;
using ::testing::MatchesRegex;

TEST(Log, AssertStatement) {
  int invocation = 0;
  auto consumer = [&invocation](MessageLevel level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(MessageLevel::InternalError, level);
    EXPECT_THAT(source, MatchesRegex(".*test_log.cpp$"));
    EXPECT_STREQ("assertion failed: 1 + 2 == 5", message);
  };

  SPIRV_ASSERT(consumer, 1 + 2 == 5);
#if defined(NDEBUG)
  (void)consumer;
  EXPECT_EQ(0, invocation);
#else
  EXPECT_EQ(1, invocation);
#endif
}

TEST(Log, AssertMessage) {
  int invocation = 0;
  auto consumer = [&invocation](MessageLevel level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(MessageLevel::InternalError, level);
    EXPECT_THAT(source, MatchesRegex(".*test_log.cpp$"));
    EXPECT_STREQ("assertion failed: happy asserting!", message);
  };

  SPIRV_ASSERT(consumer, 1 + 2 == 5, "happy asserting!");
#if defined(NDEBUG)
  (void)consumer;
  EXPECT_EQ(0, invocation);
#else
  EXPECT_EQ(1, invocation);
#endif
}

TEST(Log, AssertFormattedMessage) {
  int invocation = 0;
  auto consumer = [&invocation](MessageLevel level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(MessageLevel::InternalError, level);
    EXPECT_THAT(source, MatchesRegex(".*test_log.cpp$"));
    EXPECT_STREQ("assertion failed: 1 + 2 actually is 3", message);
  };

  SPIRV_ASSERT(consumer, 1 + 2 == 5, "1 + 2 actually is %d", 1 + 2);
#if defined(NDEBUG)
  (void)consumer;
  EXPECT_EQ(0, invocation);
#else
  EXPECT_EQ(1, invocation);
#endif
}

TEST(Log, Unimplemented) {
  int invocation = 0;
  auto consumer = [&invocation](MessageLevel level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(MessageLevel::InternalError, level);
    EXPECT_THAT(source, MatchesRegex(".*test_log.cpp$"));
    EXPECT_STREQ("unimplemented: the-ultimite-feature", message);
  };

  SPIRV_UNIMPLEMENTED(consumer, "the-ultimite-feature");
  EXPECT_EQ(1, invocation);
}

TEST(Log, Unreachable) {
  int invocation = 0;
  auto consumer = [&invocation](MessageLevel level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(MessageLevel::InternalError, level);
    EXPECT_THAT(source, MatchesRegex(".*test_log.cpp$"));
    EXPECT_STREQ("unreachable", message);
  };

  SPIRV_UNREACHABLE(consumer);
  EXPECT_EQ(1, invocation);
}

}  // anonymous namespace
