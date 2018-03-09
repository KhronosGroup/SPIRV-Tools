// Copyright (c) 2018 Google LLC.
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

#include <unistd.h>
#include <sstream>

#include "gtest/gtest.h"
#include "source/util/timer.h"

namespace {

using ::spvutils::CumulativeTimer;
using ::spvutils::PrintTimerDescription;
using ::spvutils::ScopedTimer;
using ::spvutils::Timer;

TEST(Timer, Sleep) {
  const unsigned int usleep_time = 71234;
  const double epsilon = 0.001;
  std::ostringstream buf;

  PrintTimerDescription(&buf);
  Timer timer(&buf);
  timer.Start();

  usleep(usleep_time);

  timer.Stop();
  timer.Report("TimerTest");

  EXPECT_GT(epsilon, timer.CPUTime());
  EXPECT_GT(usleep_time * 0.000001 + epsilon, timer.WallTime());
  EXPECT_GT(epsilon, timer.UserTime());
  EXPECT_GT(epsilon, timer.SystemTime());
  EXPECT_EQ(
      "                     PASS name    CPU time   WALL time    USR time"
      "    SYS time\n                     TimerTest        0.00        0.07"
      "        0.00        0.00\n",
      buf.str());
}

TEST(ScopedTimer, Sleep) {
  const unsigned int usleep_time = 71234;
  std::ostringstream buf;

  {
    ScopedTimer scopedtimer(&buf, "ScopedTimerTest");
    usleep(usleep_time);
  }

  EXPECT_EQ(
      "               ScopedTimerTest        0.00        0.07"
      "        0.00        0.00\n",
      buf.str());
}

TEST(CumulativeTimer, Sleep) {
  const unsigned int usleep_time = 71234;
  const double epsilon = 0.001;
  CumulativeTimer *ctimer;
  std::ostringstream buf;

  {
    ctimer = new CumulativeTimer(&buf, "foo");
    ctimer->Start();
    usleep(usleep_time);
    ctimer->Stop();
  }

  CumulativeTimer *foo;
  {
    foo = CumulativeTimer::GetCumulativeTimer("foo");
    foo->Start();
    usleep(usleep_time);
    foo->Stop();
    foo->Report("CumulativeTimerTest");
  }

  EXPECT_GT(epsilon, foo->CPUTime());
  EXPECT_GT(2 * usleep_time * 0.000001 + epsilon, foo->WallTime());
  EXPECT_GT(epsilon, foo->UserTime());
  EXPECT_GT(epsilon, foo->SystemTime());
  EXPECT_EQ(
      "           CumulativeTimerTest        0.00        0.14"
      "        0.00        0.00\n",
      buf.str());

  if (foo) delete foo;
}

}  // anonymous namespace
