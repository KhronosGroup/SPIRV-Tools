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

#include <algorithm>
#include <iostream>
#include <set>

#include "gmock/gmock.h"
#include "util/move_to_front.h"

namespace {

using spvutils::MoveToFront;

void CheckTree(const MoveToFront& mtf, const std::string& expected,
               bool print_timestamp = false) {
  std::stringstream ss;
  mtf.PrintTree(ss, print_timestamp);
  EXPECT_EQ(expected, ss.str());
}

TEST(MoveToFront, EmptyTree) {
  MoveToFront mtf;
  CheckTree(mtf, std::string());
}

TEST(MoveToFront, InsertLeftRotation) {
  MoveToFront mtf;

  mtf.TestInsert(30);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
30H2S2----20H1S1----D2
)").substr(1));

  mtf.TestInsert(10);
  CheckTree(mtf, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertRightRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
10H2S2----D1
          20H1S1----D2
)").substr(1));

  mtf.TestInsert(30);
  CheckTree(mtf, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertRightLeftRotation) {
  MoveToFront mtf;

  mtf.TestInsert(30);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
30H2S2----20H1S1----D2
)").substr(1));

  mtf.TestInsert(25);
  CheckTree(mtf, std::string(R"(
25H2S3----20H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertLeftRightRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
10H2S2----D1
          20H1S1----D2
)").substr(1));

  mtf.TestInsert(15);
  CheckTree(mtf, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveSingleton) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  CheckTree(mtf, std::string(R"(
10H1S1----D1
)").substr(1));

  mtf.TestRemove(10);
  CheckTree(mtf, "");
}

TEST(MoveToFront, RemoveRootWithScapegoat) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(5);
  mtf.TestInsert(15);
  CheckTree(mtf, std::string(R"(
10H2S3----5H1S1-----D2
          15H1S1----D2
)").substr(1));

  mtf.TestRemove(10);
  CheckTree(mtf, std::string(R"(
15H2S2----5H1S1-----D2
)").substr(1));
}

TEST(MoveToFront, RemoveRightRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(5);
  mtf.TestInsert(15);
  mtf.TestInsert(20);
  CheckTree(mtf, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----D2
                    20H1S1----D3
)").substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveLeftRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(1);
  CheckTree(mtf, std::string(R"(
10H3S4----5H2S2-----1H1S1-----D3
          15H1S1----D2
)").substr(1));

  mtf.TestRemove(15);

  CheckTree(mtf, std::string(R"(
5H2S3-----1H1S1-----D2
          10H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveLeftRightRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(12);
  CheckTree(mtf, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----12H1S1----D3
)").substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
12H2S3----10H1S1----D2
          15H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveRightLeftRotation) {
  MoveToFront mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(8);
  CheckTree(mtf, std::string(R"(
10H3S4----5H2S2-----D2
                    8H1S1-----D3
          15H1S1----D2
)").substr(1));

  mtf.TestRemove(15);

  CheckTree(mtf, std::string(R"(
8H2S3-----5H1S1-----D2
          10H1S1----D2
)").substr(1));
}

TEST(MoveToFront, MultipleOperations) {
  MoveToFront mtf;
  std::vector<uint32_t> vals =
      { 5, 11, 12, 16, 15, 6, 14, 2, 7, 10, 4, 8, 9, 3, 1, 13 };

  for (uint32_t i : vals) {
    mtf.TestInsert(i);
  }

  CheckTree(mtf, std::string(R"(
11H5S16---5H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S5-----6H1S1-----D4
                              9H2S3-----8H1S1-----D5
                                        10H1S1----D5
          15H3S5----13H2S3----12H1S1----D4
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mtf.TestRemove(11);

  CheckTree(mtf, std::string(R"(
12H5S15---5H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S5-----6H1S1-----D4
                              9H2S3-----8H1S1-----D5
                                        10H1S1----D5
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mtf.TestInsert(11);

  CheckTree(mtf, std::string(R"(
12H5S16---5H4S11----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    9H3S6-----7H2S3-----6H1S1-----D5
                                        8H1S1-----D5
                              10H2S2----D4
                                        11H1S1----D5
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
12H5S15---6H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    9H3S5-----7H2S2-----D4
                                        8H1S1-----D5
                              10H2S2----D4
                                        11H1S1----D5
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mtf.TestInsert(5);

  CheckTree(mtf, std::string(R"(
12H5S16---6H4S11----3H3S5-----2H2S2-----1H1S1-----D5
                              4H2S2-----D4
                                        5H1S1-----D5
                    9H3S5-----7H2S2-----D4
                                        8H1S1-----D5
                              10H2S2----D4
                                        11H1S1----D5
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mtf.TestRemove(2);
  mtf.TestRemove(1);
  mtf.TestRemove(4);
  mtf.TestRemove(3);
  mtf.TestRemove(6);
  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
12H4S10---9H3S5-----7H2S2-----D3
                              8H1S1-----D4
                    10H2S2----D3
                              11H1S1----D4
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));
}

TEST(MoveToFront, BiggerScaleTreeTest) {
  MoveToFront mtf;
  std::set<uint32_t> all_vals;

  const uint32_t kMagic1 = 2654435761;
  const uint32_t kMagic2 = 10000;

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mtf.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val % 2 == 0) {
      mtf.TestRemove(val);
      all_vals.erase(val);
    }
  }

  for (uint32_t i = 1000; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mtf.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val > 50) {
      mtf.TestRemove(val);
      all_vals.erase(val);
    }
  }

  EXPECT_EQ(all_vals, std::set<uint32_t>({2, 4, 11, 13, 24, 33, 35, 37, 46}));

  CheckTree(mtf, std::string(R"(
33H4S9----11H3S5----2H2S2-----D3
                              4H1S1-----D4
                    13H2S2----D3
                              24H1S1----D4
          37H2S3----35H1S1----D3
                    46H1S1----D3
)").substr(1));
}

TEST(MoveToFront, RankFromId) {
  MoveToFront mtf;
  EXPECT_EQ(0u, mtf.RankFromId(1));
  EXPECT_EQ(0u, mtf.RankFromId(2));
  EXPECT_EQ(0u, mtf.RankFromId(3));
  CheckTree(mtf, std::string(R"(
2H2S3T2-------1H1S1T1-------D2
              3H1S1T3-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mtf.RankFromId(1));
  CheckTree(mtf, std::string(R"(
3H2S3T3-------2H1S1T2-------D2
              1H1S1T4-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(1u, mtf.RankFromId(1));
  EXPECT_EQ(2u, mtf.RankFromId(3));
  EXPECT_EQ(3u, mtf.RankFromId(2));
  EXPECT_EQ(0u, mtf.RankFromId(4));
  EXPECT_EQ(4u, mtf.RankFromId(1));
  EXPECT_EQ(0u, mtf.RankFromId(5));
  CheckTree(mtf, std::string(R"(
2H3S5T7-------3H1S1T6-------D2
              1H2S3T9-------4H1S1T8-------D3
                            5H1S1T10------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(1u, mtf.RankFromId(5));
  EXPECT_EQ(5u, mtf.GetSize());
  CheckTree(mtf, std::string(R"(
2H3S5T7-------3H1S1T6-------D2
              1H2S3T9-------4H1S1T8-------D3
                            5H1S1T11------D3
)").substr(1), /* print_timestamp = */ true);
}

TEST(MoveToFront, IdFromRank) {
  MoveToFront mtf;
  EXPECT_EQ(1u, mtf.IdFromRank(0));
  EXPECT_EQ(2u, mtf.IdFromRank(0));
  EXPECT_EQ(2u, mtf.IdFromRank(1));
  EXPECT_EQ(2u, mtf.IdFromRank(1));
  EXPECT_EQ(1u, mtf.IdFromRank(2));
  EXPECT_EQ(3u, mtf.IdFromRank(0));
  EXPECT_EQ(3u, mtf.GetSize());
  CheckTree(mtf, std::string(R"(
1H2S3T5-------2H1S1T4-------D2
              3H1S1T6-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(4u, mtf.IdFromRank(0));
  CheckTree(mtf, std::string(R"(
1H3S4T5-------2H1S1T4-------D2
              3H2S2T6-------D2
                            4H1S1T7-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mtf.IdFromRank(2));
  CheckTree(mtf, std::string(R"(
1H3S4T5-------2H1S1T4-------D2
              4H2S2T7-------D2
                            3H1S1T8-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mtf.IdFromRank(1));
  EXPECT_EQ(3u, mtf.IdFromRank(1));
  EXPECT_EQ(2u, mtf.IdFromRank(4));
  CheckTree(mtf, std::string(R"(
4H3S4T7-------1H1S1T5-------D2
              3H2S2T10------D2
                            2H1S1T11------D3
)").substr(1), /* print_timestamp = */ true);
}

TEST(MoveToFront, DeprecateId) {
  MoveToFront mtf;
  EXPECT_EQ(0u, mtf.RankFromId(1));
  EXPECT_EQ(0u, mtf.RankFromId(2));
  EXPECT_EQ(0u, mtf.RankFromId(3));
  EXPECT_EQ(3u, mtf.GetSize());
  CheckTree(mtf, std::string(R"(
2H2S3T2-------1H1S1T1-------D2
              3H1S1T3-------D2
)").substr(1), /* print_timestamp = */ true);

  mtf.DeprecateId(3);
  CheckTree(mtf, std::string(R"(
2H2S2T2-------1H1S1T1-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(2u, mtf.GetSize());
  EXPECT_EQ(2u, mtf.IdFromRank(1));
  EXPECT_EQ(1u, mtf.IdFromRank(2));
  CheckTree(mtf, std::string(R"(
2H2S2T4-------D1
              1H1S1T5-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(2u, mtf.GetSize());
  mtf.DeprecateId(2);
  EXPECT_EQ(1u, mtf.GetSize());

  EXPECT_EQ(4u, mtf.IdFromRank(0));
  EXPECT_EQ(0u, mtf.RankFromId(5));
  CheckTree(mtf, std::string(R"(
4H2S3T6-------1H1S1T5-------D2
              5H1S1T7-------D2
)").substr(1), /* print_timestamp = */ true);
}

TEST(MoveToFront, LargerScale) {
  MoveToFront mtf;
  for (uint32_t i = 1; i < 1000; ++i) {
    ASSERT_EQ(0u, mtf.RankFromId(i));
    ASSERT_EQ(i, mtf.IdFromRank(1));
    ASSERT_EQ(i, mtf.GetSize());
  }

  EXPECT_EQ(1u, mtf.IdFromRank(999));
  EXPECT_EQ(2u, mtf.IdFromRank(999));
  EXPECT_EQ(3u, mtf.IdFromRank(999));
  EXPECT_EQ(4u, mtf.IdFromRank(999));
  EXPECT_EQ(5u, mtf.IdFromRank(999));
  EXPECT_EQ(6u, mtf.IdFromRank(999));
  EXPECT_EQ(905u, mtf.IdFromRank(101));
  EXPECT_EQ(906u, mtf.IdFromRank(101));
  EXPECT_EQ(907u, mtf.IdFromRank(101));
  EXPECT_EQ(805u, mtf.IdFromRank(201));
  EXPECT_EQ(806u, mtf.IdFromRank(201));
  EXPECT_EQ(807u, mtf.IdFromRank(201));
  EXPECT_EQ(705u, mtf.IdFromRank(301));
  EXPECT_EQ(706u, mtf.IdFromRank(301));
  EXPECT_EQ(707u, mtf.IdFromRank(301));
  EXPECT_EQ(401u, mtf.RankFromId(605));
  EXPECT_EQ(401u, mtf.RankFromId(606));
  EXPECT_EQ(401u, mtf.RankFromId(607));
  EXPECT_EQ(607u, mtf.IdFromRank(1));
  EXPECT_EQ(606u, mtf.IdFromRank(2));
  EXPECT_EQ(605u, mtf.IdFromRank(3));
  EXPECT_EQ(707u, mtf.IdFromRank(4));
  EXPECT_EQ(706u, mtf.IdFromRank(5));
  EXPECT_EQ(705u, mtf.IdFromRank(6));
  EXPECT_EQ(807u, mtf.IdFromRank(7));
  EXPECT_EQ(806u, mtf.IdFromRank(8));
  EXPECT_EQ(805u, mtf.IdFromRank(9));
  EXPECT_EQ(907u, mtf.IdFromRank(10));
  EXPECT_EQ(906u, mtf.IdFromRank(11));
  EXPECT_EQ(905u, mtf.IdFromRank(12));
  EXPECT_EQ(6u, mtf.IdFromRank(13));
  EXPECT_EQ(5u, mtf.IdFromRank(14));
  EXPECT_EQ(4u, mtf.IdFromRank(15));
  EXPECT_EQ(3u, mtf.IdFromRank(16));
  EXPECT_EQ(2u, mtf.IdFromRank(17));
  EXPECT_EQ(1u, mtf.IdFromRank(18));
  EXPECT_EQ(999u, mtf.IdFromRank(19));
  EXPECT_EQ(998u, mtf.IdFromRank(20));
  EXPECT_EQ(997u, mtf.IdFromRank(21));
  EXPECT_EQ(1u, mtf.RankFromId(997));
  EXPECT_EQ(2u, mtf.RankFromId(998));
  EXPECT_EQ(22u, mtf.RankFromId(996));
  mtf.DeprecateId(995);
  EXPECT_EQ(23u, mtf.RankFromId(994));

  for (uint32_t i = 10; i < 1000; ++i) {
    if (i != 995)
      mtf.DeprecateId(i);
  }

  CheckTree(mtf, std::string(R"(
6H4S9T2029----8H2S3T16------7H1S1T14------D3
                            9H1S1T18------D3
              2H3S5T2033----4H2S3T2031----5H1S1T2030----D4
                                          3H1S1T2032----D4
                            1H1S1T2034----D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(1000u, mtf.IdFromRank(0));
}

}  // anonymous namespace
