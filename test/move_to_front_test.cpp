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

void CheckTree(const MoveToFront& mft, const std::string& expected,
               bool print_timestamp = false) {
  std::stringstream ss;
  mft.PrintTree(ss, print_timestamp);
  EXPECT_EQ(expected, ss.str());
}

TEST(MoveToFront, InsertLeftRotation) {
  MoveToFront mft;

  mft.TestInsert(30);
  mft.TestInsert(20);

  CheckTree(mft, std::string(R"(
30H2S2----20H1S1----D2
)").substr(1));

  mft.TestInsert(10);
  CheckTree(mft, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertRightRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(20);

  CheckTree(mft, std::string(R"(
10H2S2----D1
          20H1S1----D2
)").substr(1));

  mft.TestInsert(30);
  CheckTree(mft, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertRightLeftRotation) {
  MoveToFront mft;

  mft.TestInsert(30);
  mft.TestInsert(20);

  CheckTree(mft, std::string(R"(
30H2S2----20H1S1----D2
)").substr(1));

  mft.TestInsert(25);
  CheckTree(mft, std::string(R"(
25H2S3----20H1S1----D2
          30H1S1----D2
)").substr(1));
}

TEST(MoveToFront, InsertLeftRightRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(20);

  CheckTree(mft, std::string(R"(
10H2S2----D1
          20H1S1----D2
)").substr(1));

  mft.TestInsert(15);
  CheckTree(mft, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveSingleton) {
  MoveToFront mft;

  mft.TestInsert(10);
  CheckTree(mft, std::string(R"(
10H1S1----D1
)").substr(1));

  mft.TestRemove(10);
  CheckTree(mft, "");
}

TEST(MoveToFront, RemoveRootWithScapegoat) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(5);
  mft.TestInsert(15);
  CheckTree(mft, std::string(R"(
10H2S3----5H1S1-----D2
          15H1S1----D2
)").substr(1));

  mft.TestRemove(10);
  CheckTree(mft, std::string(R"(
15H2S2----5H1S1-----D2
)").substr(1));
}

TEST(MoveToFront, RemoveRightRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(5);
  mft.TestInsert(15);
  mft.TestInsert(20);
  CheckTree(mft, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----D2
                    20H1S1----D3
)").substr(1));

  mft.TestRemove(5);

  CheckTree(mft, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveLeftRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(15);
  mft.TestInsert(5);
  mft.TestInsert(1);
  CheckTree(mft, std::string(R"(
10H3S4----5H2S2-----1H1S1-----D3
          15H1S1----D2
)").substr(1));

  mft.TestRemove(15);

  CheckTree(mft, std::string(R"(
5H2S3-----1H1S1-----D2
          10H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveLeftRightRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(15);
  mft.TestInsert(5);
  mft.TestInsert(12);
  CheckTree(mft, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----12H1S1----D3
)").substr(1));

  mft.TestRemove(5);

  CheckTree(mft, std::string(R"(
12H2S3----10H1S1----D2
          15H1S1----D2
)").substr(1));
}

TEST(MoveToFront, RemoveRightLeftRotation) {
  MoveToFront mft;

  mft.TestInsert(10);
  mft.TestInsert(15);
  mft.TestInsert(5);
  mft.TestInsert(8);
  CheckTree(mft, std::string(R"(
10H3S4----5H2S2-----D2
                    8H1S1-----D3
          15H1S1----D2
)").substr(1));

  mft.TestRemove(15);

  CheckTree(mft, std::string(R"(
8H2S3-----5H1S1-----D2
          10H1S1----D2
)").substr(1));
}

TEST(MoveToFront, MultipleOperations) {
  MoveToFront mft;
  std::vector<uint32_t> vals =
      { 5, 11, 12, 16, 15, 6, 14, 2, 7, 10, 4, 8, 9, 3, 1, 13 };

  for (uint32_t i : vals) {
    mft.TestInsert(i);
  }

  CheckTree(mft, std::string(R"(
11H5S16---5H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S5-----6H1S1-----D4
                              9H2S3-----8H1S1-----D5
                                        10H1S1----D5
          15H3S5----13H2S3----12H1S1----D4
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mft.TestRemove(11);

  CheckTree(mft, std::string(R"(
12H5S15---5H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S5-----6H1S1-----D4
                              9H2S3-----8H1S1-----D5
                                        10H1S1----D5
          15H3S4----13H2S2----D3
                              14H1S1----D4
                    16H1S1----D3
)").substr(1));

  mft.TestInsert(11);

  CheckTree(mft, std::string(R"(
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

  mft.TestRemove(5);

  CheckTree(mft, std::string(R"(
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

  mft.TestInsert(5);

  CheckTree(mft, std::string(R"(
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

  mft.TestRemove(2);
  mft.TestRemove(1);
  mft.TestRemove(4);
  mft.TestRemove(3);
  mft.TestRemove(6);
  mft.TestRemove(5);

  CheckTree(mft, std::string(R"(
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
  MoveToFront mft;
  std::set<uint32_t> all_vals;

  const uint32_t kMagic1 = 2654435761;
  const uint32_t kMagic2 = 10000;

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mft.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val % 2 == 0) {
      mft.TestRemove(val);
      all_vals.erase(val);
    }
  }

  for (uint32_t i = 1000; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mft.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val > 50) {
      mft.TestRemove(val);
      all_vals.erase(val);
    }
  }

  EXPECT_EQ(all_vals, std::set<uint32_t>({2, 4, 11, 13, 24, 33, 35, 37, 46}));

  CheckTree(mft, std::string(R"(
33H4S9----11H3S5----2H2S2-----D3
                              4H1S1-----D4
                    13H2S2----D3
                              24H1S1----D4
          37H2S3----35H1S1----D3
                    46H1S1----D3
)").substr(1));
}

TEST(MoveToFront, RankFromId) {
  MoveToFront mft;
  EXPECT_EQ(0u, mft.RankFromId(20));
  EXPECT_EQ(1u, mft.RankFromId(1));
  EXPECT_EQ(2u, mft.RankFromId(2));
  EXPECT_EQ(3u, mft.RankFromId(3));
  CheckTree(mft, std::string(R"(
1H3S4T2-------20H1S1T1------D2
              2H2S2T3-------D2
                            3H1S1T4-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(2u, mft.RankFromId(1));
  CheckTree(mft, std::string(R"(
2H3S4T3-------20H1S1T1------D2
              1H2S2T5-------3H1S1T4-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(0u, mft.RankFromId(1));
  EXPECT_EQ(1u, mft.RankFromId(3));
  EXPECT_EQ(2u, mft.RankFromId(2));
  EXPECT_EQ(4u, mft.RankFromId(4));
  EXPECT_EQ(3u, mft.RankFromId(1));
  EXPECT_EQ(5u, mft.RankFromId(5));
  CheckTree(mft, std::string(R"(
4H3S6T9-------3H2S3T7-------20H1S1T1------D3
                            2H1S1T8-------D3
              1H2S2T10------D2
                            5H1S1T11------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(0u, mft.RankFromId(5));
  EXPECT_EQ(6u, mft.GetSize());
  CheckTree(mft, std::string(R"(
4H3S6T9-------3H2S3T7-------20H1S1T1------D3
                            2H1S1T8-------D3
              1H2S2T10------D2
                            5H1S1T12------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(5u, mft.RankFromId(20));
  CheckTree(mft, std::string(R"(
4H3S6T9-------3H2S2T7-------D2
                            2H1S1T8-------D3
              5H2S3T12------1H1S1T10------D3
                            20H1S1T13-----D3
)").substr(1), /* print_timestamp = */ true);
}

TEST(MoveToFront, IdFromRank) {
  MoveToFront mft;
  EXPECT_EQ(1u, mft.IdFromRank(0));
  EXPECT_EQ(2u, mft.IdFromRank(1));
  EXPECT_EQ(2u, mft.IdFromRank(0));
  EXPECT_EQ(2u, mft.IdFromRank(0));
  EXPECT_EQ(1u, mft.IdFromRank(1));
  EXPECT_EQ(3u, mft.IdFromRank(2));
  EXPECT_EQ(3u, mft.GetSize());
  CheckTree(mft, std::string(R"(
1H2S3T5-------2H1S1T4-------D2
              3H1S1T6-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(4u, mft.IdFromRank(3));
  CheckTree(mft, std::string(R"(
1H3S4T5-------2H1S1T4-------D2
              3H2S2T6-------D2
                            4H1S1T7-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mft.IdFromRank(1));
  CheckTree(mft, std::string(R"(
1H3S4T5-------2H1S1T4-------D2
              4H2S2T7-------D2
                            3H1S1T8-------D3
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mft.IdFromRank(0));
  EXPECT_EQ(3u, mft.IdFromRank(0));
  EXPECT_EQ(2u, mft.IdFromRank(3));
  CheckTree(mft, std::string(R"(
4H3S4T7-------1H1S1T5-------D2
              3H2S2T10------D2
                            2H1S1T11------D3
)").substr(1), /* print_timestamp = */ true);
}

TEST(MoveToFront, DeprecateId) {
  MoveToFront mft;
  EXPECT_EQ(0u, mft.RankFromId(20));
  EXPECT_EQ(1u, mft.RankFromId(1));
  EXPECT_EQ(2u, mft.RankFromId(2));
  EXPECT_EQ(3u, mft.RankFromId(3));
  EXPECT_EQ(4u, mft.GetSize());
  CheckTree(mft, std::string(R"(
1H3S4T2-------20H1S1T1------D2
              2H2S2T3-------D2
                            3H1S1T4-------D3
)").substr(1), /* print_timestamp = */ true);

  mft.DeprecateId(3);
  CheckTree(mft, std::string(R"(
1H2S3T2-------20H1S1T1------D2
              2H1S1T3-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(3u, mft.GetSize());
  EXPECT_EQ(2u, mft.IdFromRank(0));
  CheckTree(mft, std::string(R"(
1H2S3T2-------20H1S1T1------D2
              2H1S1T5-------D2
)").substr(1), /* print_timestamp = */ true);

  mft.DeprecateId(20);
  CheckTree(mft, std::string(R"(
1H2S2T2-------D1
              2H1S1T5-------D2
)").substr(1), /* print_timestamp = */ true);

  EXPECT_EQ(2u, mft.GetSize());
  mft.DeprecateId(2);
  EXPECT_EQ(1u, mft.GetSize());
  EXPECT_EQ(1u, mft.IdFromRank(0));
}

TEST(MoveToFront, LargerScale) {
  MoveToFront mft;
  for (uint32_t i = 1; i < 1000; ++i) {
    ASSERT_EQ(i - 1, mft.RankFromId(i));
    ASSERT_EQ(i, mft.IdFromRank(0));
    ASSERT_EQ(i, mft.GetSize());
  }

  EXPECT_EQ(1u, mft.IdFromRank(998));
  EXPECT_EQ(2u, mft.IdFromRank(998));
  EXPECT_EQ(3u, mft.IdFromRank(998));
  EXPECT_EQ(4u, mft.IdFromRank(998));
  EXPECT_EQ(5u, mft.IdFromRank(998));
  EXPECT_EQ(6u, mft.IdFromRank(998));
  EXPECT_EQ(905u, mft.IdFromRank(100));
  EXPECT_EQ(906u, mft.IdFromRank(100));
  EXPECT_EQ(907u, mft.IdFromRank(100));
  EXPECT_EQ(805u, mft.IdFromRank(200));
  EXPECT_EQ(806u, mft.IdFromRank(200));
  EXPECT_EQ(807u, mft.IdFromRank(200));
  EXPECT_EQ(705u, mft.IdFromRank(300));
  EXPECT_EQ(706u, mft.IdFromRank(300));
  EXPECT_EQ(707u, mft.IdFromRank(300));
  EXPECT_EQ(400u, mft.RankFromId(605));
  EXPECT_EQ(400u, mft.RankFromId(606));
  EXPECT_EQ(400u, mft.RankFromId(607));
  EXPECT_EQ(607u, mft.IdFromRank(0));
  EXPECT_EQ(606u, mft.IdFromRank(1));
  EXPECT_EQ(605u, mft.IdFromRank(2));
  EXPECT_EQ(707u, mft.IdFromRank(3));
  EXPECT_EQ(706u, mft.IdFromRank(4));
  EXPECT_EQ(705u, mft.IdFromRank(5));
  EXPECT_EQ(807u, mft.IdFromRank(6));
  EXPECT_EQ(806u, mft.IdFromRank(7));
  EXPECT_EQ(805u, mft.IdFromRank(8));
  EXPECT_EQ(907u, mft.IdFromRank(9));
  EXPECT_EQ(906u, mft.IdFromRank(10));
  EXPECT_EQ(905u, mft.IdFromRank(11));
  EXPECT_EQ(6u, mft.IdFromRank(12));
  EXPECT_EQ(5u, mft.IdFromRank(13));
  EXPECT_EQ(4u, mft.IdFromRank(14));
  EXPECT_EQ(3u, mft.IdFromRank(15));
  EXPECT_EQ(2u, mft.IdFromRank(16));
  EXPECT_EQ(1u, mft.IdFromRank(17));
  EXPECT_EQ(999u, mft.IdFromRank(18));
  EXPECT_EQ(998u, mft.IdFromRank(19));
  EXPECT_EQ(997u, mft.IdFromRank(20));
  EXPECT_EQ(0u, mft.RankFromId(997));
  EXPECT_EQ(1u, mft.RankFromId(998));
  EXPECT_EQ(21u, mft.RankFromId(996));
  mft.DeprecateId(995);
  EXPECT_EQ(22u, mft.RankFromId(994));

  for (uint32_t i = 10; i < 1000; ++i) {
    if (i != 995)
      mft.DeprecateId(i);
  }

  CheckTree(mft, std::string(R"(
6H4S9T2029----8H2S3T16------7H1S1T14------D3
                            9H1S1T18------D3
              2H3S5T2033----4H2S3T2031----5H1S1T2030----D4
                                          3H1S1T2032----D4
                            1H1S1T2034----D3
)").substr(1), /* print_timestamp = */ true);
}

}  // anonymous namespace
