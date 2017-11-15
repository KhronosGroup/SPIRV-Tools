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

#include "opt/build_module.h"
#include "opt/instruction.h"
#include "opt/type_manager.h"

namespace {

using namespace spvtools;

TEST(TypeManager, TypeStrings) {
  const std::string text = R"(
    OpTypeForwardPointer !20 !2 ; id for %p is 20, Uniform is 2
    OpTypeForwardPointer !10000 !1
    %void    = OpTypeVoid
    %bool    = OpTypeBool
    %u32     = OpTypeInt 32 0
    %id4     = OpConstant %u32 4
    %s32     = OpTypeInt 32 1
    %f64     = OpTypeFloat 64
    %v3u32   = OpTypeVector %u32 3
    %m3x3    = OpTypeMatrix %v3u32 3
    %img1    = OpTypeImage %s32 Cube 0 1 1 0 R32f ReadWrite
    %img2    = OpTypeImage %s32 Cube 0 1 1 0 R32f
    %sampler = OpTypeSampler
    %si1     = OpTypeSampledImage %img1
    %si2     = OpTypeSampledImage %img2
    %a5u32   = OpTypeArray %u32 %id4
    %af64    = OpTypeRuntimeArray %f64
    %st1     = OpTypeStruct %u32
    %st2     = OpTypeStruct %f64 %s32 %v3u32
    %opaque1 = OpTypeOpaque ""
    %opaque2 = OpTypeOpaque "opaque"
    %p       = OpTypePointer Uniform %st1
    %f       = OpTypeFunction %void %u32 %u32
    %event   = OpTypeEvent
    %de      = OpTypeDeviceEvent
    %ri      = OpTypeReserveId
    %queue   = OpTypeQueue
    %pipe    = OpTypePipe ReadOnly
    %ps      = OpTypePipeStorage
    %nb      = OpTypeNamedBarrier
  )";

  std::vector<std::pair<uint32_t, std::string>> type_id_strs = {
      {1, "void"},
      {2, "bool"},
      {3, "uint32"},
      // Id 4 is used by the constant.
      {5, "sint32"},
      {6, "float64"},
      {7, "<uint32, 3>"},
      {8, "<<uint32, 3>, 3>"},
      {9, "image(sint32, 3, 0, 1, 1, 0, 3, 2)"},
      {10, "image(sint32, 3, 0, 1, 1, 0, 3, 0)"},
      {11, "sampler"},
      {12, "sampled_image(image(sint32, 3, 0, 1, 1, 0, 3, 2))"},
      {13, "sampled_image(image(sint32, 3, 0, 1, 1, 0, 3, 0))"},
      {14, "[uint32, id(4)]"},
      {15, "[float64]"},
      {16, "{uint32}"},
      {17, "{float64, sint32, <uint32, 3>}"},
      {18, "opaque('')"},
      {19, "opaque('opaque')"},
      {20, "{uint32}*"},
      {21, "(uint32, uint32) -> void"},
      {22, "event"},
      {23, "device_event"},
      {24, "reserve_id"},
      {25, "queue"},
      {26, "pipe(0)"},
      {27, "pipe_storage"},
      {28, "named_barrier"},
  };

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());

  EXPECT_EQ(type_id_strs.size(), manager.NumTypes());
  EXPECT_EQ(2u, manager.NumForwardPointers());

  for (const auto& p : type_id_strs) {
    EXPECT_EQ(p.second, manager.GetType(p.first)->str());
    EXPECT_EQ(p.first, manager.GetId(manager.GetType(p.first)));
  }
  EXPECT_EQ("forward_pointer({uint32}*)", manager.GetForwardPointer(0)->str());
  EXPECT_EQ("forward_pointer(10000)", manager.GetForwardPointer(1)->str());
}

TEST(TypeManager, DecorationOnStruct) {
  const std::string text = R"(
    OpDecorate %struct1 Block
    OpDecorate %struct2 Block
    OpDecorate %struct3 Block
    OpDecorate %struct4 Block

    %u32 = OpTypeInt 32 0             ; id: 5
    %f32 = OpTypeFloat 32             ; id: 6
    %struct1 = OpTypeStruct %u32 %f32 ; base
    %struct2 = OpTypeStruct %f32 %u32 ; different member order
    %struct3 = OpTypeStruct %f32      ; different member list
    %struct4 = OpTypeStruct %u32 %f32 ; the same
    %struct7 = OpTypeStruct %f32      ; no decoration
  )";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());

  ASSERT_EQ(7u, manager.NumTypes());
  ASSERT_EQ(0u, manager.NumForwardPointers());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(5)->str());
  ASSERT_EQ("float32", manager.GetType(6)->str());

  // Try all combinations of pairs. Expect to be the same type only when the
  // same id or (1, 4).
  for (const auto id1 : {1, 2, 3, 4, 7}) {
    for (const auto id2 : {1, 2, 3, 4, 7}) {
      if (id1 == id2 || (id1 == 1 && id2 == 4) || (id1 == 4 && id2 == 1)) {
        EXPECT_TRUE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be the same as %struct"
            << id2;
      } else {
        EXPECT_FALSE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be different with %struct"
            << id2;
      }
    }
  }
}

TEST(TypeManager, DecorationOnMember) {
  const std::string text = R"(
    OpMemberDecorate %struct1  0 Offset 0
    OpMemberDecorate %struct2  0 Offset 0
    OpMemberDecorate %struct3  0 Offset 0
    OpMemberDecorate %struct4  0 Offset 0
    OpMemberDecorate %struct5  1 Offset 0
    OpMemberDecorate %struct6  0 Offset 4

    OpDecorate %struct7 Block
    OpMemberDecorate %struct7  0 Offset 0

    %u32 = OpTypeInt 32 0              ; id: 8
    %f32 = OpTypeFloat 32              ; id: 9
    %struct1  = OpTypeStruct %u32 %f32 ; base
    %struct2  = OpTypeStruct %f32 %u32 ; different member order
    %struct3  = OpTypeStruct %f32      ; different member list
    %struct4  = OpTypeStruct %u32 %f32 ; the same
    %struct5  = OpTypeStruct %u32 %f32 ; member decorate different field
    %struct6  = OpTypeStruct %u32 %f32 ; different member decoration parameter
    %struct7  = OpTypeStruct %u32 %f32 ; extra decoration on the struct
    %struct10 = OpTypeStruct %u32 %f32 ; no member decoration
  )";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());

  ASSERT_EQ(10u, manager.NumTypes());
  ASSERT_EQ(0u, manager.NumForwardPointers());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(8)->str());
  ASSERT_EQ("float32", manager.GetType(9)->str());

  // Try all combinations of pairs. Expect to be the same type only when the
  // same id or (1, 4).
  for (const auto id1 : {1, 2, 3, 4, 5, 6, 7, 10}) {
    for (const auto id2 : {1, 2, 3, 4, 5, 6, 7, 10}) {
      if (id1 == id2 || (id1 == 1 && id2 == 4) || (id1 == 4 && id2 == 1)) {
        EXPECT_TRUE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be the same as %struct"
            << id2;
      } else {
        EXPECT_FALSE(manager.GetType(id1)->IsSame(manager.GetType(id2)))
            << "%struct" << id1 << " is expected to be different with %struct"
            << id2;
      }
    }
  }
}

TEST(TypeManager, DecorationEmpty) {
  const std::string text = R"(
    OpDecorate %struct1 Block
    OpMemberDecorate %struct2  0 Offset 0

    %u32 = OpTypeInt 32 0 ; id: 3
    %f32 = OpTypeFloat 32 ; id: 4
    %struct1  = OpTypeStruct %u32 %f32
    %struct2  = OpTypeStruct %f32 %u32
    %struct5  = OpTypeStruct %f32
  )";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());

  ASSERT_EQ(5u, manager.NumTypes());
  ASSERT_EQ(0u, manager.NumForwardPointers());
  // Make sure we get ids correct.
  ASSERT_EQ("uint32", manager.GetType(3)->str());
  ASSERT_EQ("float32", manager.GetType(4)->str());

  // %struct1 with decoration on itself
  EXPECT_FALSE(manager.GetType(1)->decoration_empty());
  // %struct2 with decoration on its member
  EXPECT_FALSE(manager.GetType(2)->decoration_empty());
  EXPECT_TRUE(manager.GetType(3)->decoration_empty());
  EXPECT_TRUE(manager.GetType(4)->decoration_empty());
  // %struct5 has no decorations
  EXPECT_TRUE(manager.GetType(5)->decoration_empty());
}

TEST(TypeManager, BeginEndForEmptyModule) {
  const std::string text = "";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());
  ASSERT_EQ(0u, manager.NumTypes());
  ASSERT_EQ(0u, manager.NumForwardPointers());

  EXPECT_EQ(manager.begin(), manager.end());
}

TEST(TypeManager, BeginEnd) {
  const std::string text = R"(
    %void1   = OpTypeVoid
    %void2   = OpTypeVoid
    %bool    = OpTypeBool
    %u32     = OpTypeInt 32 0
    %f64     = OpTypeFloat 64
  )";
  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_1, nullptr, text);
  opt::analysis::TypeManager manager(nullptr, *context->module());
  ASSERT_EQ(5u, manager.NumTypes());
  ASSERT_EQ(0u, manager.NumForwardPointers());

  EXPECT_NE(manager.begin(), manager.end());
  for (const auto& t : manager) {
    switch (t.first) {
      case 1:
      case 2:
        EXPECT_EQ("void", t.second->str());
        break;
      case 3:
        EXPECT_EQ("bool", t.second->str());
        break;
      case 4:
        EXPECT_EQ("uint32", t.second->str());
        break;
      case 5:
        EXPECT_EQ("float64", t.second->str());
        break;
      default:
        EXPECT_TRUE(false && "unreachable");
        break;
    }
  }
}

}  // anonymous namespace
