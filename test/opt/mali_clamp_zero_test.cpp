// Copyright (c) 2026 Lee Gao
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

#include <string>
#include <tuple>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using MaliClampZeroTest = PassTest<::testing::Test>;

size_t CountOccurrences(const std::string& text, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = text.find(needle); pos != std::string::npos;
       pos = text.find(needle, pos + 1)) {
    ++count;
  }
  return count;
}

const char* kClampZeroModule = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_bool = OpTypePointer Private %bool
   %cond_var = OpVariable %ptr_private_bool Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %float %cond %float_0 %float_1
          %r = OpExtInst %float %glsl FClamp %sel %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

TEST_F(MaliClampZeroTest, GivesConstantSelectedClampAnOpaqueZero) {
  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      kClampZeroModule, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  // The original bool load plus the new opaque float zero load.
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpLoad"));
}

TEST_F(MaliClampZeroTest, BitFieldInsertVariant) {
  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      kClampZeroModule, /* skip_nop = */ true, /* do_validation = */ true,
      /*rewrite_all=*/false, /*bitfield_insert=*/true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_NE(std::string::npos, std::get<0>(result).find("OpBitFieldInsert"));
}

TEST_F(MaliClampZeroTest, IntegerBitFieldInsertVariant) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
%ptr_private_bool = OpTypePointer Private %bool
   %cond_var = OpVariable %ptr_private_bool Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %uint %cond %uint_0 %uint_1
          %r = OpExtInst %uint %glsl UMin %sel %uint_0
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true,
      /*rewrite_all=*/false, /*bitfield_insert=*/true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountOccurrences(std::get<0>(result), "OpBitFieldInsert"));
}

TEST_F(MaliClampZeroTest, LeavesNonSelectedValue) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_float = OpTypePointer Private %float
        %x = OpVariable %ptr_private_float Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
        %v = OpLoad %float %x
          %r = OpExtInst %float %glsl FClamp %v %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliClampZeroTest, RewritesMinWithConstantSelectedValue) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_bool = OpTypePointer Private %bool
   %cond_var = OpVariable %ptr_private_bool Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %float %cond %float_0 %float_1
          %r = OpExtInst %float %glsl FMin %sel %float_0
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  // The bool load plus the opaque float zero load.
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpLoad"));
}

TEST_F(MaliClampZeroTest, RewritesVectorMin) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
     %v2bool = OpTypeVector %bool 2
     %true2 = OpConstantComposite %v2bool %true %true
       %uint = OpTypeInt 32 0
     %v2uint = OpTypeVector %uint 2
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
       %zero = OpConstantComposite %v2uint %uint_0 %uint_0
        %one = OpConstantComposite %v2uint %uint_1 %uint_1
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
        %sel = OpSelect %v2uint %true2 %zero %one
          %r = OpExtInst %v2uint %glsl UMin %sel %zero
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  // One opaque scalar load; one vector construction from it.
  EXPECT_EQ(1u, CountOccurrences(disassembly, "OpLoad"));
  EXPECT_EQ(1u, CountOccurrences(disassembly, "OpCompositeConstruct"));
}

TEST_F(MaliClampZeroTest, SharesOneOpaqueZeroBetweenClamps) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_bool = OpTypePointer Private %bool
   %cond_var = OpVariable %ptr_private_bool Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %float %cond %float_0 %float_1
         %r1 = OpExtInst %float %glsl FClamp %sel %float_0 %float_1
         %r2 = OpExtInst %float %glsl FClamp %sel %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  const std::string& disassembly = std::get<0>(result);
  // cond_var and one shared opaque zero variable; one load each.
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpVariable"));
  EXPECT_EQ(2u, CountOccurrences(disassembly, "OpLoad"));
}

TEST_F(MaliClampZeroTest, FollowsConstantStoreForValue) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_bool = OpTypePointer Private %bool
%ptr_private_float = OpTypePointer Private %float
   %cond_var = OpVariable %ptr_private_bool Private
        %xvar = OpVariable %ptr_private_float Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %float %cond %float_0 %float_1
               OpStore %xvar %sel
          %x = OpLoad %float %xvar
          %r = OpExtInst %float %glsl FClamp %x %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  // cond_var, xvar, and the new opaque zero load.
  EXPECT_EQ(3u, CountOccurrences(std::get<0>(result), "OpLoad"));
}

TEST_F(MaliClampZeroTest, LeavesValueWithAliasingCopyMemory) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_bool = OpTypePointer Private %bool
%ptr_private_float = OpTypePointer Private %float
   %cond_var = OpVariable %ptr_private_bool Private
        %xvar = OpVariable %ptr_private_float Private
        %yvar = OpVariable %ptr_private_float Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %float %cond %float_0 %float_1
               OpStore %xvar %sel
               OpCopyMemory %xvar %yvar
          %x = OpLoad %float %xvar
          %r = OpExtInst %float %glsl FClamp %x %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(MaliClampZeroTest, SixtyFourBitBitFieldInsertFallsBack) {
  const std::string text = R"(
               OpCapability Shader
               OpCapability Float64
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
     %double = OpTypeFloat 64
   %double_0 = OpConstant %double 0
   %double_1 = OpConstant %double 1
%ptr_private_bool = OpTypePointer Private %bool
   %cond_var = OpVariable %ptr_private_bool Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
               OpStore %cond_var %true
       %cond = OpLoad %bool %cond_var
        %sel = OpSelect %double %cond %double_0 %double_1
          %r = OpExtInst %double %glsl FClamp %sel %double_0 %double_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true,
      /*rewrite_all=*/false, /*bitfield_insert=*/true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  // 64-bit has no single-word OpBitFieldInsert path; it uses a variable.
  EXPECT_EQ(0u, CountOccurrences(std::get<0>(result), "OpBitFieldInsert"));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpLoad"));
}

TEST_F(MaliClampZeroTest, RewriteAllTakesUnselectedValue) {
  const std::string text = R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%ptr_private_float = OpTypePointer Private %float
        %x = OpVariable %ptr_private_float Private
       %glsl = OpExtInstImport "GLSL.std.450"
       %main = OpFunction %void None %fn
      %label = OpLabel
        %v = OpLoad %float %x
          %r = OpExtInst %float %glsl FClamp %v %float_0 %float_1
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true,
      /*rewrite_all=*/true);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountOccurrences(std::get<0>(result), "OpLoad"));
}

TEST_F(MaliClampZeroTest, LeavesNonZeroMinClampUnchanged) {
  const std::string text = R"(
               OpCapability Shader
        %ext = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
  %ptr_float = OpTypePointer Function %float
       %main = OpFunction %void None %fn
      %label = OpLabel
        %var = OpVariable %ptr_float Function
        %val = OpLoad %float %var
      %clamp = OpExtInst %float %ext FClamp %val %float_1 %float_2
               OpReturn
               OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<MaliClampZeroPass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
