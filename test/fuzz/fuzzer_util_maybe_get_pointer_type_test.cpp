// Copyright (c) 2021 Shiyu Liu
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

#include "gtest/gtest.h"
#include "source/fuzz/fuzzer_util.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(FuzzerUtilMaybeGetPointerTypeTest, BasicTest) {
  std::string shader = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %92 %52 %53
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpDecorate %92 BuiltIn FragCoord
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %7 = OpTypeFloat 32
          %8 = OpTypeStruct %6 %7
          %9 = OpTypePointer Function %8
         %10 = OpTypeFunction %6 %9
         %14 = OpConstant %6 0
         %15 = OpTypePointer Function %6
         %51 = OpTypePointer Private %6
         %21 = OpConstant %6 2
         %23 = OpConstant %6 1
         %24 = OpConstant %7 1
         %25 = OpTypePointer Function %7
         %50 = OpTypePointer Private %7
         %34 = OpTypeBool
         %35 = OpConstantFalse %34
         %60 = OpConstantNull %50
         %61 = OpUndef %51
         %52 = OpVariable %50 Private
         %53 = OpVariable %51 Private
         %80 = OpConstantComposite %8 %21 %24
         %90 = OpTypeVector %7 4
         %91 = OpTypePointer Input %90
         %92 = OpVariable %91 Input
         %93 = OpConstantComposite %90 %24 %24 %24 %24
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %20 = OpVariable %9 Function
         %27 = OpVariable %9 Function
         %22 = OpAccessChain %15 %20 %14
         %44 = OpCopyObject %9 %20
         %26 = OpAccessChain %25 %20 %23
         %29 = OpFunctionCall %6 %12 %27
         %30 = OpAccessChain %15 %20 %14
         %45 = OpCopyObject %15 %30
         %81 = OpCopyObject %9 %27
         %33 = OpAccessChain %15 %20 %14
               OpSelectionMerge %37 None
               OpBranchConditional %35 %36 %37
         %36 = OpLabel
         %38 = OpAccessChain %15 %20 %14
         %40 = OpAccessChain %15 %20 %14
         %43 = OpAccessChain %15 %20 %14
         %82 = OpCopyObject %9 %27
               OpBranch %37
         %37 = OpLabel
               OpReturn
               OpFunctionEnd
         %12 = OpFunction %6 None %10
         %11 = OpFunctionParameter %9
         %13 = OpLabel
         %46 = OpCopyObject %9 %11
         %16 = OpAccessChain %15 %11 %14
         %95 = OpCopyObject %8 %80
               OpReturnValue %21
        %100 = OpLabel
               OpUnreachable
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const std::unique_ptr<opt::IRContext> context =
      BuildModule(env, consumer, shader, kFuzzAssembleOption);
  spvtools::ValidatorOptions validator_options;
  ASSERT_TRUE(fuzzerutil::IsValidAndWellFormed(context.get(), validator_options,
                                               kConsoleMessageConsumer));
  TransformationContext transformation_context(
      MakeUnique<FactManager>(context.get()), validator_options);

  opt::IRContext* ir_context = context.get();
  uint32_t private_storage_class = SpvStorageClassPrivate;
  uint32_t function_storage_class = SpvStorageClassFunction;
  uint32_t input_storage_class = SpvStorageClassInput;

  // A valid pointer must have the correct |pointee_type_id| and |storageClass|.
  // A function type pointer with id = 9 and pointee type id 8 should be found.
  ASSERT_EQ(9, fuzzerutil::MaybeGetPointerType(
                   ir_context, 8,
                   static_cast<SpvStorageClass>(function_storage_class)));
  // A function type pointer with id = 15 and pointee type id 6 should be found.
  ASSERT_EQ(15, fuzzerutil::MaybeGetPointerType(
                    ir_context, 6,
                    static_cast<SpvStorageClass>(function_storage_class)));
  // A function type pointer with id = 25 and pointee type id 7 should be found.
  ASSERT_EQ(25, fuzzerutil::MaybeGetPointerType(
                    ir_context, 7,
                    static_cast<SpvStorageClass>(function_storage_class)));

  // A private type pointer with id=51 and pointee type id 6 should be found.
  ASSERT_EQ(51, fuzzerutil::MaybeGetPointerType(
                    ir_context, 6,
                    static_cast<SpvStorageClass>(private_storage_class)));
  // A function pointer with id=50 and pointee type id 7 should be found.
  ASSERT_EQ(50, fuzzerutil::MaybeGetPointerType(
                    ir_context, 7,
                    static_cast<SpvStorageClass>(private_storage_class)));

  // A input type pointer with id=91 and pointee type id 90 should be found.
  ASSERT_EQ(91, fuzzerutil::MaybeGetPointerType(
                    ir_context, 90,
                    static_cast<SpvStorageClass>(input_storage_class)));

  // A pointer with id=91 and pointee type 90 exisits, but the type should be
  // input.
  ASSERT_EQ(0, fuzzerutil::MaybeGetPointerType(
                   ir_context, 90,
                   static_cast<SpvStorageClass>(function_storage_class)));
  // A input type pointer with id=91 exists but the pointee id should be 90.
  ASSERT_EQ(0, fuzzerutil::MaybeGetPointerType(
                   ir_context, 89,
                   static_cast<SpvStorageClass>(input_storage_class)));
  // A input type pointer with pointee id 90 exists but result id of the pointer
  // should be 91.
  ASSERT_FALSE(fuzzerutil::MaybeGetPointerType(
                   ir_context, 90,
                   static_cast<SpvStorageClass>(input_storage_class)) == 58);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
