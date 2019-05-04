// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/uniform_buffer_element_descriptor.h"

#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

using opt::analysis::BoolConstant;
using opt::analysis::FloatConstant;
using opt::analysis::IntConstant;
using opt::analysis::ScalarConstant;

using opt::analysis::Bool;
using opt::analysis::Float;
using opt::analysis::Integer;
using opt::analysis::Type;

bool AddFactHelper(
    FactManager* fact_manager, opt::IRContext* context,
    std::vector<uint32_t>&& words,
    const protobufs::UniformBufferElementDescriptor& descriptor) {
  protobufs::ConstantUniformFact constant_uniform_fact;
  for (auto word : words) {
    constant_uniform_fact.add_constant_word(word);
  }
  *constant_uniform_fact.mutable_uniform_buffer_element_descriptor() =
      descriptor;
  protobufs::Fact fact;
  *fact.mutable_constant_uniform_fact() = constant_uniform_fact;
  return fact_manager->AddFact(fact, context);
}

TEST(FactManagerTest, ConstantsAvailableViaUniforms) {
  std::string shader = R"(
               OpCapability Shader
               OpCapability Int64
               OpCapability Float64
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
               OpName %4 "main"
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
         %10 = OpTypeInt 32 0
         %11 = OpTypeInt 32 1
         %12 = OpTypeInt 64 0
         %13 = OpTypeInt 64 1
         %15 = OpTypeFloat 32
         %16 = OpTypeFloat 64
         %17 = OpConstant %11 5
         %18 = OpConstant %11 20
         %19 = OpTypeVector %10 4
         %20 = OpConstant %11 6
         %21 = OpTypeVector %12 4
         %22 = OpConstant %11 10
         %23 = OpTypeVector %11 4

        %102 = OpTypeStruct %10 %10 %23
        %101 = OpTypePointer Uniform %102
        %100 = OpVariable %101 Uniform

        %203 = OpTypeArray %23 %17
        %202 = OpTypeArray %203 %18
        %201 = OpTypePointer Uniform %202
        %200 = OpVariable %201 Uniform

        %305 = OpTypeStruct %16 %16 %16 %11 %16
        %304 = OpTypeStruct %16 %16 %305
        %303 = OpTypeStruct %304
        %302 = OpTypeStruct %10 %303
        %301 = OpTypePointer Uniform %302
        %300 = OpVariable %301 Uniform

        %400 = OpVariable %101 Uniform

        %500 = OpVariable %201 Uniform

        %604 = OpTypeArray %13 %20
        %603 = OpTypeArray %604 %20
        %602 = OpTypeArray %603 %20
        %601 = OpTypePointer Uniform %602
        %600 = OpVariable %601 Uniform

        %703 = OpTypeArray %13 %20
        %702 = OpTypeArray %703 %20
        %701 = OpTypePointer Uniform %702
        %700 = OpVariable %701 Uniform

        %802 = OpTypeStruct %702 %602 %19 %202 %302
        %801 = OpTypePointer Uniform %802
        %800 = OpVariable %801 Uniform

        %902 = OpTypeStruct %702 %802 %19 %202 %302
        %901 = OpTypePointer Uniform %902
        %900 = OpVariable %901 Uniform

       %1003 = OpTypeStruct %802
       %1002 = OpTypeArray %1003 %20
       %1001 = OpTypePointer Uniform %1002
       %1000 = OpVariable %1001 Uniform

       %1101 = OpTypePointer Uniform %21
       %1100 = OpVariable %1101 Uniform

       %1202 = OpTypeArray %21 %20
       %1201 = OpTypePointer Uniform %1202
       %1200 = OpVariable %1201 Uniform

       %1302 = OpTypeArray %21 %20
       %1301 = OpTypePointer Uniform %1302
       %1300 = OpVariable %1301 Uniform

       %1402 = OpTypeArray %15 %22
       %1401 = OpTypePointer Uniform %1402
       %1400 = OpVariable %1401 Uniform

       %1501 = OpTypePointer Uniform %1402
       %1500 = OpVariable %1501 Uniform

       %1602 = OpTypeArray %1402 %22
       %1601 = OpTypePointer Uniform %1602
       %1600 = OpVariable %1601 Uniform

       %1704 = OpTypeStruct %16 %16 %16
       %1703 = OpTypeArray %1704 %22
       %1702 = OpTypeArray %1703 %22
       %1701 = OpTypePointer Uniform %1702
       %1700 = OpVariable %1701 Uniform

       %1800 = OpVariable %1701 Uniform

       %1906 = OpTypeStruct %16
       %1905 = OpTypeStruct %1906
       %1904 = OpTypeStruct %1905
       %1903 = OpTypeStruct %1904
       %1902 = OpTypeStruct %1903
       %1901 = OpTypePointer Uniform %1902
       %1900 = OpVariable %1901 Uniform

          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_3;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  uint32_t buffer_int32_min[1];
  uint32_t buffer_int64_1[2];
  uint32_t buffer_int64_max[2];
  uint32_t buffer_uint64_1[2];
  uint32_t buffer_uint64_max[2];
  uint32_t buffer_float_10[1];
  uint32_t buffer_float_20[1];
  uint32_t buffer_double_10[2];
  uint32_t buffer_double_20[2];

  {
    int32_t temp = std::numeric_limits<int32_t>::min();
    std::memcpy(&buffer_int32_min, &temp, sizeof(temp));
  }

  {
    int64_t temp = 1;
    std::memcpy(&buffer_int64_1, &temp, sizeof(temp));
  }

  {
    int64_t temp = std::numeric_limits<int64_t>::max();
    std::memcpy(&buffer_int64_max, &temp, sizeof(temp));
  }

  {
    uint64_t temp = 1;
    std::memcpy(&buffer_uint64_1, &temp, sizeof(temp));
  }

  {
    uint64_t temp = std::numeric_limits<uint64_t>::max();
    std::memcpy(&buffer_uint64_max, &temp, sizeof(temp));
  }

  {
    float temp = 10.0f;
    std::memcpy(&buffer_float_10, &temp, sizeof(float));
  }

  {
    float temp = 20.0f;
    std::memcpy(&buffer_float_20, &temp, sizeof(temp));
  }

  {
    double temp = 10.0;
    std::memcpy(&buffer_double_10, &temp, sizeof(temp));
  }

  {
    double temp = 20.0;
    std::memcpy(&buffer_double_20, &temp, sizeof(temp));
  }

  FactManager fact_manager;

  Integer type_int32_fst(32, true);
  Integer type_int32_snd(32, true);
  Integer type_int64_fst(64, true);
  Integer type_int64_snd(64, true);
  Integer type_uint32_fst(32, false);
  Integer type_uint32_snd(32, false);
  Integer type_uint64_fst(64, false);
  Integer type_uint64_snd(64, false);
  Float type_float_fst(32);
  Float type_float_snd(32);
  Float type_double_fst(64);
  Float type_double_snd(64);

  // Initially there should be no facts about uniforms.
  ASSERT_TRUE(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_uint32_snd)
          .empty());

  // 100[2][3] == int(1)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {1},
                            MakeUniformBufferElementDescriptor(100, {2, 3})));

  // 200[1][2][3] == int(1)
  ASSERT_TRUE(
      AddFactHelper(&fact_manager, context.get(), {1},
                    MakeUniformBufferElementDescriptor(200, {1, 2, 3})));

  // 300[1][0][2][3] == int(1)
  ASSERT_TRUE(
      AddFactHelper(&fact_manager, context.get(), {1},
                    MakeUniformBufferElementDescriptor(300, {1, 0, 2, 3})));

  // 400[2][3] = int32_min
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {buffer_int32_min[0]},
                            MakeUniformBufferElementDescriptor(400, {2, 3})));

  // 500[1][2][3] = int32_min
  ASSERT_TRUE(
      AddFactHelper(&fact_manager, context.get(), {buffer_int32_min[0]},
                    MakeUniformBufferElementDescriptor(500, {1, 2, 3})));

  // 600[1][2][3] = int64_max
  ASSERT_TRUE(AddFactHelper(
      &fact_manager, context.get(), {buffer_int64_max[0], buffer_int64_max[1]},
      MakeUniformBufferElementDescriptor(600, {1, 2, 3})));

  // 700[1][1] = int64_max
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(),
                            {buffer_int64_max[0], buffer_int64_max[1]},
                            MakeUniformBufferElementDescriptor(700, {1, 1})));

  // 800[2][3] = uint(1)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {1},
                            MakeUniformBufferElementDescriptor(800, {2, 3})));

  // 900[1][2][3] = uint(1)
  ASSERT_TRUE(
      AddFactHelper(&fact_manager, context.get(), {1},
                    MakeUniformBufferElementDescriptor(900, {1, 2, 3})));

  // 1000[1][0][2][3] = uint(1)
  ASSERT_TRUE(
      AddFactHelper(&fact_manager, context.get(), {1},
                    MakeUniformBufferElementDescriptor(1000, {1, 0, 2, 3})));

  // 1100[0] = uint64(1)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(),
                            {buffer_uint64_1[0], buffer_uint64_1[1]},
                            MakeUniformBufferElementDescriptor(1100, {0})));

  // 1200[0][0] = uint64_max
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(),
                            {buffer_uint64_max[0], buffer_uint64_max[1]},
                            MakeUniformBufferElementDescriptor(1200, {0, 0})));

  // 1300[1][0] = uint64_max
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(),
                            {buffer_uint64_max[0], buffer_uint64_max[1]},
                            MakeUniformBufferElementDescriptor(1300, {1, 0})));

  // 1400[6] = float(10.0)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {buffer_float_10[0]},
                            MakeUniformBufferElementDescriptor(1400, {6})));

  // 1500[7] = float(10.0)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {buffer_float_10[0]},
                            MakeUniformBufferElementDescriptor(1500, {7})));

  // 1600[9][9] = float(10.0)
  ASSERT_TRUE(AddFactHelper(&fact_manager, context.get(), {buffer_float_10[0]},
                            MakeUniformBufferElementDescriptor(1600, {9, 9})));

  // 1700[9][9][1] = double(10.0)
  ASSERT_TRUE(AddFactHelper(
      &fact_manager, context.get(), {buffer_double_10[0], buffer_double_10[1]},
      MakeUniformBufferElementDescriptor(1700, {9, 9, 1})));

  // 1800[9][9][2] = double(10.0)
  ASSERT_TRUE(AddFactHelper(
      &fact_manager, context.get(), {buffer_double_10[0], buffer_double_10[1]},
      MakeUniformBufferElementDescriptor(1800, {9, 9, 2})));

  // 1900[0][0][0][0][0] = double(20.0)
  ASSERT_TRUE(AddFactHelper(
      &fact_manager, context.get(), {buffer_double_20[0], buffer_double_20[1]},
      MakeUniformBufferElementDescriptor(1900, {0, 0, 0, 0, 0})));

  // The available constants should be the same regardless of which version of
  // each type we use.
  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_int32_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_int32_fst));

  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_int64_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_int64_fst));

  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_uint32_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_uint32_snd));

  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_uint64_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_uint64_fst));

  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_float_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_float_snd));

  ASSERT_EQ(
      fact_manager.GetConstantsAvailableFromUniformsForType(type_double_fst),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_double_snd));

  ASSERT_EQ(
      2, fact_manager.GetConstantsAvailableFromUniformsForType(type_int32_fst)
             .size());
  ASSERT_EQ(
      1, fact_manager.GetConstantsAvailableFromUniformsForType(type_int64_fst)
             .size());
  ASSERT_EQ(
      1, fact_manager.GetConstantsAvailableFromUniformsForType(type_uint32_fst)
             .size());
  ASSERT_EQ(
      2, fact_manager.GetConstantsAvailableFromUniformsForType(type_uint64_fst)
             .size());
  ASSERT_EQ(
      1, fact_manager.GetConstantsAvailableFromUniformsForType(type_float_fst)
             .size());
  ASSERT_EQ(
      2, fact_manager.GetConstantsAvailableFromUniformsForType(type_double_fst)
             .size());

  ASSERT_EQ(
      std::numeric_limits<int64_t>::max(),
      fact_manager.GetConstantsAvailableFromUniformsForType(type_int64_fst)[0]
          ->AsIntConstant()
          ->GetS64());
  ASSERT_EQ(1, fact_manager
                   .GetConstantsAvailableFromUniformsForType(type_uint32_fst)[0]
                   ->AsIntConstant()
                   ->GetU32());
  ASSERT_EQ(
      10.0f,
      fact_manager.GetConstantsAvailableFromUniformsForType(type_float_fst)[0]
          ->AsFloatConstant()
          ->GetFloat());
  const std::vector<const ScalarConstant*>& double_constants =
      fact_manager.GetConstantsAvailableFromUniformsForType(type_double_fst);
  ASSERT_EQ(10.0, double_constants[0]->AsFloatConstant()->GetDouble());
  ASSERT_EQ(20.0, double_constants[1]->AsFloatConstant()->GetDouble());

  const std::vector<protobufs::UniformBufferElementDescriptor>*
      descriptors_for_double_10 =
          fact_manager.GetUniformDescriptorsForConstant(*double_constants[0]);
  ASSERT_EQ(2, descriptors_for_double_10->size());
  {
    auto temp = MakeUniformBufferElementDescriptor(1700, {9, 9, 1});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_10)[0]));
  }
  {
    auto temp = MakeUniformBufferElementDescriptor(1800, {9, 9, 2});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_10)[1]));
  }
  const std::vector<protobufs::UniformBufferElementDescriptor>*
      descriptors_for_double_20 =
          fact_manager.GetUniformDescriptorsForConstant(*double_constants[1]);
  ASSERT_EQ(1, descriptors_for_double_20->size());
  {
    auto temp = MakeUniformBufferElementDescriptor(1900, {0, 0, 0, 0, 0});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_20)[0]));
  }

  auto constant_1 = fact_manager.GetConstantFromUniformDescriptor(
      MakeUniformBufferElementDescriptor(1800, {9, 9, 2}));
  ASSERT_TRUE(constant_1 != nullptr);

  auto constant_2 = fact_manager.GetConstantFromUniformDescriptor(
      MakeUniformBufferElementDescriptor(1900, {0, 0, 0, 0, 0}));
  ASSERT_TRUE(constant_2 != nullptr);

  ASSERT_TRUE(opt::analysis::ConstantEqual()(double_constants[0], constant_1));

  ASSERT_TRUE(opt::analysis::ConstantEqual()(double_constants[1], constant_2));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
