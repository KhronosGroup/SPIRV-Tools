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

TEST(FactManagerTest, ConstantsAvailableViaUniforms) {
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

  fact_manager.AddUniformIntValueFact(
      32, true, {1}, MakeUniformBufferElementDescriptor(1, {2, 3}));
  fact_manager.AddUniformIntValueFact(
      32, true, {1}, MakeUniformBufferElementDescriptor(2, {1, 2, 3}));
  fact_manager.AddUniformIntValueFact(
      32, true, {1}, MakeUniformBufferElementDescriptor(3, {1, 0, 2, 3}));

  fact_manager.AddUniformIntValueFact(
      32, true, {buffer_int32_min[0]},
      MakeUniformBufferElementDescriptor(4, {2, 3}));
  fact_manager.AddUniformIntValueFact(
      32, true, {buffer_int32_min[0]},
      MakeUniformBufferElementDescriptor(5, {1, 2, 3}));

  fact_manager.AddUniformIntValueFact(
      64, true, {buffer_int64_max[0], buffer_int64_max[1]},
      MakeUniformBufferElementDescriptor(6, {1, 2, 3}));
  fact_manager.AddUniformIntValueFact(
      64, true, {buffer_int64_max[0], buffer_int64_max[1]},
      MakeUniformBufferElementDescriptor(7, {1, 1}));

  fact_manager.AddUniformIntValueFact(
      32, false, {1}, MakeUniformBufferElementDescriptor(8, {2, 3}));
  fact_manager.AddUniformIntValueFact(
      32, false, {1}, MakeUniformBufferElementDescriptor(9, {1, 2, 3}));
  fact_manager.AddUniformIntValueFact(
      32, false, {1}, MakeUniformBufferElementDescriptor(10, {1, 0, 2, 3}));

  fact_manager.AddUniformIntValueFact(
      64, false, {buffer_uint64_1[0], buffer_uint64_1[1]},
      MakeUniformBufferElementDescriptor(11, {0}));

  fact_manager.AddUniformIntValueFact(
      64, false, {buffer_uint64_max[0], buffer_uint64_max[0]},
      MakeUniformBufferElementDescriptor(12, {0, 0}));

  fact_manager.AddUniformIntValueFact(
      64, false, {buffer_uint64_max[0], buffer_uint64_max[0]},
      MakeUniformBufferElementDescriptor(13, {1, 0}));

  fact_manager.AddUniformFloatValueFact(
      32, {buffer_float_10[0]}, MakeUniformBufferElementDescriptor(14, {6}));

  fact_manager.AddUniformFloatValueFact(
      32, {buffer_float_10[0]}, MakeUniformBufferElementDescriptor(15, {7}));

  fact_manager.AddUniformFloatValueFact(
      32, {buffer_float_10[0]}, MakeUniformBufferElementDescriptor(16, {9, 9}));

  fact_manager.AddUniformFloatValueFact(
      64, {buffer_double_10[0], buffer_double_10[1]},
      MakeUniformBufferElementDescriptor(17, {9, 9, 1}));

  fact_manager.AddUniformFloatValueFact(
      64, {buffer_double_10[0], buffer_double_10[1]},
      MakeUniformBufferElementDescriptor(18, {9, 9, 2}));

  fact_manager.AddUniformFloatValueFact(
      64, {buffer_double_20[0], buffer_double_20[1]},
      MakeUniformBufferElementDescriptor(19, {0, 0, 0, 0, 0}));

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
    auto temp = MakeUniformBufferElementDescriptor(17, {9, 9, 1});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_10)[0]));
  }
  {
    auto temp = MakeUniformBufferElementDescriptor(18, {9, 9, 2});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_10)[1]));
  }
  const std::vector<protobufs::UniformBufferElementDescriptor>*
      descriptors_for_double_20 =
          fact_manager.GetUniformDescriptorsForConstant(*double_constants[1]);
  ASSERT_EQ(1, descriptors_for_double_20->size());
  {
    auto temp = MakeUniformBufferElementDescriptor(19, {0, 0, 0, 0, 0});
    ASSERT_TRUE(UniformBufferElementDescriptorEquals()(
        &temp, &(*descriptors_for_double_20)[0]));
  }

  auto constant_1 = fact_manager.GetConstantFromUniformDescriptor(
      MakeUniformBufferElementDescriptor(18, {9, 9, 2}));
  ASSERT_TRUE(constant_1 != nullptr);

  auto constant_2 = fact_manager.GetConstantFromUniformDescriptor(
      MakeUniformBufferElementDescriptor(19, {0, 0, 0, 0, 0}));
  ASSERT_TRUE(constant_2 != nullptr);

  ASSERT_TRUE(opt::analysis::ConstantEqual()(double_constants[0], constant_1));

  ASSERT_TRUE(opt::analysis::ConstantEqual()(double_constants[1], constant_2));
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
