// Copyright (c) 2023 Google Inc.
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

#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "source/enum_set.h"
#include "source/enum_string_mapping.h"
#include "source/spirv_target_env.h"
#include "test/unit_spirv.h"

namespace spvtools {

// Useful to pretty-print the set when a test fails.
void PrintTo(const ExtensionSet& set, std::ostream* os) {
  *os << "{ ";
  for (auto extension : set) {
    *os << ExtensionToString(extension) << ", ";
  }
  *os << "}";
}

namespace {

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

class AssemblyGrammarTest : public ::testing::TestWithParam<spv_target_env> {
 public:
  inline spv_target_env target_env() const { return target_env_; }
  inline const AssemblyGrammar& grammar() const { return *grammar_; }
  inline const spv_context& context() const { return context_; }

 private:
  void SetUp() override {
    target_env_ = GetParam();
    context_ = spvContextCreate(target_env_);
    grammar_ = std::make_unique<AssemblyGrammar>(context_);
  }

  void TearDown() override { spvContextDestroy(context_); }

  spv_target_env target_env_;
  spv_context context_;
  std::unique_ptr<AssemblyGrammar> grammar_;
};

struct AssemblyGrammarExtensionTestData {
  spv::Capability capability;
  ExtensionSet extensions;
};

class AssemblyGrammarExtensionTest
    : public ::testing::TestWithParam<
          std::tuple<spv_target_env, spv::Capability, ExtensionSet>> {
 public:
  // This allows GTest to name the tests with something more useful than /0, /1
  // etc.
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    auto spirv_version = spvVersionForTargetEnv(std::get<0>(info.param));

    std::stringstream stream;
    stream << "spv" << SPV_SPIRV_VERSION_MAJOR_PART(spirv_version) << "_"
           << SPV_SPIRV_VERSION_MINOR_PART(spirv_version) << "_"
           << CapabilityToString(std::get<1>(info.param));
    return stream.str();
  }
};

TEST_P(AssemblyGrammarTest, ReportsIsValid) {
  EXPECT_TRUE(grammar().isValid());
}

TEST_P(AssemblyGrammarTest, ReportsCorrectTargetEnv) {
  EXPECT_EQ(grammar().target_env(), target_env());
}

INSTANTIATE_TEST_SUITE_P(AssemblyGrammarTestSuite, AssemblyGrammarTest,
                         ValuesIn(spvtest::AllTargetEnvironments()));

TEST_P(AssemblyGrammarExtensionTest, CheckExtensionDeclaredForCapability) {
  const spv_target_env target_env = std::get<0>(GetParam());
  spv_context context = spvContextCreate(target_env);
  AssemblyGrammar grammar = AssemblyGrammar(context);

  ExtensionSet result = grammar.getExtensionsDeclaring(std::get<1>(GetParam()));
  EXPECT_EQ(result, std::get<2>(GetParam()));

  spvContextDestroy(context);
}

INSTANTIATE_TEST_SUITE_P(
    AssemblyGrammarExtensionTestSuite, AssemblyGrammarExtensionTest,
    ValuesIn(
        std::vector<std::tuple<spv_target_env, spv::Capability, ExtensionSet>>{
            // clang-format off
            {SPV_ENV_UNIVERSAL_1_0, spv::Capability::Matrix,  {}},
            {SPV_ENV_UNIVERSAL_1_0, spv::Capability::Shader,  {}},
            {SPV_ENV_UNIVERSAL_1_0, spv::Capability::Float16, {}},

            // Those capabilities were brought into core.
            // Extension requirement depend on the SPIRV version.
            {SPV_ENV_UNIVERSAL_1_0, spv::Capability::MultiView,      { kSPV_KHR_multiview }},
            {SPV_ENV_VULKAN_1_3,    spv::Capability::MultiView,      {}},
            {SPV_ENV_VULKAN_1_0,    spv::Capability::DrawParameters, { kSPV_KHR_shader_draw_parameters }},
            {SPV_ENV_VULKAN_1_3,    spv::Capability::DrawParameters, { }},

            // This capability was added with Spirv 1.1.
            // Not returning any extension is valid in both cases.
            {SPV_ENV_UNIVERSAL_1_0,    spv::Capability::SubgroupDispatch, {}},
            {SPV_ENV_UNIVERSAL_1_1,    spv::Capability::SubgroupDispatch, {}},

            // Groups can be used with SPV_AMD_shader_ballot if some opcodes
            // are used, but it is also a core capability.
            {SPV_ENV_UNIVERSAL_1_0, spv::Capability::Groups,  {}},

            // This capability implies RayQueryKHR & RayTracingKHR, each depending on an extension.
            {SPV_ENV_VULKAN_1_3, spv::Capability::RayTraversalPrimitiveCullingKHR,  { kSPV_KHR_ray_query, kSPV_KHR_ray_tracing }},

            // PerViewAttributesNV implies MultiView. But the extension is enough to allow the
            // MultiView capability, even in 1.0 (No need for SPV_KHR_multiview).
            {SPV_ENV_VULKAN_1_0, spv::Capability::PerViewAttributesNV,  { kSPV_NVX_multiview_per_view_attributes }},
            {SPV_ENV_VULKAN_1_3, spv::Capability::PerViewAttributesNV,  { kSPV_NVX_multiview_per_view_attributes }},

            // This capability can be enabled by either the core, of the vendor extension.
            {SPV_ENV_VULKAN_1_3, spv::Capability::FragmentBarycentricKHR,
              { kSPV_NV_fragment_shader_barycentric, kSPV_KHR_fragment_shader_barycentric }},

            // clang-format on
        }),
    AssemblyGrammarExtensionTest::PrintToStringParamName);

}  // namespace
}  // namespace spvtools
