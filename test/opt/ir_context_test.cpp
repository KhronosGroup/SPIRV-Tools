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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>

#include "opt/ir_context.h"
#include "opt/pass.h"
#include "pass_fixture.h"
#include "pass_utils.h"

namespace {

using namespace spvtools;
using ir::IRContext;
using Analysis = IRContext::Analysis;
using ::testing::Each;

class DummyPassPreservesNothing : public opt::Pass {
 public:
  DummyPassPreservesNothing(Status s) : opt::Pass(), status_to_return_(s) {}
  const char* name() const override { return "dummy-pass"; }
  Status Process(IRContext*) override { return status_to_return_; }
  Status status_to_return_;
};

class DummyPassPreservesAll : public opt::Pass {
 public:
  DummyPassPreservesAll(Status s) : opt::Pass(), status_to_return_(s) {}
  const char* name() const override { return "dummy-pass"; }
  Status Process(IRContext*) override { return status_to_return_; }
  Status status_to_return_;
  virtual Analysis GetPreservedAnalyses() override {
    return Analysis(IRContext::kAnalysisEnd - 1);
  }
};

class DummyPassPreservesFirst : public opt::Pass {
 public:
  DummyPassPreservesFirst(Status s) : opt::Pass(), status_to_return_(s) {}
  const char* name() const override { return "dummy-pass"; }
  Status Process(IRContext*) override { return status_to_return_; }
  Status status_to_return_;
  virtual Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisBegin;
  }
};

using IRContextTest = PassTest<::testing::Test>;

TEST_F(IRContextTest, IndividualValidAfterBuild) {
  std::unique_ptr<ir::Module> module(new ir::Module());
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    EXPECT_TRUE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllValidAfterBuild) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }
  EXPECT_TRUE(localContext.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, AllValidAfterPassNoChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }

  DummyPassPreservesNothing pass(opt::Pass::Status::SuccessWithoutChange);
  opt::Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithoutChange);
  EXPECT_TRUE(localContext.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, NoneValidAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesNothing pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllPreservedAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesAll pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_TRUE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, PreserveFirstOnlyAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    localContext.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesFirst pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&localContext);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  EXPECT_TRUE(localContext.AreAnalysesValid(IRContext::kAnalysisBegin));
  for (Analysis i = IRContext::kAnalysisBegin << 1; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(localContext.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, KillMemberName) {
  const std::string text = R"(
              OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource GLSL 430
               OpName %3 "stuff"
               OpMemberName %3 0 "refZ"
               OpMemberDecorate %3 0 Offset 0
               OpDecorate %3 Block
          %4 = OpTypeFloat 32
          %3 = OpTypeStruct %4
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %2 = OpFunction %5 None %6
          %7 = OpLabel
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<ir::IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);

  // Build the decoration manager.
  context->get_decoration_mgr();

  // Delete the OpTypeStruct.  Should delete the OpName, OpMemberName, and
  // OpMemberDecorate associated with it.
  context->KillDef(3);

  // Make sure all of the name are removed.
  for (auto& inst : context->debugs2()) {
    EXPECT_EQ(inst.opcode(), SpvOpNop);
  }

  // Make sure all of the decorations are removed.
  for (auto& inst : context->annotations()) {
    EXPECT_EQ(inst.opcode(), SpvOpNop);
  }
}

TEST_F(IRContextTest, TakeNextUniqueIdIncrementing) {
  const uint32_t NUM_TESTS = 1000;
  IRContext localContext(SPV_ENV_UNIVERSAL_1_2, nullptr);
  for (uint32_t i = 1; i < NUM_TESTS; ++i)
    EXPECT_EQ(i, localContext.TakeNextUniqueId());
}

}  // anonymous namespace
