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
  IRContext context(std::move(module), spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
    EXPECT_TRUE(context.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllValidAfterBuild) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext context(std::move(module), spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }
  EXPECT_TRUE(context.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, AllValidAfterPassNoChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext context(std::move(module), spvtools::MessageConsumer());

  Analysis built_analyses = IRContext::kAnalysisNone;
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
    built_analyses |= i;
  }

  DummyPassPreservesNothing pass(opt::Pass::Status::SuccessWithoutChange);
  opt::Pass::Status s = pass.Run(&context);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithoutChange);
  EXPECT_TRUE(context.AreAnalysesValid(built_analyses));
}

TEST_F(IRContextTest, NoneValidAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext context(std::move(module), spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesNothing pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&context);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(context.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllPreservedAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext context(std::move(module), spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesAll pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&context);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_TRUE(context.AreAnalysesValid(i));
  }
}

TEST_F(IRContextTest, AllPreserveFirstOnlyAfterPassWithChange) {
  std::unique_ptr<ir::Module> module = MakeUnique<ir::Module>();
  IRContext context(std::move(module), spvtools::MessageConsumer());

  for (Analysis i = IRContext::kAnalysisBegin; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    context.BuildInvalidAnalyses(i);
  }

  DummyPassPreservesAll pass(opt::Pass::Status::SuccessWithChange);
  opt::Pass::Status s = pass.Run(&context);
  EXPECT_EQ(s, opt::Pass::Status::SuccessWithChange);
  EXPECT_TRUE(context.AreAnalysesValid(IRContext::kAnalysisBegin));
  for (Analysis i = IRContext::kAnalysisBegin << 1; i < IRContext::kAnalysisEnd;
       i <<= 1) {
    EXPECT_FALSE(context.AreAnalysesValid(i));
  }
}
}  // anonymous namespace
