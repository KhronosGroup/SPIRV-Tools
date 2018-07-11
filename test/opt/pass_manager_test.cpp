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

#include "gmock/gmock.h"

#include <initializer_list>

#include "module_utils.h"
#include "opt/make_unique.h"
#include "pass_fixture.h"

namespace {

using namespace spvtools;
using spvtest::GetIdBound;
using ::testing::Eq;

// A null pass whose constructors accept arguments
class NullPassWithArgsToken : public opt::NullPassToken {
 public:
  NullPassWithArgsToken(uint32_t) {}
  NullPassWithArgsToken(std::string) {}
  NullPassWithArgsToken(const std::vector<int>&) {}
  NullPassWithArgsToken(const std::vector<int>&, uint32_t) {}

  const char* name() const override { return "null-with-args"; }
};

TEST(PassManager, Interface) {
  opt::PassManager manager;
  EXPECT_EQ(0u, manager.NumPasses());

  manager.AddPassToken<opt::StripDebugInfoPassToken>();
  EXPECT_EQ(1u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(0)->name());

  manager.AddPassToken(MakeUnique<opt::NullPassToken>());
  EXPECT_EQ(2u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(0)->name());
  EXPECT_STREQ("null", manager.GetPassToken(1)->name());

  manager.AddPassToken<opt::StripDebugInfoPassToken>();
  EXPECT_EQ(3u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(0)->name());
  EXPECT_STREQ("null", manager.GetPassToken(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(2)->name());

  manager.AddPassToken<NullPassWithArgsToken>(1u);
  manager.AddPassToken<NullPassWithArgsToken>("null pass args");
  manager.AddPassToken<NullPassWithArgsToken>(std::initializer_list<int>{1, 2});
  manager.AddPassToken<NullPassWithArgsToken>(std::initializer_list<int>{1, 2}, 3);
  EXPECT_EQ(7u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(0)->name());
  EXPECT_STREQ("null", manager.GetPassToken(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPassToken(2)->name());
  EXPECT_STREQ("null-with-args", manager.GetPassToken(3)->name());
  EXPECT_STREQ("null-with-args", manager.GetPassToken(4)->name());
  EXPECT_STREQ("null-with-args", manager.GetPassToken(5)->name());
  EXPECT_STREQ("null-with-args", manager.GetPassToken(6)->name());
}

// A pass that appends an OpNop instruction to the debug1 section.
class AppendOpNopPass : public opt::Pass {
 public:
  Status Process(opt::IRContext* irContext) override {
    irContext->AddDebug1Inst(MakeUnique<opt::Instruction>(irContext));
    return Status::SuccessWithChange;
  }
};

class AppendOpNopPassToken : public opt::PassToken {
 public:
  AppendOpNopPassToken() = default;
  ~AppendOpNopPassToken() = default;

  const char* name() const override { return "AppendOpNop"; }

  std::unique_ptr<opt::Pass> CreatePass() const override {
    return MakeUnique<AppendOpNopPass>();
  }
};

// A pass that appends specified number of OpNop instructions to the debug1
// section.
class AppendMultipleOpNopPass : public opt::Pass {
 public:
  explicit AppendMultipleOpNopPass(uint32_t num_nop) : num_nop_(num_nop) {}

  Status Process(opt::IRContext* irContext) override {
    for (uint32_t i = 0; i < num_nop_; i++) {
      irContext->AddDebug1Inst(MakeUnique<opt::Instruction>(irContext));
    }
    return Status::SuccessWithChange;
  }

 private:
  uint32_t num_nop_;
};

class AppendMultipleOpNopPassToken : public opt::PassToken {
 public:
  explicit AppendMultipleOpNopPassToken(uint32_t num_nop) : num_nop_(num_nop) {}
  ~AppendMultipleOpNopPassToken() = default;

  const char* name() const override { return "AppendOpNop"; }

  std::unique_ptr<opt::Pass> CreatePass() const override {
    return MakeUnique<AppendMultipleOpNopPass>(num_nop_);
  }

 private:
  uint32_t num_nop_;
};

// A pass that duplicates the last instruction in the debug1 section.
class DuplicateInstPass : public opt::Pass {
 public:
  Status Process(opt::IRContext* irContext) override {
    auto inst = MakeUnique<opt::Instruction>(
        *(--irContext->debug1_end())->Clone(irContext));
    irContext->AddDebug1Inst(std::move(inst));
    return Status::SuccessWithChange;
  }
};

class DuplicateInstPassToken : public opt::PassToken {
 public:
  DuplicateInstPassToken() = default;
  ~DuplicateInstPassToken() = default;

  const char* name() const override { return "DuplicateInst"; }

  std::unique_ptr<opt::Pass> CreatePass() const override {
    return MakeUnique<DuplicateInstPass>();
  }
};

using PassManagerTest = PassTest<::testing::Test>;

TEST_F(PassManagerTest, Run) {
  const std::string text = "OpMemoryModel Logical GLSL450\nOpSource ESSL 310\n";

  AddPassToken<AppendOpNopPassToken>();
  AddPassToken<AppendOpNopPassToken>();
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\n").c_str());

  RenewPassManger();
  AddPassToken<AppendOpNopPassToken>();
  AddPassToken<DuplicateInstPassToken>();
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\n").c_str());

  RenewPassManger();
  AddPassToken<DuplicateInstPassToken>();
  AddPassToken<AppendOpNopPassToken>();
  RunAndCheck(text.c_str(), (text + "OpSource ESSL 310\nOpNop\n").c_str());

  RenewPassManger();
  AddPassToken<AppendMultipleOpNopPassToken>(3);
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\nOpNop\n").c_str());
}

// A pass that appends an OpTypeVoid instruction that uses a given id.
class AppendTypeVoidInstPass : public opt::Pass {
 public:
  explicit AppendTypeVoidInstPass(uint32_t result_id) : result_id_(result_id) {}

  Status Process(opt::IRContext* irContext) override {
    auto inst = MakeUnique<opt::Instruction>(
        irContext, SpvOpTypeVoid, 0, result_id_, std::vector<opt::Operand>{});
    irContext->AddType(std::move(inst));
    return Status::SuccessWithChange;
  }

 private:
  uint32_t result_id_;
};

class AppendTypeVoidInstPassToken : public opt::PassToken {
 public:
  AppendTypeVoidInstPassToken(uint32_t result_id) : result_id_(result_id) {}
  ~AppendTypeVoidInstPassToken() = default;

  const char* name() const override { return "AppendTypeVoidInstPass"; }

  std::unique_ptr<opt::Pass> CreatePass() const override {
    return MakeUnique<AppendTypeVoidInstPass>(result_id_);
  }

 private:
  uint32_t result_id_;
};

TEST(PassManager, RecomputeIdBoundAutomatically) {
  opt::PassManager manager;
  std::unique_ptr<opt::Module> module(new opt::Module());
  opt::IRContext context(SPV_ENV_UNIVERSAL_1_2, std::move(module),
                         manager.consumer());
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));

  manager.Run(&context);
  manager.AddPassToken<AppendOpNopPassToken>();
  // With no ID changes, the ID bound does not change.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));

  // Now we force an Id of 100 to be used.
  manager.AddPassToken(MakeUnique<AppendTypeVoidInstPassToken>(100));
  EXPECT_THAT(GetIdBound(*context.module()), Eq(0u));
  manager.Run(&context);
  // The Id has been updated automatically, even though the pass
  // did not update it.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(101u));

  // Try one more time!
  manager.AddPassToken(MakeUnique<AppendTypeVoidInstPassToken>(200));
  manager.Run(&context);
  EXPECT_THAT(GetIdBound(*context.module()), Eq(201u));

  // Add another pass, but which uses a lower Id.
  manager.AddPassToken(MakeUnique<AppendTypeVoidInstPassToken>(10));
  manager.Run(&context);
  // The Id stays high.
  EXPECT_THAT(GetIdBound(*context.module()), Eq(201u));
}

}  // anonymous namespace
