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

// A null pass whose construtors accept arguments
class NullPassWithArgs : public opt::NullPass {
 public:
  NullPassWithArgs(const MessageConsumer& c, uint32_t) : NullPass(c) {}
  NullPassWithArgs(const MessageConsumer& c, std::string) : NullPass(c) {}
  NullPassWithArgs(const MessageConsumer& c, const std::vector<int>&)
      : NullPass(c) {}
  NullPassWithArgs(const MessageConsumer& c, const std::vector<int>&, uint32_t)
      : NullPass(c) {}

  const char* name() const override { return "null-with-args"; }
};

TEST(PassManager, Interface) {
  opt::PassManager manager(IgnoreMessage);
  EXPECT_EQ(0u, manager.NumPasses());

  manager.AddPass<opt::StripDebugInfoPass>();
  EXPECT_EQ(1u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());

  manager.AddPass(MakeUnique<opt::NullPass>(IgnoreMessage));
  EXPECT_EQ(2u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());

  manager.AddPass<opt::StripDebugInfoPass>();
  EXPECT_EQ(3u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPass(2)->name());

  manager.AddPass<NullPassWithArgs>(1u);
  manager.AddPass<NullPassWithArgs>("null pass args");
  manager.AddPass<NullPassWithArgs>(std::initializer_list<int>{1, 2});
  manager.AddPass<NullPassWithArgs>(std::initializer_list<int>{1, 2}, 3);
  EXPECT_EQ(7u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPass(2)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(3)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(4)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(5)->name());
  EXPECT_STREQ("null-with-args", manager.GetPass(6)->name());
}

// A pass that appends an OpNop instruction to the debug section.
class AppendOpNopPass : public opt::Pass {
 public:
  explicit AppendOpNopPass(const MessageConsumer& c) : opt::Pass(c) {}

  const char* name() const override { return "AppendOpNop"; }
  bool Process(ir::Module* module) override {
    auto inst = MakeUnique<ir::Instruction>();
    module->AddDebugInst(std::move(inst));
    return true;
  }
};

// A pass that appends specified number of OpNop instructions to the debug
// section.
class AppendMultipleOpNopPass : public opt::Pass {
 public:
  AppendMultipleOpNopPass(const MessageConsumer& c, uint32_t num_nop)
      : opt::Pass(c), num_nop_(num_nop) {}

  const char* name() const override { return "AppendOpNop"; }
  bool Process(ir::Module* module) override {
    for (uint32_t i = 0; i < num_nop_; i++) {
      auto inst = MakeUnique<ir::Instruction>();
      module->AddDebugInst(std::move(inst));
    }
    return true;
  }

 private:
  uint32_t num_nop_;
};

// A pass that duplicates the last instruction in the debug section.
class DuplicateInstPass : public opt::Pass {
 public:
  explicit DuplicateInstPass(const MessageConsumer& c) : opt::Pass(c) {}

  const char* name() const override { return "DuplicateInst"; }
  bool Process(ir::Module* module) override {
    auto inst = MakeUnique<ir::Instruction>(*(--module->debug_end()));
    module->AddDebugInst(std::move(inst));
    return true;
  }
};

using PassManagerTest = PassTest<::testing::Test>;

TEST_F(PassManagerTest, Run) {
  const std::string text = "OpMemoryModel Logical GLSL450\nOpSource ESSL 310\n";

  AddPass<AppendOpNopPass>();
  AddPass<AppendOpNopPass>();
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\n").c_str());

  RenewPassManger();
  AddPass<AppendOpNopPass>();
  AddPass<DuplicateInstPass>();
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\n").c_str());

  RenewPassManger();
  AddPass<DuplicateInstPass>();
  AddPass<AppendOpNopPass>();
  RunAndCheck(text.c_str(), (text + "OpSource ESSL 310\nOpNop\n").c_str());

  RenewPassManger();
  AddPass<AppendMultipleOpNopPass>(3);
  RunAndCheck(text.c_str(), (text + "OpNop\nOpNop\nOpNop\n").c_str());
}

// A pass that appends an OpTypeVoid instruction that uses a given id.
class AppendTypeVoidInstPass : public opt::Pass {
 public:
  AppendTypeVoidInstPass(const MessageConsumer& c, uint32_t result_id)
      : opt::Pass(c), result_id_(result_id) {}

  const char* name() const override { return "AppendTypeVoidInstPass"; }
  bool Process(ir::Module* module) override {
    auto inst = MakeUnique<ir::Instruction>(SpvOpTypeVoid, 0, result_id_,
                                            std::vector<ir::Operand>{});
    module->AddType(std::move(inst));
    return true;
  }

 private:
  uint32_t result_id_;
};

TEST(PassManager, RecomputeIdBoundAutomatically) {
  ir::Module module;
  EXPECT_THAT(GetIdBound(module), Eq(0u));

  opt::PassManager manager(IgnoreMessage);
  manager.Run(&module);
  manager.AddPass<AppendOpNopPass>();
  // With no ID changes, the ID bound does not change.
  EXPECT_THAT(GetIdBound(module), Eq(0u));

  // Now we force an Id of 100 to be used.
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(IgnoreMessage, 100));
  EXPECT_THAT(GetIdBound(module), Eq(0u));
  manager.Run(&module);
  // The Id has been updated automatically, even though the pass
  // did not update it.
  EXPECT_THAT(GetIdBound(module), Eq(101u));

  // Try one more time!
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(IgnoreMessage, 200));
  manager.Run(&module);
  EXPECT_THAT(GetIdBound(module), Eq(201u));

  // Add another pass, but which uses a lower Id.
  manager.AddPass(MakeUnique<AppendTypeVoidInstPass>(IgnoreMessage, 10));
  manager.Run(&module);
  // The Id stays high.
  EXPECT_THAT(GetIdBound(module), Eq(201u));
}

}  // anonymous namespace
