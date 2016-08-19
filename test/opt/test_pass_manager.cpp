// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "pass_fixture.h"
#include "opt/make_unique.h"

namespace {

using namespace spvtools;

TEST(PassManager, Interface) {
  opt::PassManager manager;
  EXPECT_EQ(0u, manager.NumPasses());

  manager.AddPass<opt::StripDebugInfoPass>();
  EXPECT_EQ(1u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());

  manager.AddPass(MakeUnique<opt::NullPass>());
  EXPECT_EQ(2u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());

  manager.AddPass<opt::StripDebugInfoPass>();
  EXPECT_EQ(3u, manager.NumPasses());
  EXPECT_STREQ("strip-debug", manager.GetPass(0)->name());
  EXPECT_STREQ("null", manager.GetPass(1)->name());
  EXPECT_STREQ("strip-debug", manager.GetPass(2)->name());
}

// A pass that appends an OpNop instruction to the debug section.
class AppendOpNopPass : public opt::Pass {
  const char* name() const override { return "AppendOpNop"; }
  bool Process(ir::Module* module) override {
    auto inst = MakeUnique<ir::Instruction>();
    module->AddDebugInst(std::move(inst));
    return true;
  }
};

// A pass that duplicates the last instruction in the debug section.
class DuplicateInstPass : public opt::Pass {
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
}

}  // anonymous namespace
