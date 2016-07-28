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

#include "opt/pass_registry_impl.h"

#include <gtest/gtest.h>

#include "pass_fixture.h"

namespace spvtools {

namespace {
// A single-char-named empty pass for testing.
template <const char CHAR>
class SingleCharNamedSamplePass : public opt::Pass {
 public:
  SingleCharNamedSamplePass() : Pass(), c_(CHAR) {}
  const char* name() const override { return &c_; }
  bool Process(ir::Module*) override { return false; }

 private:
  const char c_;
};

// A sample pass with "a" as its name.
using SamplePassA = SingleCharNamedSamplePass<'a'>;
// A sample pass with "b" as its name.
using SamplePassB = SingleCharNamedSamplePass<'b'>;

class PassRegistryForTest : public opt::PassRegistryImpl {
 public:
  size_t RegistrySize() { return pass_create_info_.size(); }
};

}  // anonymous namespace

TEST(PassRegistry, RegisterSucceeded) {
  PassRegistryForTest registry;
  EXPECT_EQ(0u, registry.RegistrySize());
  const char* cmd_arg_pass_a = "cmd_arg_pass_a";
  EXPECT_EQ(true, registry.Register(cmd_arg_pass_a, &opt::MakePass<SamplePassA>,
                                    &opt::DeletePass<SamplePassA>));
  EXPECT_EQ(1u, registry.RegistrySize());
  const char* cmd_arg_pass_b = "cmd_arg_pass_b";
  EXPECT_EQ(true, registry.Register(cmd_arg_pass_b, &opt::MakePass<SamplePassA>,
                                    opt::DeletePass<SamplePassB>));
  EXPECT_EQ(2u, registry.RegistrySize());
  // Valid to have same pass but registered with differnt command arguments.
  const char* cmd_arg_pass_a_prime = "cmd_arg_pass_a_prime";
  EXPECT_EQ(true,
            registry.Register(cmd_arg_pass_a_prime, &opt::MakePass<SamplePassA>,
                              &opt::DeletePass<SamplePassA>));
  EXPECT_EQ(3u, registry.RegistrySize());
}

TEST(PassRegistry, RegisterFailed) {
  PassRegistryForTest registry;
  EXPECT_EQ(0u, registry.RegistrySize());
  const char* cmd_arg_pass_a = "cmd_arg_pass_a";
  EXPECT_EQ(true, registry.Register(cmd_arg_pass_a, &opt::MakePass<SamplePassA>,
                                    &opt::DeletePass<SamplePassA>));
  EXPECT_EQ(1u, registry.RegistrySize());
  // Invalid to register with same name more than once, even though the passes
  // to be registered are different.
  const char* cmd_arg_pass_b_same_as_a = "cmd_arg_pass_a";
  EXPECT_FALSE(registry.Register(cmd_arg_pass_b_same_as_a,
                                 opt::MakePass<SamplePassB>,
                                 opt::DeletePass<SamplePassB>));
  EXPECT_EQ(1u, registry.RegistrySize());
}

TEST(PassRegistry, GetPassSucceeded) {
  PassRegistryForTest registry;
  registry.Register("pass_a", &opt::MakePass<SamplePassA>,
                    &opt::DeletePass<SamplePassA>);
  registry.Register("pass_b", &opt::MakePass<SamplePassB>,
                    &opt::DeletePass<SamplePassB>);
  auto pass_a = registry.GetPass("pass_a");
  auto pass_b = registry.GetPass("pass_b");
  // The register should be able to find the passes.
  EXPECT_NE(nullptr, pass_a);
  EXPECT_NE(nullptr, pass_b);
  // The created pass instances' names should match.
  EXPECT_EQ('a', *pass_a->name());
  EXPECT_EQ('b', *pass_b->name());
}

TEST(PassRegistry, GetPassFailed) {
  PassRegistryForTest registry;
  EXPECT_EQ(0u, registry.RegistrySize());
  auto pass = registry.GetPass("no-registered pass");
  EXPECT_EQ(nullptr, pass);
}

}  // namespace spvtools
