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

#ifndef LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_
#define LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "opt/libspirv.hpp"
#include "opt/pass_manager.h"
#include "opt/passes.h"

namespace spvtools {

// Template class for testing passes. It contains some handy utility methods for
// running passes and checking results.
//
// To write value-Parameterized tests:
//   using ValueParamTest = PassTest<::testing::TestWithParam<std::string>>;
// To use as normal fixture:
//   using FixtureTest = PassTest<::testing::Test>;
template <typename TestT>
class PassTest : public TestT {
 public:
  PassTest()
      : tools_(SPV_ENV_UNIVERSAL_1_1), manager_(new opt::PassManager()) {}

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, and checks whether the optimized binary can be
  // disassembled to the |expected| assembly. This does *not* involve pass
  // manager. Callers are suggested to use SCOPED_TRACE() for better messages.
  template <typename PassT>
  void SinglePassRunAndCheck(const std::string& original,
                             const std::string& expected) {
    std::unique_ptr<ir::Module> module = tools_.BuildModule(original);
    ASSERT_NE(nullptr, module);

    const bool modified =
        std::unique_ptr<PassT>(new PassT)->Process(module.get());
    // Check whether the pass returns the correct modification indication.
    EXPECT_EQ(original != expected, modified);

    std::vector<uint32_t> binary;
    module->ToBinary(&binary, /* skip_nop = */ false);

    std::string optimized;
    EXPECT_EQ(SPV_SUCCESS, tools_.Disassemble(binary, &optimized));
    EXPECT_EQ(expected, optimized);
  }

  // Adds a pass to be run.
  template <typename PassT>
  void AddPass() {
    manager_->AddPass<PassT>();
  }

  // Renews the pass manager, including clearing all previously added passes.
  void RenewPassManger() { manager_.reset(new opt::PassManager()); }

  // Runs the passes added thus far using a pass manager on the binary assembled
  // from the |original| assembly, and checks whether the optimized binary can
  // be disassembled to the |expected| assembly. Callers are suggested to use
  // SCOPED_TRACE() for better messages.
  void RunAndCheck(const std::string& original, const std::string& expected) {
    assert(manager_->NumPasses());

    std::unique_ptr<ir::Module> module = tools_.BuildModule(original);
    ASSERT_NE(nullptr, module);

    manager_->Run(module.get());

    std::vector<uint32_t> binary;
    module->ToBinary(&binary, /* skip_nop = */ false);

    std::string optimized;
    EXPECT_EQ(SPV_SUCCESS, tools_.Disassemble(binary, &optimized));
    EXPECT_EQ(expected, optimized);
  }

 private:
  SpvTools tools_;  // An instance for calling SPIRV-Tools functionalities.
  std::unique_ptr<opt::PassManager> manager_;  // The pass manager.
};

}  // namespace spvtools

#endif  // LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_
