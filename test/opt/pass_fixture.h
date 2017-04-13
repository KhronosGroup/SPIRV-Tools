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

#ifndef LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_
#define LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "opt/build_module.h"
#include "opt/make_unique.h"
#include "opt/pass_manager.h"
#include "opt/passes.h"
#include "spirv-tools/libspirv.hpp"

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
      : consumer_(nullptr),
        tools_(SPV_ENV_UNIVERSAL_1_1),
        manager_(new opt::PassManager()),
        assemble_options_(SpirvTools::kDefaultAssembleOption),
        disassemble_options_(SpirvTools::kDefaultDisassembleOption) {}

  // Runs the given |pass| on the binary assembled from the |original|.
  // Returns a tuple of the optimized binary and the boolean value returned 
  // from pass Process() function.
  std::tuple<std::vector<uint32_t>, opt::Pass::Status> OptimizeToBinary(
      opt::Pass* pass, const std::string& original, bool skip_nop) {
    std::unique_ptr<ir::Module> module = BuildModule(
        SPV_ENV_UNIVERSAL_1_1, consumer_, original, assemble_options_);
    EXPECT_NE(nullptr, module) << "Assembling failed for shader:\n"
                               << original << std::endl;
    if (!module) {
      return std::make_tuple(std::vector<uint32_t>(), 
          opt::Pass::Status::Failure);
    }

    const auto status = pass->Process(module.get());

    std::vector<uint32_t> binary;
    module->ToBinary(&binary, skip_nop);
    return std::make_tuple(binary, status);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |assembly|. Returns a tuple of the optimized binary and the boolean value 
  // from the pass Process() function.
  template <typename PassT, typename... Args>
  std::tuple<std::vector<uint32_t>, opt::Pass::Status> SinglePassRunToBinary(
      const std::string& assembly, bool skip_nop, Args&&... args) {
    auto pass = MakeUnique<PassT>(std::forward<Args>(args)...);
    pass->SetMessageConsumer(consumer_);
    return OptimizeToBinary(pass.get(), assembly, skip_nop);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |assembly|, disassembles the optimized binary. Returns a tuple of
  // disassembly string and the boolean value from the pass Process() function.
  template <typename PassT, typename... Args>
  std::tuple<std::string, opt::Pass::Status> SinglePassRunAndDisassemble(
      const std::string& assembly, bool skip_nop, Args&&... args) {
    std::vector<uint32_t> optimized_bin;
    auto status = opt::Pass::Status::SuccessWithoutChange;
    std::tie(optimized_bin, status) = SinglePassRunToBinary<PassT>(
        assembly, skip_nop, std::forward<Args>(args)...);
    std::string optimized_asm;
    EXPECT_TRUE(tools_.Disassemble(optimized_bin, &optimized_asm,
                                   disassemble_options_))
        << "Disassembling failed for shader:\n"
        << assembly << std::endl;
    return std::make_tuple(optimized_asm, status);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, and checks whether the optimized binary can be
  // disassembled to the |expected| assembly. Optionally will also validate
  // the optimized binary. This does *not* involve pass manager. Callers 
  // are suggested to use SCOPED_TRACE() for better messages.
  template <typename PassT, typename... Args>
  void SinglePassRunAndCheck(const std::string& original,
                             const std::string& expected, bool skip_nop,
                             bool do_validation, Args&&... args) {
    std::vector<uint32_t> optimized_bin;
    auto status = opt::Pass::Status::SuccessWithoutChange;
    std::tie(optimized_bin, status) = SinglePassRunToBinary<PassT>(
        original, skip_nop, std::forward<Args>(args)...);
    // Check whether the pass returns the correct modification indication.
    EXPECT_NE(opt::Pass::Status::Failure, status);
    EXPECT_EQ(original == expected,
              status == opt::Pass::Status::SuccessWithoutChange);
    if (do_validation) {
      spv_target_env target_env = SPV_ENV_UNIVERSAL_1_1;
      spv_context context = spvContextCreate(target_env);
      spv_diagnostic diagnostic = nullptr;
      spv_const_binary_t binary = {optimized_bin.data(),
          optimized_bin.size()};
      spv_result_t error = spvValidate(context, &binary, &diagnostic);
      EXPECT_EQ(error, 0);
      if (error != 0)
        spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
      spvContextDestroy(context);
    }
    std::string optimized_asm;
    EXPECT_TRUE(tools_.Disassemble(optimized_bin, &optimized_asm,
                                   disassemble_options_))
        << "Disassembling failed for shader:\n"
        << original << std::endl;
    EXPECT_EQ(expected, optimized_asm);
  }

  // Runs a single pass of class |PassT| on the binary assembled from the
  // |original| assembly, and checks whether the optimized binary can be
  // disassembled to the |expected| assembly. This does *not* involve pass
  // manager. Callers are suggested to use SCOPED_TRACE() for better messages.
  template <typename PassT, typename... Args>
  void SinglePassRunAndCheck(const std::string& original,
                             const std::string& expected, bool skip_nop,
                             Args&&... args) {
    SinglePassRunAndCheck<PassT>(original, expected, skip_nop, false,
      std::forward<Args>(args)...);
  }

  // Adds a pass to be run.
  template <typename PassT, typename... Args>
  void AddPass(Args&&... args) {
    manager_->AddPass<PassT>(std::forward<Args>(args)...);
  }

  // Renews the pass manager, including clearing all previously added passes.
  void RenewPassManger() {
    manager_.reset(new opt::PassManager());
    manager_->SetMessageConsumer(consumer_);
  }

  // Runs the passes added thus far using a pass manager on the binary assembled
  // from the |original| assembly, and checks whether the optimized binary can
  // be disassembled to the |expected| assembly. Callers are suggested to use
  // SCOPED_TRACE() for better messages.
  void RunAndCheck(const std::string& original, const std::string& expected) {
    assert(manager_->NumPasses());

    std::unique_ptr<ir::Module> module = BuildModule(
        SPV_ENV_UNIVERSAL_1_1, nullptr, original, assemble_options_);
    ASSERT_NE(nullptr, module);

    manager_->Run(module.get());

    std::vector<uint32_t> binary;
    module->ToBinary(&binary, /* skip_nop = */ false);

    std::string optimized;
    EXPECT_TRUE(tools_.Disassemble(binary, &optimized,
                                   disassemble_options_));
    EXPECT_EQ(expected, optimized);
  }

  void SetAssembleOptions(uint32_t assemble_options) {
    assemble_options_ = assemble_options;
  }

  void SetDisassembleOptions(uint32_t disassemble_options) {
    disassemble_options_ = disassemble_options;
  }

 private:
  MessageConsumer consumer_;  // Message consumer.
  SpirvTools tools_;  // An instance for calling SPIRV-Tools functionalities.
  std::unique_ptr<opt::PassManager> manager_;  // The pass manager.
  uint32_t assemble_options_;
  uint32_t disassemble_options_;
};

}  // namespace spvtools

#endif  // LIBSPIRV_TEST_OPT_PASS_FIXTURE_H_
