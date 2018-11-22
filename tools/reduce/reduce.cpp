// Copyright (c) 2018 Google LLC
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

#include <cassert>
#include <cerrno>
#include <cstring>
#include <functional>

#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/reduce/operand_to_const_reduction_pass.h"
#include "source/reduce/operand_to_dominating_id_reduction_pass.h"
#include "source/reduce/reducer.h"
#include "source/reduce/remove_unreferenced_instruction_reduction_pass.h"
#include "source/spirv_reducer_options.h"
#include "source/util/make_unique.h"
#include "source/util/string_utils.h"
#include "spirv-tools/libspirv.hpp"
#include "tools/io.h"
#include "tools/util/cli_consumer.h"

using namespace spvtools::reduce;

namespace {

using ErrorOrInt = std::pair<std::string, int>;

// Check that the std::system function can actually be used.
bool CheckExecuteCommand() {
  int res = std::system(nullptr);
  return res != 0;
}

// Execute a command using the shell.
// Returns true if and only if the command's exit status was 0.
bool ExecuteCommand(const std::string& command) {
  errno = 0;
  int status = std::system(command.c_str());
  assert(errno == 0 && "failed to execute command");
  // The result returned by 'system' is implementation-defined, but is
  // usually the case that the returned value is 0 when the command's exit
  // code was 0.  We are assuming that here, and that's all we depend on.
  return status == 0;
}

// Status and actions to perform after parsing command-line arguments.
enum ReduceActions { REDUCE_CONTINUE, REDUCE_STOP };

struct ReduceStatus {
  ReduceActions action;
  int code;
};

void PrintUsage(const char* program) {
  // NOTE: Please maintain flags in lexicographical order.
  printf(
      R"(%s - Reduce a SPIR-V binary file with respect to a user-provided
              interestingness test.

USAGE: %s [options] <input> <interestingness-test>

The SPIR-V binary is read from <input>.

Whether a binary is interesting is determined by <interestingness-test>, which
is typically a script.

NOTE: The reducer is a work in progress.

Options (in lexicographical order):
  -h, --help
               Print this help.
  --step-limit
               32-bit unsigned integer specifying maximum number of
               steps the reducer will take before giving up.
  --version
               Display reducer version information.
)",
      program, program);
}

// Message consumer for this tool.  Used to emit diagnostics during
// initialization and setup. Note that |source| and |position| are irrelevant
// here because we are still not processing a SPIR-V input file.
void ReduceDiagnostic(spv_message_level_t level, const char* /*source*/,
                      const spv_position_t& /*position*/, const char* message) {
  if (level == SPV_MSG_ERROR) {
    fprintf(stderr, "error: ");
  }
  fprintf(stderr, "%s\n", message);
}

ReduceStatus ParseFlags(int argc, const char** argv, const char** in_file,
                        const char** interestingness_test,
                        spvtools::ReducerOptions* reducer_options) {
  uint32_t positional_arg_index = 0;

  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        spvtools::Logf(ReduceDiagnostic, SPV_MSG_INFO, nullptr, {}, "%s\n",
                       spvSoftwareVersionDetailsString());
        return {REDUCE_STOP, 0};
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        return {REDUCE_STOP, 0};
      } else if ('\0' == cur_arg[1]) {
        // We do not support reduction from standard input.  We could support
        // this if there was a compelling use case.
        PrintUsage(argv[0]);
        return {REDUCE_STOP, 0};
      } else if (0 == strncmp(cur_arg,
                              "--step-limit=", sizeof("--step-limit=") - 1)) {
        const auto split_flag = spvtools::utils::SplitFlagArgs(cur_arg);
        char* end = nullptr;
        errno = 0;
        const auto step_limit =
            static_cast<uint32_t>(strtol(split_flag.second.c_str(), &end, 10));
        assert(end != split_flag.second.c_str() && errno == 0);
        reducer_options->set_step_limit(step_limit);
      }
    } else if (positional_arg_index == 0) {
      // Input file name
      assert(!*in_file);
      *in_file = cur_arg;
      positional_arg_index++;
    } else if (positional_arg_index == 1) {
      assert(!*interestingness_test);
      *interestingness_test = cur_arg;
      positional_arg_index++;
    } else {
      spvtools::Error(ReduceDiagnostic, nullptr, {},
                      "Too many positional arguments specified");
      return {REDUCE_STOP, 1};
    }
  }

  if (!*in_file) {
    spvtools::Error(ReduceDiagnostic, nullptr, {}, "No input file specified");
    return {REDUCE_STOP, 1};
  }

  if (!*interestingness_test) {
    spvtools::Error(ReduceDiagnostic, nullptr, {},
                    "No interestingness test specified");
    return {REDUCE_STOP, 1};
  }

  return {REDUCE_CONTINUE, 0};
}

}  // namespace

const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_3;

int main(int argc, const char** argv) {
  const char* in_file = nullptr;
  const char* interestingness_test = nullptr;

  spv_target_env target_env = kDefaultEnvironment;
  spvtools::ReducerOptions reducer_options;

  ReduceStatus status =
      ParseFlags(argc, argv, &in_file, &interestingness_test, &reducer_options);

  if (status.action == REDUCE_STOP) {
    return status.code;
  }

  if (!CheckExecuteCommand()) {
    std::cerr << "could not find shell interpreter for executing a command"
              << std::endl;
    return 2;
  }

  Reducer reducer(target_env);

  reducer.SetInterestingnessFunction(
      [interestingness_test](std::vector<uint32_t> binary,
                             uint32_t reductions_applied) -> bool {
        std::stringstream ss;
        ss << "temp_" << std::setw(4) << std::setfill('0') << reductions_applied
           << ".spv";
        const auto spv_file = ss.str();
        const std::string command =
            std::string(interestingness_test) + " " + spv_file;
        auto write_file_succeeded =
            WriteFile(spv_file.c_str(), "wb", &binary[0], binary.size());
        (void)(write_file_succeeded);
        assert(write_file_succeeded);
        return ExecuteCommand(command);
      });

  reducer.AddReductionPass(
      spvtools::MakeUnique<OperandToConstReductionPass>(target_env));
  reducer.AddReductionPass(
      spvtools::MakeUnique<OperandToDominatingIdReductionPass>(target_env));
  reducer.AddReductionPass(
      spvtools::MakeUnique<RemoveUnreferencedInstructionReductionPass>(
          target_env));

  reducer.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);

  std::vector<uint32_t> binary_in;
  if (!ReadFile<uint32_t>(in_file, "rb", &binary_in)) {
    return 1;
  }

  std::vector<uint32_t> binary_out;
  const auto reduction_status =
      reducer.Run(std::move(binary_in), &binary_out, reducer_options);

  if (reduction_status ==
          Reducer::ReductionResultStatus::kInitialStateNotInteresting ||
      !WriteFile<uint32_t>("_reduced_final.spv", "wb", binary_out.data(),
                           binary_out.size())) {
    return 1;
  }

  return 0;
}
