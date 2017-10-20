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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "opt/set_spec_constant_default_value_pass.h"
#include "spirv-tools/optimizer.hpp"

#include "message.h"
#include "tools/io.h"

using namespace spvtools;

namespace {

// Status and actions to perform after parsing command-line arguments.
enum OptActions { OPT_CONTINUE, OPT_STOP };

struct OptStatus {
  OptActions action;
  int code;
};

std::string GetListOfPassesAsString(const spvtools::Optimizer& optimizer) {
  std::stringstream ss;
  for (const auto& name : optimizer.GetPassNames()) {
    ss << "\n\t\t" << name;
  }
  return ss.str();
}

std::string GetOptimizationPasses() {
  spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_2);
  optimizer.RegisterPerformancePasses();
  return GetListOfPassesAsString(optimizer);
}

std::string GetSizePasses() {
  spvtools::Optimizer optimizer(SPV_ENV_UNIVERSAL_1_2);
  optimizer.RegisterSizePasses();
  return GetListOfPassesAsString(optimizer);
}

void PrintUsage(const char* program) {
  printf(
      R"(%s - Optimize a SPIR-V binary file.

USAGE: %s [options] [<input>] -o <output>

The SPIR-V binary is read from <input>. If no file is specified,
or if <input> is "-", then the binary is read from standard input.
if <output> is "-", then the optimized output is written to
standard output.

NOTE: The optimizer is a work in progress.

Options:
  --strip-debug
               Remove all debug instructions.
  --freeze-spec-const
               Freeze the values of specialization constants to their default
               values.
  --eliminate-dead-const
               Eliminate dead constants.
  --fold-spec-const-op-composite
               Fold the spec constants defined by OpSpecConstantOp or
               OpSpecConstantComposite instructions to front-end constants
               when possible.
  --set-spec-const-default-value "<spec id>:<default value> ..."
               Set the default values of the specialization constants with
               <spec id>:<default value> pairs specified in a double-quoted
               string. <spec id>:<default value> pairs must be separated by
               blank spaces, and in each pair, spec id and default value must
               be separated with colon ':' without any blank spaces in between.
               e.g.: --set-spec-const-default-value "1:100 2:400"
  --unify-const
               Remove the duplicated constants.
  --flatten-decorations
               Replace decoration groups with repeated OpDecorate and
               OpMemberDecorate instructions.
  --compact-ids
               Remap result ids to a compact range starting from %%1 and without
               any gaps.
  --cfg-cleanup
               Cleanup the control flow graph. This will remove any unnecessary
               code from the CFG like unreachable code. Performed on entry
               point call tree functions and exported functions.
  --inline-entry-points-exhaustive
               Exhaustively inline all function calls in entry point call tree
               functions. Currently does not inline calls to functions with
               early return in a loop.
  --convert-local-access-chains
               Convert constant index access chain loads/stores into
               equivalent load/stores with inserts and extracts. Performed
               on function scope variables referenced only with load, store,
               and constant index access chains in entry point call tree
               functions.
  --eliminate-common-uniform
               Perform load/load elimination for duplicate uniform values.
               Converts any constant index access chain uniform loads into
               its equivalent load and extract. Some loads will be moved
               to facilitate sharing. Performed only on entry point
               call tree functions.
  --eliminate-local-single-block
               Perform single-block store/load and load/load elimination.
               Performed only on function scope variables in entry point
               call tree functions.
  --eliminate-local-single-store
               Replace stores and loads of function scope variables that are
               only stored once. Performed on variables referenceed only with
               loads and stores. Performed only on entry point call tree
               functions.
  --eliminate-local-multi-store
               Replace stores and loads of function scope variables that are
               stored multiple times. Performed on variables referenceed only
               with loads and stores. Performed only on entry point call tree
               functions.
  --eliminate-insert-extract
               Replace extract from a sequence of inserts with the
               corresponding value. Performed only on entry point call tree
               functions.
  --eliminate-dead-code-aggressive
               Delete instructions which do not contribute to a function's
               output. Performed only on entry point call tree functions.
  --eliminate-dead-branches
               Convert conditional branches with constant condition to the
               indicated unconditional brranch. Delete all resulting dead
               code. Performed only on entry point call tree functions.
  --eliminate-dead-functions
               Deletes functions that cannot be reached from entry points or
               exported functions.
  --merge-blocks
               Join two blocks into a single block if the second has the
               first as its only predecessor. Performed only on entry point
               call tree functions.
  --strength-reduction
               Replaces instructions with equivalent and less expensive ones.
  --eliminate-dead-variables
               Deletes module scope variables that are not referenced.
  -O
               Optimize for performance. Apply a sequence of transformations
               in an attempt to improve the performance of the generated
               code. For this version of the optimizer, this flag is equivalent
               to specifying the following optimization code names:
               %s
  -Os
               Optimize for size. Apply a sequence of transformations in an
               attempt to minimize the size of the generated code. For this
               version of the optimizer, this flag is equivalent to specifying
               the following optimization code names:
               %s

               NOTE: The specific transformations done by -O and -Os change
                     from release to release.
  -Oconfig=<file>
               Apply the sequence of transformations indicated in <file>.
               This file contains a sequence of strings separated by whitespace
               (tabs, newlines or blanks). Each string is one of the flags
               accepted by spirv-opt. Optimizations will be applied in the
               sequence they appear in the file. This is equivalent to
               specifying all the flags on the command line. For example,
               given the file opts.cfg with the content:

                --inline-entry-points-exhaustive
                --eliminate-dead-code-aggressive

               The following two invocations to spirv-opt are equivalent:

               $ spirv-opt -Oconfig=opts.cfg program.spv

               $ spirv-opt --inline-entry-points-exhaustive \
                    --eliminate-dead-code-aggressive program.spv

               Lines starting with the character '#' in the configuration
               file indicate a comment and will be ignored.

               The -O, -Os, and -Oconfig flags act as macros. Using one of them
               is equivalent to explicitly inserting the underlying flags at
               that position in the command line. For example, the invocation
               'spirv-opt --merge-blocks -O ...' applies the transformation
               --merge-blocks followed by all the transformations implied by
               -O.
  -h, --help
               Print this help.
  --version
               Display optimizer version information.
)",
      program, program, GetOptimizationPasses().c_str(),
      GetSizePasses().c_str());
}

// Reads command-line flags  the file specified in |oconfig_flag|. This string
// is assumed to have the form "-Oconfig=FILENAME". This function parses the
// string and extracts the file name after the '=' sign.
//
// Flags found in |FILENAME| are pushed at the end of the vector |file_flags|.
//
// This function returns true on success, false on failure.
bool ReadFlagsFromFile(const char* oconfig_flag,
                       std::vector<std::string>* file_flags) {
  const char* fname = strchr(oconfig_flag, '=');
  if (fname == nullptr || fname[0] != '=') {
    fprintf(stderr, "error: Invalid -Oconfig flag %s\n", oconfig_flag);
    return false;
  }
  fname++;

  std::ifstream input_file;
  input_file.open(fname);
  if (input_file.fail()) {
    fprintf(stderr, "error: Could not open file '%s'\n", fname);
    return false;
  }

  while (!input_file.eof()) {
    std::string flag;
    input_file >> flag;
    if (flag.length() > 0 && flag[0] != '#') {
      file_flags->push_back(flag);
    }
  }

  return true;
}

OptStatus ParseFlags(int argc, const char** argv, Optimizer* optimizer,
                     const char** in_file, const char** out_file);

// Parses and handles the -Oconfig flag. |prog_name| contains the name of
// the spirv-opt binary (used to build a new argv vector for the recursive
// invocation to ParseFlags). |opt_flag| contains the -Oconfig=FILENAME flag.
// |optimizer|, |in_file| and |out_file| are as in ParseFlags.
//
// This returns the same OptStatus instance returned by ParseFlags.
OptStatus ParseOconfigFlag(const char* prog_name, const char* opt_flag,
                           Optimizer* optimizer, const char** in_file,
                           const char** out_file) {
  std::vector<std::string> flags;
  flags.push_back(prog_name);

  std::vector<std::string> file_flags;
  if (!ReadFlagsFromFile(opt_flag, &file_flags)) {
    fprintf(stderr,
            "error: Could not read optimizer flags from configuration file\n");
    return {OPT_STOP, 1};
  }
  flags.insert(flags.end(), file_flags.begin(), file_flags.end());

  const char** new_argv = new const char*[flags.size()];
  for (size_t i = 0; i < flags.size(); i++) {
    if (flags[i].find("-Oconfig=") != std::string::npos) {
      fprintf(stderr,
              "error: Flag -Oconfig= may not be used inside the configuration "
              "file\n");
      return {OPT_STOP, 1};
    }
    new_argv[i] = flags[i].c_str();
  }

  return ParseFlags(static_cast<int>(flags.size()), new_argv, optimizer,
                    in_file, out_file);
}

// Parses command-line flags. |argc| contains the number of command-line flags.
// |argv| points to an array of strings holding the flags. |optimizer| is the
// Optimizer instance used to optimize the program.
//
// On return, this function stores the name of the input program in |in_file|.
// The name of the output file in |out_file|. The return value indicates whether
// optimization should continue and a status code indicating an error or
// success.
OptStatus ParseFlags(int argc, const char** argv, Optimizer* optimizer,
                     const char** in_file, const char** out_file) {
  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\n", spvSoftwareVersionDetailsString());
        return {OPT_STOP, 0};
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        return {OPT_STOP, 0};
      } else if (0 == strcmp(cur_arg, "-o")) {
        if (!*out_file && argi + 1 < argc) {
          *out_file = argv[++argi];
        } else {
          PrintUsage(argv[0]);
          return {OPT_STOP, 1};
        }
      } else if (0 == strcmp(cur_arg, "--strip-debug")) {
        optimizer->RegisterPass(CreateStripDebugInfoPass());
      } else if (0 == strcmp(cur_arg, "--set-spec-const-default-value")) {
        if (++argi < argc) {
          auto spec_ids_vals =
              opt::SetSpecConstantDefaultValuePass::ParseDefaultValuesString(
                  argv[argi]);
          if (!spec_ids_vals) {
            fprintf(stderr,
                    "error: Invalid argument for "
                    "--set-spec-const-default-value: %s\n",
                    argv[argi]);
            return {OPT_STOP, 1};
          }
          optimizer->RegisterPass(
              CreateSetSpecConstantDefaultValuePass(std::move(*spec_ids_vals)));
        } else {
          fprintf(
              stderr,
              "error: Expected a string of <spec id>:<default value> pairs.");
          return {OPT_STOP, 1};
        }
      } else if (0 == strcmp(cur_arg, "--freeze-spec-const")) {
        optimizer->RegisterPass(CreateFreezeSpecConstantValuePass());
      } else if (0 == strcmp(cur_arg, "--inline-entry-points-exhaustive")) {
        optimizer->RegisterPass(CreateInlineExhaustivePass());
      } else if (0 == strcmp(cur_arg, "--inline-entry-points-opaque")) {
        optimizer->RegisterPass(CreateInlineOpaquePass());
      } else if (0 == strcmp(cur_arg, "--convert-local-access-chains")) {
        optimizer->RegisterPass(CreateLocalAccessChainConvertPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-code-aggressive")) {
        optimizer->RegisterPass(CreateAggressiveDCEPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-insert-extract")) {
        optimizer->RegisterPass(CreateInsertExtractElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-single-block")) {
        optimizer->RegisterPass(CreateLocalSingleBlockLoadStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-single-store")) {
        optimizer->RegisterPass(CreateLocalSingleStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--merge-blocks")) {
        optimizer->RegisterPass(CreateBlockMergePass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-branches")) {
        optimizer->RegisterPass(CreateDeadBranchElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-functions")) {
        optimizer->RegisterPass(CreateEliminateDeadFunctionsPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-local-multi-store")) {
        optimizer->RegisterPass(CreateLocalMultiStoreElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-common-uniform")) {
        optimizer->RegisterPass(CreateCommonUniformElimPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-const")) {
        optimizer->RegisterPass(CreateEliminateDeadConstantPass());
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-variables")) {
        optimizer->RegisterPass(CreateDeadVariableEliminationPass());
      } else if (0 == strcmp(cur_arg, "--fold-spec-const-op-composite")) {
        optimizer->RegisterPass(CreateFoldSpecConstantOpAndCompositePass());
      } else if (0 == strcmp(cur_arg, "--strength-reduction")) {
        optimizer->RegisterPass(CreateStrengthReductionPass());
      } else if (0 == strcmp(cur_arg, "--unify-const")) {
        optimizer->RegisterPass(CreateUnifyConstantPass());
      } else if (0 == strcmp(cur_arg, "--flatten-decorations")) {
        optimizer->RegisterPass(CreateFlattenDecorationPass());
      } else if (0 == strcmp(cur_arg, "--compact-ids")) {
        optimizer->RegisterPass(CreateCompactIdsPass());
      } else if (0 == strcmp(cur_arg, "--cfg-cleanup")) {
        optimizer->RegisterPass(CreateCFGCleanupPass());
      } else if (0 == strcmp(cur_arg, "-O")) {
        optimizer->RegisterPerformancePasses();
      } else if (0 == strcmp(cur_arg, "-Os")) {
        optimizer->RegisterSizePasses();
      } else if (0 == strncmp(cur_arg, "-Oconfig=", sizeof("-Oconfig=") - 1)) {
        OptStatus status =
            ParseOconfigFlag(argv[0], cur_arg, optimizer, in_file, out_file);
        if (status.action != OPT_CONTINUE) {
          return status;
        }
      } else if ('\0' == cur_arg[1]) {
        // Setting a filename of "-" to indicate stdin.
        if (!*in_file) {
          *in_file = cur_arg;
        } else {
          fprintf(stderr, "error: More than one input file specified\n");
          return {OPT_STOP, 1};
        }
      } else {
        fprintf(
            stderr,
            "error: Unknown flag '%s'. Use --help for a list of valid flags\n",
            cur_arg);
        return {OPT_STOP, 1};
      }
    } else {
      if (!*in_file) {
        *in_file = cur_arg;
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return {OPT_STOP, 1};
      }
    }
  }

  return {OPT_CONTINUE, 0};
}

}  // namespace

int main(int argc, const char** argv) {
  const char* in_file = nullptr;
  const char* out_file = nullptr;

  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_2;

  spvtools::Optimizer optimizer(target_env);
  optimizer.SetMessageConsumer([](spv_message_level_t level, const char* source,
                                  const spv_position_t& position,
                                  const char* message) {
    std::cerr << StringifyMessage(level, source, position, message)
              << std::endl;
  });

  OptStatus status = ParseFlags(argc, argv, &optimizer, &in_file, &out_file);
  if (status.action == OPT_STOP) {
    return status.code;
  }

  if (out_file == nullptr) {
    fprintf(stderr, "error: -o required\n");
    return 1;
  }

  std::vector<uint32_t> binary;
  if (!ReadFile<uint32_t>(in_file, "rb", &binary)) {
    return 1;
  }

  // Let's do validation first.
  spv_context context = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary_struct = {binary.data(), binary.size()};
  spv_result_t error = spvValidate(context, &binary_struct, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return error;
  }
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  // By using the same vector as input and output, we save time in the case
  // that there was no change.
  bool ok = optimizer.Run(binary.data(), binary.size(), &binary);

  if (!WriteFile<uint32_t>(out_file, "wb", binary.data(), binary.size())) {
    return 1;
  }

  return ok ? 0 : 1;
}
