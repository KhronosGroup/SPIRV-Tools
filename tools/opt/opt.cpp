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

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "message.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_loader.h"
#include "source/opt/pass_manager.h"
#include "source/opt/passes.h"
#include "tools/io.h"

using namespace spvtools;

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
  -h, --help   Print this help.
  --version    Display optimizer version information.
)",
      program, program);
}

int main(int argc, char** argv) {
  const char* in_file = nullptr;
  const char* out_file = nullptr;

  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_1;

  opt::PassManager pass_manager;
  pass_manager.SetMessageConsumer(
      [](spv_message_level_t level, const char* source,
         const spv_position_t& position, const char* message) {
        std::cerr << StringifyMessage(level, source, position, message)
                  << std::endl;
      });

  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\n", spvSoftwareVersionDetailsString());
        return 0;
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        return 0;
      } else if (0 == strcmp(cur_arg, "-o")) {
        if (!out_file && argi + 1 < argc) {
          out_file = argv[++argi];
        } else {
          PrintUsage(argv[0]);
          return 1;
        }
      } else if (0 == strcmp(cur_arg, "--strip-debug")) {
        pass_manager.AddPass<opt::StripDebugInfoPass>();
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
            return 1;
          }
          pass_manager.AddPass<opt::SetSpecConstantDefaultValuePass>(
              std::move(*spec_ids_vals));
        } else {
          fprintf(
              stderr,
              "error: Expected a string of <spec id>:<default value> pairs.");
          return 1;
        }
      } else if (0 == strcmp(cur_arg, "--freeze-spec-const")) {
        pass_manager.AddPass<opt::FreezeSpecConstantValuePass>();
      } else if (0 == strcmp(cur_arg, "--eliminate-dead-const")) {
        pass_manager.AddPass<opt::EliminateDeadConstantPass>();
      } else if (0 == strcmp(cur_arg, "--fold-spec-const-op-composite")) {
        pass_manager.AddPass<opt::FoldSpecConstantOpAndCompositePass>();
      } else if (0 == strcmp(cur_arg, "--unify-const")) {
        pass_manager.AddPass<opt::UnifyConstantPass>();
      } else if ('\0' == cur_arg[1]) {
        // Setting a filename of "-" to indicate stdin.
        if (!in_file) {
          in_file = cur_arg;
        } else {
          fprintf(stderr, "error: More than one input file specified\n");
          return 1;
        }
      } else {
        PrintUsage(argv[0]);
        return 1;
      }
    } else {
      if (!in_file) {
        in_file = cur_arg;
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  if (out_file == nullptr) {
    fprintf(stderr, "error: -o required\n");
    return 1;
  }

  std::vector<uint32_t> source;
  if (!ReadFile<uint32_t>(in_file, "rb", &source)) return 1;

  // Let's do validation first.
  spv_context context = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t binary = {source.data(), source.size()};
  spv_result_t error = spvValidate(context, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return error;
  }
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  std::unique_ptr<ir::Module> module = BuildModule(
      target_env, pass_manager.consumer(), source.data(), source.size());
  pass_manager.Run(module.get());

  std::vector<uint32_t> target;
  module->ToBinary(&target, /* skip_nop = */ true);

  if (!WriteFile<uint32_t>(out_file, "wb", target.data(), target.size())) {
    return 1;
  }

  return 0;
}
