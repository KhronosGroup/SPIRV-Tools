// Copyright (c) 2015-2016 The Khronos Group Inc.
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
#include <cstdio>
#include <cstring>
#include <vector>

#include "source/spirv_target_env.h"
#include "spirv-tools/libspirv.h"
#include "tools/io.h"
#include "tools/util/flags.h"

static const auto kDefaultEnvironment = "spv1.6";

FLAG_SHORT_bool(h, /* default_value= */ false, "Print this help.", false);
FLAG_LONG_bool(help, /* default_value= */ false, "Print this help.", false);
FLAG_SHORT_string(o, /* default_value= */ "",
                  "Set the output filename. Use '-' to mean stdout.",
                  /* required= */ false);
FLAG_LONG_bool(version, /* default_value= */ false,
               "Display assembler version information.", /* required= */ false);
FLAG_LONG_bool(preserve_numeric_ids, /* default_value= */ false,
               "Numeric IDs in the binary will have the same values as in the "
               "source. Non-numeric IDs are allocated by filling in the gaps, "
               "starting with 1 and going up.",
               /* required= */ false);
FLAG_LONG_string(target_env, kDefaultEnvironment, "Use specified environment.",
                 /* required= */ false);

int main(int, const char** argv) {
  if (!flags::Parse(argv)) {
    return 1;
  }

  if (flags::h.value() || flags::help.value()) {
    flags::PrintHelp(argv, "{binary} {required} [options] <filename>");
    return 0;
  }

  if (flags::version.value()) {
    spv_target_env target_env;
    bool success = spvParseTargetEnv(kDefaultEnvironment, &target_env);
    assert(success && "Default environment should always parse.");
    if (!success) {
      fprintf(stderr,
              "error: invalid default target environment. Please report this "
              "issue.");
      return 1;
    }
    printf("%s\n", spvSoftwareVersionDetailsString());
    printf("Target: %s\n", spvTargetEnvDescription(target_env));
    return 0;
  }

  std::string outFile = flags::o.value();
  if (outFile.empty()) {
    outFile = "out.spv";
  }

  uint32_t options = 0;
  if (flags::preserve_numeric_ids.value()) {
    options |= SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS;
  }

  spv_target_env target_env;
  if (!spvParseTargetEnv(flags::target_env.value().c_str(), &target_env)) {
    fprintf(stderr, "error: Unrecognized target env: %s\n",
            flags::target_env.value().c_str());
    return 1;
  }

  if (flags::positional_arguments.size() != 1) {
    fprintf(stderr, "error: exactly one input file must be specified.\n");
    return 1;
  }
  std::string inFile = flags::positional_arguments[0];

  std::vector<char> contents;
  if (!ReadTextFile<char>(inFile.c_str(), &contents)) return 1;

  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;
  spv_context context = spvContextCreate(target_env);
  spv_result_t error = spvTextToBinaryWithOptions(
      context, contents.data(), contents.size(), options, &binary, &diagnostic);
  spvContextDestroy(context);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  if (!WriteFile<uint32_t>(outFile.c_str(), "wb", binary->code,
                           binary->wordCount)) {
    spvBinaryDestroy(binary);
    return 1;
  }

  spvBinaryDestroy(binary);

  return 0;
}
