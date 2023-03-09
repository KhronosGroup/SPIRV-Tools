// Copyright (c) 2023 Google LLC.
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

#include "extract_source.h"
#include "source/opt/log.h"
#include "tools/io.h"
#include "tools/util/cli_consumer.h"
#include "tools/util/flags.h"

namespace {

constexpr auto kHelpTextFmt =
    R"(%s - Dumps information from a SPIR-V binary.

Usage: %s [options] <filename>

one of the following switches must be given:
  --source        Extract source files obtained from debug symbols, output to stdout.
  --entrypoint    Extracts the entrypoint name of the module, output to stdout.
  --compiler-cmd  Extracts the command line used to compile this module, output to stdout.


General options:
  -h, --help      Print this help.
  --version       Display assembler version information.
  -f,--force      Allow output file overwrite.

Source dump options:
  --list          Do not extract source code, only print filenames to stdout.
  --outdir        Where shall the exrtacted HLSL/HLSL files be written to?
                  File written to stdout if '-' is given. Default is `-`.
)";

}  // namespace

// clang-format off
FLAG_SHORT_bool(  h,            /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   help,         /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   version,      /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   source,       /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   entrypoint,   /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   compiler_cmd, /* default_value= */ false, /* required= */ false);
FLAG_SHORT_bool(  f,            /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool(   force,        /* default_value= */ false, /* required= */ false);
FLAG_LONG_string( outdir,       /* default_value= */ "-",   /* required= */ false);
FLAG_LONG_bool(   list,         /* default_value= */ false, /* required= */ false);
// clang-format on

int main(int, const char** argv) {
  if (!flags::Parse(argv)) {
    return 1;
  }
  if (flags::h.value() || flags::help.value()) {
    printf(kHelpTextFmt, argv[0], argv[0]);
    return 0;
  }
  if (flags::version.value()) {
    printf("%s\n", spvSoftwareVersionDetailsString());
    return 0;
  }

  if (flags::positional_arguments.size() != 1) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, nullptr, {},
                    "expected exactly one input file.");
    return 1;
  }
  if (flags::source.value() || flags::entrypoint.value() ||
      flags::compiler_cmd.value()) {
    spvtools::Error(spvtools::utils::CLIMessageConsumer, nullptr, {},
                    "not implemented yet.");
    return 1;
  }

  std::vector<uint32_t> binary;
  if (!ReadBinaryFile(flags::positional_arguments[0].c_str(), &binary)) {
    return 1;
  }

  if (flags::source.value()) {
    std::unordered_map<std::string, std::string> output;
    return extract_source_from_module(binary, &output) ? 0 : 1;
  }

  // FIXME: implement logic.
  return 0;
}
