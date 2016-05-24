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

#include <cstring>
#include <iostream>
#include <vector>

#include "opt/pass_manager.h"
#include "opt/spv_builder.h"
#include "spirv-tools/libspirv.h"

using namespace spvtools::opt;

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
  --strip-debug-info
               Remove all debug related instructions.
  -h, --help   Print this help.
  --version    Display optimizer version information.
)",
      program, program);
}

spv_result_t BuildSpvHeader(void* builder, spv_endianness_t /* endian */,
                            uint32_t magic, uint32_t version,
                            uint32_t generator, uint32_t id_bound,
                            uint32_t reserved) {
  reinterpret_cast<ir::SpvBuilder*>(builder)->SetModuleHeader(
      magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
};

spv_result_t BuildSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  reinterpret_cast<ir::SpvBuilder*>(builder)->AddInstruction(inst);
  return SPV_SUCCESS;
};

int main(int argc, char** argv) {
  const char* in_file = nullptr;
  const char* out_file = nullptr;
  bool strip_debug_info = false;

  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_1;

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
      } else if (0 == strcmp(cur_arg, "--strip-debug-info")) {
        strip_debug_info = true;
      } else if (0 == cur_arg[1]) {
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
  const bool use_file = in_file && strcmp("-", in_file);
  if (FILE* fp = (use_file ? fopen(in_file, "rb") : stdin)) {
    uint32_t buf[1024];
    while (size_t len = fread(buf, sizeof(uint32_t),
                              sizeof(buf) / sizeof(uint32_t), fp)) {
      source.insert(source.end(), buf, buf + len);
    }
    if (use_file) fclose(fp);
  } else {
    fprintf(stderr, "error: file does not exist '%s'\n", in_file);
    return 1;
  }

  spv_context context = spvContextCreate(target_env);
  spv_diagnostic diagnostic = nullptr;

#if 0
  // Let's do validation first.
  spv_const_binary_t binary = {source.data(), source.size()};
  spv_result_t error = spvValidate(context, &binary, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }
#endif

  ir::Module module;
  ir::SpvBuilder spv_builder(&module);
  spvBinaryParse(context, &spv_builder, source.data(), source.size(),
                 BuildSpvHeader, BuildSpvInst, &diagnostic);

  PassManager pass_manager;
  if (strip_debug_info) {
    std::unique_ptr<DebugInfoRemovalPass> pass(new DebugInfoRemovalPass);
    pass_manager.AddPass(std::move(pass));
  }
  pass_manager.run(&module);

  std::vector<uint32_t> target;
  module.ToBinary(&target);

  const bool use_stdout = out_file[0] == '-' && out_file[1] == 0;
  if (FILE* fp = (use_stdout ? stdout : fopen(out_file, "wb"))) {
    size_t written = fwrite(target.data(), sizeof(uint32_t), target.size(), fp);
    if (target.size() != written) {
      fprintf(stderr, "error: could not write to file '%s'\n", out_file);
      return 1;
    }
    if (!use_stdout) fclose(fp);
  } else {
    fprintf(stderr, "error: could not open file '%s'\n", out_file);
    return 1;
  }

  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);
  return 0;
}
