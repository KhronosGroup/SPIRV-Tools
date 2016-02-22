// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "spirv-tools/libspirv.h"

void print_usage(char* argv0) {
  printf(
      R"(%s - Validate a SPIR-V binary file.

USAGE: %s [options] [<filename>]

The SPIR-V binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

Options:
  -all       Perform all validation (default OFF)
  -basic     Perform basic validation (default OFF)
  -id        Perform id validation (default OFF)
  -layout    Perform layout validation (default OFF)
  -rules     Perform rules validation (default OFF)
  --version  Display validator verision information
)",
      argv0, argv0);
}

const char kBuildVersion[] =
#include "build-version.inc"
;

int main(int argc, char** argv) {
  const char* inFile = nullptr;
  uint32_t options = 0;

  for (int argi = 1; argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\r\n", kBuildVersion);
        printf("Target: SPIR-V %d.%d rev %d\r\n", SPV_SPIRV_VERSION_MAJOR,
               SPV_SPIRV_VERSION_MINOR, SPV_SPIRV_VERSION_REVISION);
        return 0;
      } else if (!strcmp("all", cur_arg + 1)) {
        options |= SPV_VALIDATE_ALL;
      } else if (!strcmp("basic", cur_arg + 1)) {
        options |= SPV_VALIDATE_BASIC_BIT;
      } else if (!strcmp("id", cur_arg + 1)) {
        options |= SPV_VALIDATE_ID_BIT;
      } else if (!strcmp("layout", cur_arg + 1)) {
        options |= SPV_VALIDATE_LAYOUT_BIT;
      } else if (!strcmp("rules", cur_arg + 1)) {
        options |= SPV_VALIDATE_RULES_BIT;
      } else if (0 == cur_arg[1]) {
        // Setting a filename of "-" to indicate stdin.
        if (!inFile) {
          inFile = cur_arg;
        } else {
          fprintf(stderr, "error: More than one input file specified\n");
          return 1;
        }

      } else {
        print_usage(argv[0]);
        return 1;
      }
    } else {
      if (!inFile) {
        inFile = cur_arg;
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  std::vector<uint32_t> contents;
  const bool use_file = inFile && strcmp("-", inFile);
  if (FILE* fp = (use_file ? fopen(inFile, "rb") : stdin)) {
    uint32_t buf[1024];
    while (size_t len = fread(buf, sizeof(uint32_t),
                              sizeof(buf) / sizeof(uint32_t), fp)) {
      contents.insert(contents.end(), buf, buf + len);
    }
    if (use_file) fclose(fp);
  } else {
    fprintf(stderr, "error: file does not exist '%s'\n", inFile);
    return 1;
  }

  spv_const_binary_t binary = {contents.data(), contents.size()};

  spv_diagnostic diagnostic = nullptr;
  spv_context context = spvContextCreate();
  spv_result_t error = spvValidate(context, &binary, options, &diagnostic);
  spvContextDestroy(context);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  return 0;
}
