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

#include "libspirv/libspirv.h"

void print_usage(char* argv0) {
  printf(
      "Validate a SPIR-V binary file.\n\n"
      "USAGE: %s [options] <filename>\n\n"
      "        -basic                     Perform basic validation (disabled)\n"
      "        -layout                    Perform layout validation "
      "(disabled)\n"
      "        -id                        Perform id validation (default ON)\n"
      "        -capability <capability>   Performs OpCode validation "
      "(disabled)\n",
      argv0);
}

int main(int argc, char** argv) {
  if (2 > argc) {
    print_usage(argv[0]);
    return 1;
  }

  const char* inFile = nullptr;
  uint32_t options = 0;

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      if (!strcmp("basic", argv[argi] + 1)) {
        options |= SPV_VALIDATE_BASIC_BIT;
      } else if (!strcmp("layout", argv[argi] + 1)) {
        options |= SPV_VALIDATE_LAYOUT_BIT;
      } else if (!strcmp("id", argv[argi] + 1)) {
        options |= SPV_VALIDATE_ID_BIT;
      } else if (!strcmp("rules", argv[argi] + 1)) {
        options |= SPV_VALIDATE_RULES_BIT;
      } else {
        print_usage(argv[0]);
        return 1;
      }
    } else {
      if (!inFile) {
        inFile = argv[argi];
      } else {
        print_usage(argv[0]);
        return 1;
      }
    }
  }

  if (!inFile) {
    fprintf(stderr, "error: input file is empty.\n");
    return 1;
  }

  std::vector<uint32_t> contents;
  if (FILE* fp = fopen(inFile, "rb")) {
    uint32_t buf[1024];
    while (size_t len = fread(buf, sizeof(uint32_t),
                              sizeof(buf) / sizeof(uint32_t), fp)) {
      contents.insert(contents.end(), buf, buf + len);
    }
    fclose(fp);
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
