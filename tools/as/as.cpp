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

#include <stdio.h>

#include <vector>

#include "libspirv/libspirv.h"

void print_usage(char* argv0) {
  printf(
      R"(%s - Create a SPIR-V binary module from SPIR-V assembly text

Usage: %s [options] <filename>

The SPIR-V assembly text is read from <filename>.  The SPIR-V binary
module is written to file "out.spv", unless the -o option is used.

Options:

  -h              Print this help.

  -o <filename>   Set the output filename.
)",
      argv0, argv0);
}

int main(int argc, char** argv) {
  if (2 > argc) {
    print_usage(argv[0]);
    return 1;
  }

  const char* inFile = nullptr;
  const char* outFile = nullptr;

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h': {
          print_usage(argv[0]);
          return 0;
        }
        case 'o': {
          if (!outFile && argi + 1 < argc) {
            outFile = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        default:
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

  if (!outFile) {
    outFile = "out.spv";
  }

  if (!inFile) {
    fprintf(stderr, "error: input file is empty.\n");
    return 1;
  }

  std::vector<char> contents;
  if (FILE* fp = fopen(inFile, "r")) {
    char buf[1024];
    while (size_t len = fread(buf, 1, sizeof(buf), fp))
      contents.insert(contents.end(), buf, buf + len);
    fclose(fp);
  } else {
    fprintf(stderr, "error: file does not exist '%s'\n", inFile);
    return 1;
  }

  spv_binary binary;
  spv_diagnostic diagnostic = nullptr;
  spv_context context = spvContextCreate();
  spv_result_t error = spvTextToBinary(context, contents.data(),
                                       contents.size(), &binary, &diagnostic);
  spvContextDestroy(context);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  if (FILE* fp = fopen(outFile, "wb")) {
    size_t written =
        fwrite(binary->code, sizeof(uint32_t), (size_t)binary->wordCount, fp);
    if (binary->wordCount != written) {
      fprintf(stderr, "error: could not write to file '%s'\n", outFile);
      return 1;
    }
  } else {
    fprintf(stderr, "error: could not open file '%s'\n", outFile);
    return 1;
  }

  spvBinaryDestroy(binary);

  return 0;
}
