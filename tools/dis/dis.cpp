// Copyright (c) 2015 The Khronos Group Inc.
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

#include <cstring>
#include <vector>

#include "libspirv/libspirv.h"

static void print_usage(char* argv0) {
  printf(
      R"(%s - Disassemble a binary SPIR-V module

Usage: %s [options] [<filename>]

Options:

  -h, --help      Print this help.

  -o <filename>   Set the output filename.
                  Otherwise output goes to standard output.

  --no-color      Don't print in color.
                  The default when output goes to a file.

  --no-indent     Don't indent instructions.
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

  bool allow_color = false;
#ifdef SPIRV_COLOR_TERMINAL
  allow_color = true;
#endif
  bool allow_indent = true;

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'o': {
          if (!outFile && argi + 1 < argc) {
            outFile = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '-': {
          // Long options
          if (0 == strcmp(argv[argi], "--no-color")) allow_color = false;
          if (0 == strcmp(argv[argi], "--no-indent")) allow_indent = false;
          if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
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

  uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NONE;

  if (allow_indent)
    options |= SPV_BINARY_TO_TEXT_OPTION_INDENT;

  if (!outFile) {
    // Print to standard output.
    options |= SPV_BINARY_TO_TEXT_OPTION_PRINT;
    if (allow_color) {
      options |= SPV_BINARY_TO_TEXT_OPTION_COLOR;
    }
  }

  if (!inFile) {
    fprintf(stderr, "error: input file is empty.\n");
    return 1;
  }

  std::vector<uint32_t> contents;
  if (FILE* fp = fopen(inFile, "rb")) {
    uint32_t buf[1024];
    while (size_t len = fread(buf, sizeof(uint32_t), 1024, fp)) {
      contents.insert(contents.end(), buf, buf + len);
    }
    fclose(fp);
  } else {
    fprintf(stderr, "error: file does not exist '%s'\n", inFile);
    return 1;
  }

  // If printing to standard output, then spvBinaryToText should
  // do the printing.  In particular, colour printing on Windows is
  // controlled by modifying console objects synchronously while
  // outputting to the stream rather than by injecting escape codes
  // into the output stream.
  // If the printing option is off, then save the text in memory, so
  // it can be emitted later in this function.
  const bool print_to_stdout =
      spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options);
  spv_text text;
  spv_text* textOrNull = print_to_stdout ? nullptr : &text;
  spv_diagnostic diagnostic = nullptr;
  spv_result_t error = spvBinaryToText(contents.data(), contents.size(),
                                       options, textOrNull, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  // Output the result.
  if (!print_to_stdout) {
    if (FILE* fp = fopen(outFile, "w")) {
      size_t written =
          fwrite(text->str, sizeof(char), (size_t)text->length, fp);
      if (text->length != written) {
        spvTextDestroy(text);
        fprintf(stderr, "error: could not write to file '%s'\n", outFile);
        return 1;
      }
    } else {
      spvTextDestroy(text);
      fprintf(stderr, "error: could not open file '%s'\n", outFile);
      return 1;
    }
  }

  return 0;
}
