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

#include <vector>

#include "libspirv/libspirv.h"

void print_usage(char* argv0) {
  printf(
      "Dissassemble a *.sv file into a *.svasm text file.\n\n"
      "USAGE: %s [options] <filename>\n\n"
      "  -o <filename>   set the output filename\n"
      "  -p              print dissassembly to stdout, this\n"
      "                  overrides file output\n",
      argv0);
}

int main(int argc, char** argv) {
  if (2 > argc) {
    print_usage(argv[0]);
    return 1;
  }

  uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NONE;
  const char* inFile = nullptr;
  const char* outFile = nullptr;

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'o': {
          if (!outFile && argi + 1 < argc) {
            outFile = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case 'p': {
          options |= SPV_BINARY_TO_TEXT_OPTION_PRINT;
#ifdef SPV_COLOR_TERMINAL
          options |= SPV_BINARY_TO_TEXT_OPTION_COLOR;
#endif
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
    outFile = "out.spvasm";
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

  // If the printing option is turned on, then spvBinaryToText should
  // do the printing.  In particular, colour printing on Windows is
  // controlled by modifying console objects synchronously while
  // outputting to the stream rather than by injecting escape codes
  // into the output stream.
  // If the printing option is off, then save the text in memory, so
  // it can be emitted later in this function.
  const bool printOptionOn =
      spvIsInBitfield(SPV_BINARY_TO_TEXT_OPTION_PRINT, options);
  spv_text text;
  spv_text* textOrNull = printOptionOn ? nullptr : &text;
  spv_diagnostic diagnostic = nullptr;
  spv_result_t error = spvBinaryToText(contents.data(), contents.size(),
                                       options, textOrNull, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  // Output the result.
  if (!printOptionOn) {
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
