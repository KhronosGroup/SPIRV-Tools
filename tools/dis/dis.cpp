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

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <stdio.h>  // Need fileno
#include <unistd.h>
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "source/print.h"
#include "spirv-tools/libspirv.h"
#include "tools/io.h"
#include "tools/util/flags.h"

namespace print = spvtools::print;

static const std::string kHelpText = R"(%s - Disassemble a SPIR-V binary module

Usage: %s [options] [<filename>]

The SPIR-V binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

A text-based hex stream is also accepted as binary input, which should either
consist of 32-bit words or 8-bit bytes.  The 0x or x prefix is optional, but
should be consistently present in the stream.

Options:

  -h, --help        Print this help.
  --version         Display disassembler version information.

  -o <filename>     Set the output filename.
                    Output goes to standard output if this option is
                    not specified, or if the filename is "-".

  --color           Force color output.  The default when printing to a terminal.
                    Overrides a previous --no-color option.
  --no-color        Don't print in color.  Overrides a previous --color option.
                    The default when output goes to something other than a
                    terminal (e.g. a file, a pipe, or a shell redirection).

  --style           Overriding --color, use a more information-packed color scheme.
  --no-style        Override --style and disable it.

  --no-indent       Don't indent instructions.

  --no-header       Don't output the header as leading comments.

  --raw-id          Show raw Id values instead of friendly names.

  --nested-indent   Indentation is adjusted to indicate nesting in structured
                    control flow.

  --reorder-blocks  Reorder blocks to match the structured control flow of SPIR-V.
                    With this option, the order of instructions will no longer
                    match the input binary, but the result will be more readable.

  --offsets         Show byte offsets for each instruction.

  --comment         Add comments to make reading easier
)";

// clang-format off
FLAG_SHORT_bool  (h,              /* default_value= */ false, /* required= */ false);
FLAG_SHORT_string(o,              /* default_value= */ "-",   /* required= */ false);
FLAG_LONG_bool   (help,           /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (version,        /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (color,          /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (no_color,       /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (style,          /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (no_style,       /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (no_indent,      /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (no_header,      /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (raw_id,         /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (nested_indent,  /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (reorder_blocks, /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (offsets,        /* default_value= */ false, /* required= */ false);
FLAG_LONG_bool   (comment,        /* default_value= */ false, /* required= */ false);
// clang-format on

static const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_5;

void PrintStylized(spv_text text) {
  constexpr print::Color kColorFloat64 = print::Color::Orange;
  constexpr print::Color kColorFloat32 = print::Color::Yellow;
  constexpr print::Color kColorFloat16OrLess = print::Color::Green;
  constexpr print::Color kColorInt = print::Color::Blue;
  constexpr print::Color kColorUint = print::Color::Cyan;
  constexpr print::Color kColorBool = print::Color::Magenta;
  constexpr print::Color kColorImage = print::Color::Purple;
  constexpr print::Color kColorSampler = print::Color::Brown;
  constexpr print::Color kColorStringLiteral = print::Color::Green;
  constexpr print::Color kColorNumericLiteral = print::Color::Red;

  constexpr print::Style kStylePointer = print::Style::Bold;
  constexpr print::Style kStyleConstant = print::Style::Italic;
  constexpr print::Style kStyleType = print::Style::Underline;
  constexpr print::Style kStyleTypePointer = print::Style::BoldUnderline;
  constexpr print::Style kStyleLabel = print::Style::Faint;

  // First, print the legend, so the reader can make sense of the colors.
  std::cout << "; Legend:\n";
  std::cout << ";   Base types: ";
  print::SetColor(std::cout, true, kColorFloat64) << "float64";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorFloat32) << "float32";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorFloat16OrLess) << "float16-";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorInt) << "int";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorUint) << "uint";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorBool) << "bool";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorImage) << "image";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorSampler) << "sampler";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorStringLiteral) << "string-literal";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, kColorNumericLiteral) << "numeric-literal";
  print::SetColor(std::cout, true, print::Color::Reset);
  std::cout << "\n";
  std::cout << ";   Kinds: ";
  print::SetColor(std::cout, true, print::Color::Reset, kStylePointer)
      << "pointer";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, print::Color::Reset, kStyleConstant)
      << "constant";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, print::Color::Reset, kStyleType) << "type";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, print::Color::Reset, kStyleTypePointer)
      << "type-pointer";
  print::SetColor(std::cout, true, print::Color::Reset) << " ";
  print::SetColor(std::cout, true, print::Color::Reset, kStyleLabel) << "label";
  print::SetColor(std::cout, true, print::Color::Reset);
  std::cout << "\n";

  // Then go over the SPIR-V and output it piecemeal, replacing the style
  // markers with print colors.
  const char* str = text->str;
  const char* end = str + text->length;

  while (true) {
    // Find the next style begin delimiter.
    const char* style_begin = strchr(str, SPV_BINARY_TO_TEXT_STYLE_BEGIN);

    // If none are found, output the rest of the SPIR-V and finish.
    if (style_begin == nullptr) {
      std::cout.write(str, end - str);
      break;
    }

    // If a style is found, output the SPIR-V so far first.
    std::cout.write(str, style_begin - str);

    // Look at the style markers until the style end delimiter is seen.
    str = ++style_begin;
    print::Color color = print::Color::Reset;
    print::Style style = print::Style::Reset;

    bool is_style_end = false;
    do {
      switch (*str) {
        case SPV_BINARY_TO_TEXT_STYLE_END:
          is_style_end = true;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_FLOAT64:
          color = kColorFloat64;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_FLOAT32:
          color = kColorFloat32;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_FLOAT16_OR_LESS:
          color = kColorFloat16OrLess;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_INT:
          color = kColorInt;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_UINT:
          color = kColorUint;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_BOOL:
          color = kColorBool;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_IMAGE:
          color = kColorImage;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_SAMPLER:
          color = kColorSampler;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_STRING_LITERAL:
          color = kColorStringLiteral;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_NUMERIC_LITERAL:
          color = kColorNumericLiteral;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_POINTER:
          style = kStylePointer;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_CONSTANT:
          style = kStyleConstant;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_TYPE:
          style = kStyleType;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_TYPE_POINTER:
          style = kStyleTypePointer;
          break;
        case SPV_BINARY_TO_TEXT_STYLE_LABEL:
          style = kStyleLabel;
          break;
        default:
          // Unexpected marker
          assert(false);
      }
      ++str;
    } while (!is_style_end);

    // Apply the style and continue until the next one.
    print::SetColor(std::cout, true, color, style);
  }
}

int main(int, const char** argv) {
  if (!flags::Parse(argv)) {
    return 1;
  }

  if (flags::h.value() || flags::help.value()) {
    printf(kHelpText.c_str(), argv[0], argv[0]);
    return 0;
  }

  if (flags::version.value()) {
    printf("%s\n", spvSoftwareVersionDetailsString());
    printf("Target: %s\n", spvTargetEnvDescription(kDefaultEnvironment));
    return 0;
  }

  if (flags::positional_arguments.size() > 1) {
    fprintf(stderr, "error: more than one input file specified.\n");
    return 1;
  }

  const std::string inFile = flags::positional_arguments.size() == 0
                                 ? "-"
                                 : flags::positional_arguments[0];
  const std::string outFile = flags::o.value();

  bool color_is_possible =
#if SPIRV_COLOR_TERMINAL
      true;
#else
      false;
#endif

  uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NONE;
  bool print_to_stdout = false;

  if (!flags::no_indent.value()) options |= SPV_BINARY_TO_TEXT_OPTION_INDENT;

  if (flags::offsets.value())
    options |= SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET;

  if (flags::no_header.value()) options |= SPV_BINARY_TO_TEXT_OPTION_NO_HEADER;

  if (!flags::raw_id.value())
    options |= SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES;

  if (flags::nested_indent.value())
    options |= SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT;

  if (flags::reorder_blocks.value())
    options |= SPV_BINARY_TO_TEXT_OPTION_REORDER_BLOCKS;

  if (flags::comment.value()) options |= SPV_BINARY_TO_TEXT_OPTION_COMMENT;

  if (flags::o.value() == "-") {
    // If printing to standard output, then spvBinaryToText should
    // do the printing.  In particular, colour printing on Windows is
    // controlled by modifying console objects synchronously while
    // outputting to the stream rather than by injecting escape codes
    // into the output stream.
    print_to_stdout = true;
    options |= SPV_BINARY_TO_TEXT_OPTION_PRINT;
    if (color_is_possible && !flags::no_color.value()) {
      bool output_is_tty = true;
#if defined(_POSIX_VERSION)
      output_is_tty = isatty(fileno(stdout));
#endif
      if (output_is_tty || flags::color.value()) {
        options |= SPV_BINARY_TO_TEXT_OPTION_COLOR;
      }
    }

    if (flags::style.value() && !flags::no_style.value()) {
      // If stylizing the output, the responsibility of coloring and printing is
      // on the caller.  In that case, remove the COLOR and PRINT options.
      options |= SPV_BINARY_TO_TEXT_OPTION_STYLE;
      options &= ~SPV_BINARY_TO_TEXT_OPTION_PRINT;
      options &= ~SPV_BINARY_TO_TEXT_OPTION_COLOR;
    }
  }

  // Read the input binary.
  std::vector<uint32_t> contents;
  if (!ReadBinaryFile(inFile.c_str(), &contents)) return 1;

  // If the printing option is off, then save the text in memory, so
  // it can be emitted later in this function. If stylized, also save the text
  // in memory since it needs to be post-processed before printing.
  const bool isStylized = (options & SPV_BINARY_TO_TEXT_OPTION_STYLE) != 0;
  spv_text text = nullptr;
  spv_text* textOrNull = print_to_stdout && !isStylized ? nullptr : &text;
  spv_diagnostic diagnostic = nullptr;
  spv_context context = spvContextCreate(kDefaultEnvironment);
  spv_result_t error =
      spvBinaryToText(context, contents.data(), contents.size(), options,
                      textOrNull, &diagnostic);
  spvContextDestroy(context);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  if (!print_to_stdout) {
    if (!WriteFile<char>(outFile.c_str(), "w", text->str, text->length)) {
      spvTextDestroy(text);
      return 1;
    }
  } else if (isStylized) {
    PrintStylized(text);
  } else {
    // Output is already printed to stdout
    assert(textOrNull == nullptr);
  }
  spvTextDestroy(text);

  return 0;
}
