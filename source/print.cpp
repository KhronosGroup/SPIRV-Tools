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

#include "source/print.h"

#if defined(SPIRV_WINDOWS)
#include <windows.h>
#endif

namespace spvtools {
namespace print {

static std::ostream& WriteAnsiColor(std::ostream& stream, Color color,
                                    Style style) {
  if (color == Color::Reset && style == Style::Reset) {
    stream << "\x1b[0m";
    return stream;
  }

  stream << "\x1b[";

  switch (style) {
    case Style::Reset:
      break;
    case Style::Bold:
      stream << "1;";
      break;
    case Style::BoldUnderline:
      stream << "1;4;";
      break;
    case Style::Faint:
      stream << "2;";
      break;
    case Style::Italic:
      stream << "3;";
      break;
    case Style::Underline:
      stream << "4;";
      break;
  }

  switch (color) {
    case Color::Reset:
      stream << "39";
      break;
    case Color::Black:
      stream << "30";
      break;
    case Color::Red:
      stream << "31";
      break;
    case Color::Green:
      stream << "32";
      break;
    case Color::Blue:
      stream << "94";
      break;
    case Color::Yellow:
      stream << "33";
      break;
    case Color::Cyan:
      stream << "36";
      break;
    case Color::Magenta:
      stream << "35";
      break;
    case Color::Brown:
      stream << "38;5;94";
      break;
    case Color::Purple:
      stream << "38;5;127";
      break;
    case Color::Orange:
      stream << "38;5;208";
      break;
  }

  stream << "m";
  return stream;
}

#if defined(SPIRV_WINDOWS)
static void SetConsoleForegroundColorPrimary(HANDLE hConsole, WORD color) {
  // Get screen buffer information from console handle
  CONSOLE_SCREEN_BUFFER_INFO bufInfo;
  GetConsoleScreenBufferInfo(hConsole, &bufInfo);

  // Get background color
  color = WORD(color | (bufInfo.wAttributes & 0xfff0));

  // Set foreground color
  SetConsoleTextAttribute(hConsole, color);
}

static void SetConsoleForegroundColor(WORD color) {
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_OUTPUT_HANDLE), color);
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_ERROR_HANDLE), color);
}

std::ostream& SetColor(std::ostream& stream, bool isPrintToConsole, Color color,
                       Style style) {
  if (!isPrintToConsole) {
    return WriteAnsiColor(stream, color, style);
  }

  WORD console_color = 0;
  switch (style) {
    case Style::Reset:
      break;
    case Style::Bold:
    case Style::BoldUnderline:
      // No underline in the windows console.
      console_color |= FOREGROUND_INTENSITY;
      break;
    case Style::Faint:
    case Style::Italic:
    case Style::Underline:
      // No such configurations in the windows console.
      break;
  }

  switch (color) {
    case Color::Reset:
      console_color |= FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE |
                       FOREGROUND_INTENSITY;
      break;
    case Color::Black:
      break;
    case Color::Red:
      console_color |= FOREGROUND_RED;
      break;
    case Color::Green:
      console_color |= FOREGROUND_GREEN;
      break;
    case Color::Blue:
      console_color |= FOREGROUND_BLUE;
      break;
    case Color::Yellow:
      console_color |= FOREGROUND_RED | FOREGROUND_GREEN;
      break;
    case Color::Cyan:
      console_color |= FOREGROUND_BLUE | FOREGROUND_GREEN;
      break;
    case Color::Magenta:
      console_color |= FOREGROUND_BLUE | FOREGROUND_RED;
      break;
    case Color::Brown:
      // No brown, use bright yellow
      console_color |= FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
      break;
    case Color::Purple:
      // No purple, use bright blue
      console_color |= FOREGROUND_BLUE | FOREGROUND_INTENSITY;
      break;
    case Color::Orange:
      // No orange, use bright red
      console_color |= FOREGROUND_RED | FOREGROUND_INTENSITY;
      break;
  }

  SetConsoleForegroundColor(console_color);
  return stream;
}

#else
std::ostream& SetColor(std::ostream& stream, bool, Color color, Style style) {
  return WriteAnsiColor(stream, color, style);
}
#endif
}  // namespace print
}  // namespace spvtools
