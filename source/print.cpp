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

#include "print.h"

#if defined(SPIRV_ANDROID) || defined(SPIRV_LINUX) || defined(SPIRV_MAC) || defined(SPIRV_FREEBSD)
namespace libspirv {

clr::reset::operator const char*() { return "\x1b[0m"; }

clr::grey::operator const char*() { return "\x1b[1;30m"; }

clr::red::operator const char*() { return "\x1b[31m"; }

clr::green::operator const char*() { return "\x1b[32m"; }

clr::yellow::operator const char*() { return "\x1b[33m"; }

clr::blue::operator const char*() { return "\x1b[34m"; }

}  // namespace libspirv
#elif defined(SPIRV_WINDOWS)
#include <windows.h>

namespace libspirv {

static void SetConsoleForegroundColorPrimary(HANDLE hConsole, WORD color)
{
  // Get screen buffer information from console handle
  CONSOLE_SCREEN_BUFFER_INFO bufInfo;
  GetConsoleScreenBufferInfo(hConsole, &bufInfo);

  // Get background color
  color |= (bufInfo.wAttributes & 0xfff0);

  // Set foreground color
  SetConsoleTextAttribute(hConsole, color);
}

static void SetConsoleForegroundColor(WORD color)
{
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_OUTPUT_HANDLE), color);
  SetConsoleForegroundColorPrimary(GetStdHandle(STD_ERROR_HANDLE), color);
}

clr::reset::operator const char*() {
  SetConsoleForegroundColor(0xf);
  return "";
}

clr::grey::operator const char*() {
  SetConsoleForegroundColor(FOREGROUND_INTENSITY);
  return "";
}

clr::red::operator const char*() {
  SetConsoleForegroundColor(FOREGROUND_RED);
  return "";
}

clr::green::operator const char*() {
  SetConsoleForegroundColor(FOREGROUND_GREEN);
  return "";
}

clr::yellow::operator const char*() {
  SetConsoleForegroundColor(FOREGROUND_RED | FOREGROUND_GREEN);
  return "";
}

clr::blue::operator const char*() {
  SetConsoleForegroundColor(FOREGROUND_BLUE);
  return "";
}

}  // namespace libspirv
#else
namespace libspirv {

clr::reset::operator const char*() { return ""; }

clr::grey::operator const char*() { return ""; }

clr::red::operator const char*() { return ""; }

clr::green::operator const char*() { return ""; }

clr::yellow::operator const char*() { return ""; }

clr::blue::operator const char*() { return ""; }

}  // namespace libspirv
#endif
