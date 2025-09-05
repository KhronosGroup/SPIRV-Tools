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

#ifndef SOURCE_PRINT_H_
#define SOURCE_PRINT_H_

#include <iostream>
#include <sstream>

namespace spvtools {

// Wrapper for out stream selection.
class out_stream {
 public:
  out_stream() : pStream(nullptr) {}
  explicit out_stream(std::stringstream& stream) : pStream(&stream) {}

  std::ostream& get() {
    if (pStream) {
      return *pStream;
    }
    return std::cout;
  }

 private:
  std::stringstream* pStream;
};

namespace print {
enum class Color {
  Reset,
  Black,
  Red,
  Green,
  Blue,
  Yellow,
  Cyan,
  Magenta,
  Brown,
  Purple,
  Orange,
};

enum class Style {
  Reset,
  Bold,
  Faint,
  Underline,
  Italic,
  BoldUnderline,
};

std::ostream& SetColor(std::ostream& stream, bool isPrintToConsole, Color color,
                       Style style = Style::Reset);
}  // namespace print
}  // namespace spvtools

#endif  // SOURCE_PRINT_H_
