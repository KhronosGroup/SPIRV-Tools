// Copyright (c) 2016 Google Inc.
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

#include "message.h"

#include <sstream>

namespace spvtools {

std::string StringifyMessage(MessageLevel level, const char* source,
                             const spv_position_t& position,
                             const char* message) {
  const char* level_string = nullptr;
  switch (level) {
    case MessageLevel::Fatal:
      level_string = "fatal";
      break;
    case MessageLevel::InternalError:
      level_string = "internal error";
      break;
    case MessageLevel::Error:
      level_string = "error";
      break;
    case MessageLevel::Warning:
      level_string = "warning";
      break;
    case MessageLevel::Info:
      level_string = "info";
      break;
    case MessageLevel::Debug:
      level_string = "debug";
      break;
  }
  std::ostringstream oss;
  oss << level_string << ": ";
  if (source) oss << source << ":";
  oss << position.line << ":" << position.column << ":";
  oss << position.index << ": ";
  if (message) oss << message;
  oss << "\n";
  return oss.str();
}

}  // namespace spvtools
