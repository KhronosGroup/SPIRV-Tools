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

#ifndef SPIRV_TOOLS_MESSAGE_H_
#define SPIRV_TOOLS_MESSAGE_H_

#include <functional>
#include <string>

#include "spirv-tools/libspirv.h"

namespace spvtools {

// TODO(antiagainst): This eventually should be in the C++ interface.

// Severity levels of messages communicated to the consumer.
typedef enum spv_message_level_t {
  SPV_MSG_FATAL,           // Unrecoverable error due to environment.
                           // Will exit the program immediately. E.g.,
                           // out of memory.
  SPV_MSG_INTERNAL_ERROR,  // Unrecoverable error due to SPIRV-Tools
                           // internals.
                           // Will exit the program immediately. E.g.,
                           // unimplemented feature.
  SPV_MSG_ERROR,           // Normal error due to user input.
  SPV_MSG_WARNINING,       // Warning information.
  SPV_MSG_INFO,            // General information.
  SPV_MSG_DEBUG,           // Debug information.
} spv_message_level_t;

// Message consumer. The C strings for source and message are only alive for the
// specific invocation.
using MessageConsumer = std::function<void(
    spv_message_level_t /* level */, const char* /* source */,
    const spv_position_t& /* position */, const char* /* message */
    )>;

// A message consumer that ignores all messages.
inline void IgnoreMessage(spv_message_level_t, const char*,
                          const spv_position_t&, const char*) {}

// A helper function to compose and return a string from the message in the
// following format:
//   "<level>: <source>:<line>:<column>:<index>: <message>"
std::string StringifyMessage(spv_message_level_t level, const char* source,
                             const spv_position_t& position,
                             const char* message);

}  // namespace spvtools

#endif  // SPIRV_TOOLS_MESSAGE_H_
