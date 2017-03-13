// Copyright (c) 2017 Google Inc.
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

#ifndef LIBSPIRV_UTIL_STRING_UTILS_H_
#define LIBSPIRV_UTIL_STRING_UTILS_H_

#include <string>

namespace spvutils {

// Converts cardinal number to ordinal number string.
std::string CardinalToOrdinal(size_t cardinal) {
  const size_t mod10 = cardinal % 10;
  const size_t mod100 = cardinal % 100;
  std::string suffix;
  if (mod10 == 1 && mod100 != 11)
    suffix = "st";
  else if (mod10 == 2 && mod100 != 12)
    suffix = "nd";
  else if (mod10 == 3 && mod100 != 13)
    suffix = "rd";
  else
    suffix = "th";

  return std::to_string(cardinal) + suffix;
}

}  // namespace spvutils

#endif  // LIBSPIRV_UTIL_STRING_UTILS_H_
