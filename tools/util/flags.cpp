// Copyright (c) 2023 Google LLC.
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

#include "flags.h"

#include <cstring>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace flags {

std::vector<FlagList::FlagInfo> FlagList::flags_;
std::vector<std::string> positional_arguments;

namespace {
// Extracts the flag name from a potential token.
// This function only looks for a '=', to split the flag name from the value for
// long-form flags. Returns the name of the flag, prefixed with the hyphen(s).
inline std::string get_flag_name(const std::string& flag, bool is_short_flag) {
  if (is_short_flag) {
    return flag;
  }

  size_t equal_index = flag.find('=');
  if (equal_index == std::string::npos) {
    return flag;
  }
  return flag.substr(0, equal_index);
}

// Parse a boolean flag. Returns `true` if the parsing succeeded, `false`
// otherwise.
bool parse_flag(Flag<bool>& flag, bool is_short_flag,
                const std::string& token) {
  if (is_short_flag) {
    flag.value() = true;
    return true;
  }

  const std::string raw_flag(token);
  size_t equal_index = raw_flag.find('=');
  if (equal_index == std::string::npos) {
    flag.value() = true;
    return true;
  }

  const std::string value = raw_flag.substr(equal_index + 1);
  if (value == "true") {
    flag.value() = true;
    return true;
  }

  if (value == "false") {
    flag.value() = false;
    return true;
  }

  return false;
}

// Parse a string flag. Moved the iterator to the last flag's token if it's a
// multi-token flag. Returns `true` if the parsing succeeded.
bool parse_flag(Flag<std::string>& flag, bool is_short_flag,
                const char*** iterator) {
  const std::string raw_flag(**iterator);
  const size_t equal_index = raw_flag.find('=');
  if (is_short_flag || equal_index == std::string::npos) {
    if ((*iterator)[1] == nullptr) {
      return false;
    }

    flag.value() = (*iterator)[1];
    *iterator += 1;
    return true;
  }

  const std::string value = raw_flag.substr(equal_index + 1);
  flag.value() = value;
  return true;
}
}  // namespace

// This is the function to expand if you want to support a new type.
bool FlagList::parse_flag_info(FlagInfo& info, const char*** iterator) {
  bool success = false;

  std::visit(
      [&](auto&& item) {
        using T = std::decay_t<decltype(item.get())>;
        if constexpr (std::is_same_v<T, Flag<bool>>) {
          success = parse_flag(item.get(), info.is_short, **iterator);
        } else if constexpr (std::is_same_v<T, Flag<std::string>>) {
          success = parse_flag(item.get(), info.is_short, iterator);
        } else {
          static_assert(always_false_v<T>, "Unsupported flag type.");
        }
      },
      info.flag);

  return success;
}

void FlagList::print_usage(const char* binary_name,
                           const std::string& usage_format) {
  std::string required = "";
  for (const auto& flag : flags_) {
    if (!flag.required) {
      continue;
    }

    if (flag.is_short) {
      required += flag.name + " ";
    } else {
      required += flag.name + "=<value> ";
    }
  }

  const std::regex binary_re("\\{binary\\}");
  const std::regex required_re("\\{required\\}");
  std::string usage = std::regex_replace(usage_format, binary_re, binary_name);
  usage = std::regex_replace(usage, required_re, required);
  std::cout << "USAGE: " << usage << std::endl << std::endl;
}

void FlagList::print_help(const char** argv, const std::string& usage_format, const std::string& title, const std::string& summary) {
  std::cout << title << std::endl << std::endl;
  print_usage(argv[0], usage_format);
  std::cout << summary << std::endl << std::endl;

  size_t longuest_flag = 0;
  for (const auto& flag : flags_) {
    longuest_flag = std::max(longuest_flag, flag.name.size());
  }

  std::cout << "OPTIONS:" << std::endl;
  for (const auto& flag : flags_) {
    const size_t inline_alignment = longuest_flag - flag.name.size() + 1;
    std::cout << "  " << flag.name << ":" << std::string(inline_alignment, ' ')
              << flag.help << std::endl;
  }
}

bool FlagList::parse(const char** argv) {
  flags::positional_arguments.clear();
  std::unordered_set<const FlagInfo*> parsed_flags;

  bool ignore_flags = false;
  for (const char** it = argv + 1; *it != nullptr; it++) {
    if (ignore_flags) {
      flags::positional_arguments.emplace_back(*it);
      continue;
    }

    // '--' alone is used to mark the end of the flags.
    if (std::strcmp(*it, "--") == 0) {
      ignore_flags = true;
      continue;
    }

    // '-' alone is not a flag, but often used to say 'stdin'.
    if (std::strcmp(*it, "-") == 0) {
      flags::positional_arguments.emplace_back(*it);
      continue;
    }

    const std::string raw_flag(*it);
    if (raw_flag.size() == 0) {
      continue;
    }

    if (raw_flag[0] != '-') {
      flags::positional_arguments.emplace_back(*it);
      continue;
    }

    // Only case left: flags (long and shorts).
    if (raw_flag.size() < 2) {
      std::cerr << "Unknown flag " << raw_flag << std::endl;
      return false;
    }
    const bool is_short_flag = std::strncmp(*it, "--", 2) != 0;
    const std::string flag_name = get_flag_name(raw_flag, is_short_flag);

    auto needle = find_if(
        flags_.begin(), flags_.end(),
        [&flag_name](const auto& item) { return item.name == flag_name; });
    if (needle == flags_.end()) {
      std::cerr << "Unknown flag " << flag_name << std::endl;
      return false;
    }

    if (parsed_flags.count(&*needle) != 0) {
      std::cerr << "The flag " << flag_name << " was specified multiple times."
                << std::endl;
      return false;
    }
    parsed_flags.insert(&*needle);

    if (!parse_flag_info(*needle, &it)) {
      std::cerr << "Invalid usage for flag " << flag_name << std::endl;
      return false;
    }
  }

  // Check that we parsed all required flags.
  for (const auto& flag : flags_) {
    if (!flag.required) {
      continue;
    }

    if (parsed_flags.count(&flag) == 0) {
      std::cerr << "Missing required flag " << flag.name << std::endl;
      return false;
    }
  }

  return true;
}

// Just the public wrapper around the parse function.
bool Parse(const char** argv) { return FlagList::parse(argv); }

// Just the public wrapper around the print_help function.
void PrintHelp(const char** argv, const std::string& usage_format, const std::string& title, const std::string& summary) {
  FlagList::print_help(argv, usage_format, title, summary);
}

}  // namespace flags
