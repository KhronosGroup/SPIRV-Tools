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

#ifdef __linux__
#define _XOPEN_SOURCE 500
#include <ftw.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <Windows.h>
#endif


#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "source/spirv_stats.h"
#include "spirv-tools/libspirv.hpp"
#include "stats_analyzer.h"
#include "tools/io.h"

using libspirv::SpirvStats;

namespace {

std::vector<std::string> g_input_files;

void AddInputFile(const char* path) {
  g_input_files.push_back(path);
}

#ifdef __linux__
int AddInputFile(const char* path, const struct stat*, int tflag, struct FTW*) {
  if (tflag == FTW_F)
    AddInputFile(path);
  return 0;
}
#endif  // __linux__

bool IsDir(const char* path) {
#ifdef __linux__
  struct stat path_stat;
  stat(path, &path_stat);
  return S_ISDIR(path_stat.st_mode);
#elif defined(_WIN32)
  return FILE_ATTRIBUTE_DIRECTORY & GetFileAttributes(path);
#else
  return false;
#endif
}

void AddInputDir(const char* path) {
#ifdef __linux__
  nftw(path, AddInputFile, 20, FTW_PHYS);
#endif  // __linux__
}

void PrintUsage(char* argv0) {
  printf(
      R"(%s - Collect statistics from one or more SPIR-V binary file(s).

USAGE: %s [options] [<filepaths>] [<dirpaths>]

The SPIR-V binaries read from the given <filepaths> combined with the files
found in the given <dirpaths> and their subdirectories.

<dirpaths> feature is currently only implmeneted in Linux.

Options:
  -h, --help                       Print this help.
)",
      argv0, argv0);
}

void DiagnosticsMessageHandler(spv_message_level_t level, const char*,
                               const spv_position_t& position,
                               const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool continue_processing = true;
  int return_code = 0;

  std::vector<const char*> arg_paths;

  for (int argi = 1; continue_processing && argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        continue_processing = false;
        return_code = 0;
      } else {
        PrintUsage(argv[0]);
        continue_processing = false;
        return_code = 1;
      }
    } else {
      arg_paths.push_back(cur_arg);
    }
  }

  // Exit if command line parsing was not successful.
  if (!continue_processing) {
    return return_code;
  }

  for (const char* path : arg_paths) {
    if (IsDir(path)) {
#ifndef __linux__
      std::cerr << "error: Directory traversal only implemented on Linux"
                << std::endl;
      return 1;
#endif  // __linux__
      AddInputDir(path);
    } else {
      AddInputFile(path);
    }
  }

  std::cerr << "Processing " << g_input_files.size()
            << " files..." << std::endl;

  spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_1);
  tools.SetMessageConsumer(DiagnosticsMessageHandler);

  libspirv::SpirvStats stats;

  for (size_t index = 0; index < g_input_files.size(); ++index) {
    constexpr size_t kMilestonePeriod = 1000;
    if (index % kMilestonePeriod == kMilestonePeriod - 1)
      std::cerr << "Processed " << index + 1 << " files..." << std::endl;

    const std::string& path = g_input_files[index];
    std::vector<uint32_t> contents;
    if (!ReadFile<uint32_t>(path.c_str(), "rb", &contents)) return 1;

    if (!tools.AggregateStats(contents.data(), contents.size(), &stats)) {
      std::cerr << "error: Failed to aggregate stats for " << path << std::endl;
      return 1;
    }
  }

  StatsAnalyzer analyzer(stats);

  std::ostream& out = std::cout;

  out << std::endl;
  analyzer.WriteVersion(out);
  analyzer.WriteGenerator(out);

  out << std::endl;
  analyzer.WriteCapability(out);

  out << std::endl;
  analyzer.WriteExtension(out);

  out << std::endl;
  analyzer.WriteOpcode(out);

  return 0;
}
