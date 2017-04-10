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

#include "stats_analyzer.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "source/enum_string_mapping.h"
#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "spirv/1.1/spirv.h"

using libspirv::SpirvStats;

namespace {

std::string GetVersionString(uint32_t word) {
  std::stringstream ss;
  ss << "Version " << SPV_SPIRV_VERSION_MAJOR_PART(word)
     << "." << SPV_SPIRV_VERSION_MINOR_PART(word);
  return ss.str();
}

std::string GetGeneratorString(uint32_t word) {
  return spvGeneratorStr(SPV_GENERATOR_TOOL_PART(word));
}

std::string GetOpcodeString(uint32_t word) {
  return spvOpcodeString(static_cast<SpvOp>(word));
}

std::string GetCapabilityString(uint32_t word) {
  return libspirv::CapabilityToString(static_cast<SpvCapability>(word));
}

std::string KeyIsLabel(std::string key) {
  return key;
}

template <class Key>
std::unordered_map<Key, double> GetRecall(
    const std::unordered_map<Key, uint32_t>& hist, uint64_t total) {
  std::unordered_map<Key, double> freq;
  for (const auto& pair : hist) {
    const double frequency =
        static_cast<double>(pair.second) / static_cast<double>(total);
    freq.emplace(pair.first, frequency);
  }
  return freq;
}

template <class Key>
std::unordered_map<Key, double> GetPrevalence(
    const std::unordered_map<Key, uint32_t>& hist) {
  uint64_t total = 0;
  for (const auto& pair : hist) {
    total += pair.second;
  }

  return GetRecall(hist, total);
}

template <class Key>
void WriteFreq(std::ostream& out, const std::unordered_map<Key, double>& freq,
               std::string (*label_from_key)(Key)) {
  std::vector<std::pair<Key, double>> sorted_freq(freq.begin(), freq.end());
  std::sort(sorted_freq.begin(), sorted_freq.end(),
            [](const std::pair<Key, double>& left,
               const std::pair<Key, double>& right) {
              return left.second > right.second;
            });

  for (const auto& pair : sorted_freq) {
    out << label_from_key(pair.first) << " " << pair.second * 100.0
        << "%" << std::endl;
  }
}

}  // namespace

StatsAnalyzer::StatsAnalyzer(const SpirvStats& stats) : stats_(stats) {
  num_modules_ = 0;
  for (const auto& pair : stats_.version_hist) {
    num_modules_ += pair.second;
  }

  version_freq_ = GetRecall(stats_.version_hist, num_modules_);
  generator_freq_ = GetRecall(stats_.generator_hist, num_modules_);
  capability_freq_ = GetRecall(stats_.capability_hist, num_modules_);
  extension_freq_ = GetRecall(stats_.extension_hist, num_modules_);
  opcode_freq_ = GetPrevalence(stats_.opcode_hist);
}

void StatsAnalyzer::WriteVersion(std::ostream& out) {
  WriteFreq(out, version_freq_, GetVersionString);
}

void StatsAnalyzer::WriteGenerator(std::ostream& out) {
  WriteFreq(out, generator_freq_, GetGeneratorString);
}

void StatsAnalyzer::WriteCapability(std::ostream& out) {
  WriteFreq(out, capability_freq_, GetCapabilityString);
}

void StatsAnalyzer::WriteExtension(std::ostream& out) {
  WriteFreq(out, extension_freq_, KeyIsLabel);
}

void StatsAnalyzer::WriteOpcode(std::ostream& out) {
  out << "Total unique opcodes used: " << opcode_freq_.size() << std::endl;
  WriteFreq(out, opcode_freq_, GetOpcodeString);
}
