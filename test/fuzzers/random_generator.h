// Copyright (c) 2021 Google Inc.
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

#ifndef TEST_FUZZERS_RANDOM_GENERATOR_H_
#define TEST_FUZZERS_RANDOM_GENERATOR_H_

#include <cstdint>
#include <random>

#include "source/spirv_target_env.h"

namespace spvtools {
namespace fuzzers {

/// Pseudo random generator utility class for fuzzing
class RandomGenerator {
 public:
  /// @brief Initializes the internal engine
  /// @param seed - seed value passed to engine
  explicit RandomGenerator(uint64_t seed);

  /// @brief Initializes the internal engine
  /// @param data - data to calculate the seed from
  /// @param size - size of the data
  explicit RandomGenerator(const uint8_t* data, size_t size);

  ~RandomGenerator() {}

  /// Calculate a seed value based on a blob of data.
  /// Currently hashes bytes near the front of the buffer, after skipping N
  /// bytes.
  /// @param data - pointer to data to base calculation off of, must be !nullptr
  /// @param size - number of elements in |data|, must be > 0
  static uint64_t CalculateSeed(const uint8_t* data, size_t size);

  /// Get random target env.
  spv_target_env GetTargetEnv();

 private:
  std::mt19937 engine_;
};  // class RandomGenerator

}  // namespace fuzzers
}  // namespace spvtools

#endif  // TEST_FUZZERS_RANDOM_GENERATOR_UTILS_H_
