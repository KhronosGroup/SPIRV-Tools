// Copyright (c) 2022 Google LLC
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

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

#include <cstdint>
#include <string>
#include <iostream>
#include <string>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>
using namespace emscripten;


void print_msg_to_stderr (spv_message_level_t, const char*,
                          const spv_position_t&, const char* m) {
  std::cerr << "error: " << m << std::endl;
}

std::vector<uint32_t> aligned_copy(const std::string& str) {
  std::vector<uint32_t> result(str.size()/4);
  std::memcpy(result.data(), str.data(), str.size());
  return result;
}

struct Optimizer {
 public:
  Optimizer(unsigned env) : optimizer_(static_cast<spv_target_env>(env)) {
    optimizer_.SetMessageConsumer(print_msg_to_stderr);
  }
  void registerPerformancePasses() { optimizer_.RegisterPerformancePasses(); }
  void registerSizePasses() { optimizer_.RegisterSizePasses(); }
  std::string run(std::string const& binary) {
    auto copy = aligned_copy(binary);
    std::vector<uint32_t> optimized_binary;
    optimizer_.Run(copy.data(), copy.size(), &optimized_binary);
    const auto num_output_bytes = optimized_binary.size()*4;
    return std::string(reinterpret_cast<char*>(optimized_binary.data()),num_output_bytes);
  }

  ::spvtools::Optimizer optimizer_;
};

std::string optimizePerformance(unsigned env, std::string binary) {
  Optimizer o(env);
  o.registerPerformancePasses();
  return o.run(binary);
}

std::string optimizeSize(unsigned env, std::string binary) {
  Optimizer o(env);
  o.registerSizePasses();
  return o.run(binary);
}

EMSCRIPTEN_BINDINGS(SpirvToolsOpt) {
  function("optimizePerformance",&optimizePerformance);
  function("optimizeSize",&optimizeSize);

#include "spv_env.inc"
}
