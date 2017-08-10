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

#include "id_descriptor.h"

#include <cassert>
#include <iostream>

#include "opcode.h"
#include "operand.h"

namespace libspirv {

namespace {

// Hashes an array of words. Order of words is important.
uint32_t HashU32Array(const std::vector<uint32_t>& words) {
  // The hash function is a sum of hashes of each word seeded by word index.
  // Knuth's multiplicative hash is used to hash the words.
  const uint32_t kKnuthMulHash = 2654435761;
  uint32_t val = 0;
  for (uint32_t i = 0; i < words.size(); ++i) {
    val += (words[i] + i + 123) * kKnuthMulHash;
  }
  return val;
}

}  // namespace

uint32_t IdDescriptorCollection::ProcessInstruction(
    const spv_parsed_instruction_t& inst) {
  if (!inst.result_id)
    return 0;

  assert(words_.empty());
  words_.push_back(inst.words[0]);

  for (size_t operand_index = 0; operand_index < inst.num_operands;
       ++operand_index) {
    const auto &operand = inst.operands[operand_index];
    if (spvIsIdType(operand.type)) {
      const uint32_t id = inst.words[operand.offset];
      const auto it = id_to_descriptor_.find(id);
      // Forward declared ids are not hashed.
      if (it != id_to_descriptor_.end()) {
        words_.push_back(it->second);
      }
    } else {
      for (size_t operand_word_index = 0;
           operand_word_index < operand.num_words; ++operand_word_index) {
        words_.push_back(inst.words[operand.offset + operand_word_index]);
      }
    }
  }

  const uint32_t descriptor = HashU32Array(words_);
  assert(descriptor);

  words_.clear();

  const auto result = id_to_descriptor_.emplace(inst.result_id, descriptor);
  assert(result.second);
  (void)result;
  return descriptor;
}

}  // namespace libspirv
