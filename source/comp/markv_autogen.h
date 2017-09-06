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

#ifndef LIBSPIRV_COMP_MARKV_AUTOGEN_H_
#define LIBSPIRV_COMP_MARKV_AUTOGEN_H_

#include <map>
#include <memory>
#include <numeric>
#include <unordered_set>

#include "util/huffman_codec.h"

inline uint64_t GetMarkvNonOfTheAbove() {
  // Magic number.
  return 1111111111111111111;
}

// Returns of histogram of CombineOpcodeAndNumOperands(opcode, num_operands).
std::map<uint64_t, uint32_t> GetOpcodeAndNumOperandsHist();

// Returns Huffman codecs based on a Markov chain of histograms of
// CombineOpcodeAndNumOperands(opcode, num_operands).
// Map prev_opcode -> codec.
std::map<uint32_t, std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>>
    GetOpcodeAndNumOperandsMarkovHuffmanCodecs();

// Returns Huffman codecs for literal strings.
// Map opcode -> codec.
std::map<uint32_t, std::unique_ptr<spvutils::HuffmanCodec<std::string>>>
    GetLiteralStringHuffmanCodecs();

// Returns Huffman codecs for single-word non-id operand slots.
// Map <opcode, operand_index> -> codec.
std::map<std::pair<uint32_t, uint32_t>,
    std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>>
    GetNonIdWordHuffmanCodecs();

// Returns Huffman codecs for id descriptors used by common operand slots.
// Map <opcode, operand_index> -> codec.
std::map<std::pair<uint32_t, uint32_t>,
    std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>>
    GetIdDescriptorHuffmanCodecs();

// Returns a set of all descriptors which are encodable by at least one codec
// returned by GetIdDescriptorHuffmanCodecs().
std::unordered_set<uint32_t> GetDescriptorsWithCodingScheme();

#endif  // LIBSPIRV_COMP_MARKV_AUTOGEN_H_
