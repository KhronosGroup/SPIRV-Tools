// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fuzzer_pass_split_blocks.h"

#include <tuple>
#include <vector>

#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPassSplitBlocks::FuzzerPassSplitBlocks(
    opt::IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, fact_manager, fuzzer_context, transformations) {}

FuzzerPassSplitBlocks::~FuzzerPassSplitBlocks() = default;

void FuzzerPassSplitBlocks::Apply() {
  // Gather up pointers to all the blocks in the module.  We are then able to
  // iterate over these pointers and split the blocks to which they point;
  // we cannot safely split blocks while we iterate through the module.
  std::vector<opt::BasicBlock*> blocks;
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      blocks.push_back(&block);
    }
  }

  // Now go through all the block pointers that were gathered.
  for (auto& block : blocks) {
    // Probabilistically decide whether to try to split this block.
    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfSplittingBlock())) {
      // We are not going to try to split this block.
      continue;
    }
    // We are going to try to split this block.  We now need to choose where
    // to split it.  We describe the instruction before which we would like to
    // split a block via the opcode 'opc' of the relevant instruction, a base
    // instruction 'base' that has a result id, and the number of instructions
    // with opcode 'opc' that we should skip when searching from 'base' for the
    // desired instruction.  (When the instruction before which we would like to
    // split the block actually has a result id, the instruction is used as
    // base, 'opc' is its opcode, and there are 0 instructions to skip.)
    std::vector<std::tuple<uint32_t, SpvOp, uint32_t>> base_opcode_skip_triples;

    // The initial base instruction is the block label.
    uint32_t base = block->id();

    // Counts the number of times we have seen each opcode since we reset the
    // base instruction.
    std::map<SpvOp, uint32_t> skip_count;

    // Consider every instruction in the block.  The label is excluded: it is
    // only necessary to consider it as a base in case the first instruction
    // in the block does not have a result id.
    for (auto& inst : *block) {
      if (inst.HasResultId()) {
        // In the case that the instruction has a result id, we use the
        // instruction as its own base, and clear the skip counts we have
        // collected.
        base = inst.result_id();
        skip_count.clear();
      }
      const SpvOp opcode = inst.opcode();
      base_opcode_skip_triples.emplace_back(
          base, opcode, skip_count.count(opcode) ? skip_count.at(opcode) : 0);
      if (!inst.HasResultId()) {
        skip_count[opcode] =
            skip_count.count(opcode) ? skip_count.at(opcode) + 1 : 1;
      }
    }
    // Having identified all the places we might be able to split the block,
    // we choose one of them.
    std::tuple<uint32_t, SpvOp, uint32_t> base_opcode_skip =
        base_opcode_skip_triples[GetFuzzerContext()->RandomIndex(
            base_opcode_skip_triples)];
    auto transformation = TransformationSplitBlock(
        MakeInstructionDescriptor(std::get<0>(base_opcode_skip),
                                  std::get<1>(base_opcode_skip),
                                  std::get<2>(base_opcode_skip)),
        GetFuzzerContext()->GetFreshId());
    // If the position we have chosen turns out to be a valid place to split
    // the block, we apply the split. Otherwise the block just doesn't get
    // split.
    if (transformation.IsApplicable(GetIRContext(), *GetFactManager())) {
      transformation.Apply(GetIRContext(), GetFactManager());
      *GetTransformations()->add_transformation() = transformation.ToMessage();
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
