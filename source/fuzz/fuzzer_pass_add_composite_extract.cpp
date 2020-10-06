// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_add_composite_extract.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_composite_extract.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddCompositeExtract::FuzzerPassAddCompositeExtract(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddCompositeExtract::~FuzzerPassAddCompositeExtract() = default;

void FuzzerPassAddCompositeExtract::Apply() {
  /*
  std::vector<const protobufs::DataDescriptor*> available_synonyms;
  for (const auto* dd : GetTransformationContext()
                            ->GetFactManager()
                            ->GetAllSynonyms()) {
    // Note that it is possible to create OpCopmposite

    // Skip all invalid ids.
    if (!GetIRContext()->get_def_use_mgr()->GetDef(dd->object())) {
      continue;
    }

    available_synonyms.push_back(dd);
  }

  ForEachInstructionWithInstructionDescriptor(
      [this, &available_synonyms](
          opt::Function* function, opt::BasicBlock* block,
          opt::BasicBlock::iterator inst_it,
          const protobufs::InstructionDescriptor& instruction_descriptor) {
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                SpvOpCompositeExtract, inst_it)) {
          return;
        }

        // We can't create synonyms in dead blocks.
        if (GetTransformationContext()->GetFactManager()->BlockIsDead(block->id())) {
          return;
        }

        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAddingCompositeExtract())) {
          return;
        }


      });*/
}

}  // namespace fuzz
}  // namespace spvtools
