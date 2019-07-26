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

#include "source/fuzz/transformation_copy_object.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationCopyObject::TransformationCopyObject(
    const protobufs::TransformationCopyObject& message)
    : message_(message) {}

TransformationCopyObject::TransformationCopyObject(uint32_t fresh_id,
                                                   uint32_t object,
                                                   uint32_t insert_after_id,
                                                   uint32_t offset) {
  message_.set_fresh_id(fresh_id);
  message_.set_object(object);
  message_.set_insert_after_id(insert_after_id);
  message_.set_offset(offset);
}

bool TransformationCopyObject::IsApplicable(
    opt::IRContext* context, const FactManager& fact_manager) const {
  if (!fuzzerutil::IsFreshId(context, message_.fresh_id())) {
    // We require the id for the object copy to be unused.
    return false;
  }
  // The id of the object to be copied must exist
  auto object_inst = context->get_def_use_mgr()->GetDef(
          message_.object());
  if (!object_inst) {
    return false;
  }
  if (!object_inst->type_id()) {
    // We can only apply OpCopyObject to instructions that have types.
    return false;
  }
  if (!context->get_decoration_mgr()->GetDecorationsFor(message_.object(), true).empty()) {
    // We do not copy objects that have decorations: if the copy is not decorated analogously,
    // using the original object vs. its copy may not be equivalent.
    // TODO(afd): it would be possible to make the copy but not add an id synonym.
    return false;
  }

  auto insert_after_inst = context->get_def_use_mgr()->GetDef(message_.insert_after_id());
  if (!insert_after_inst) {
    // The given id to insert after is not defined.
    return false;
  }

  auto destination_block = context->get_instr_block(insert_after_inst);
  if (!destination_block) {
    // The given id to insert after is not in a block.
    return false;
  }

//  if (!(boolean_constant->opcode() == SpvOpConstantFalse ||
//        boolean_constant->opcode() == SpvOpConstantTrue)) {
//    return false;
//  }

  // - |message_.insert_after_id| must be the result id of an instruction
  //   'base' in some block 'blk'.
  // - 'blk' must contain an instruction 'inst' located |message_.offset|
  //   instructions after 'base' (if |message_.offset| = 0 then 'inst' =
  //   'base').
  // - It must be legal to insert an OpCopyObject instruction directly
  //   before 'inst'.
  // - |message_object| must be available directly before 'inst'.
  (void)(context);
  (void)(fact_manager);
  return false;
}

void TransformationCopyObject::Apply(opt::IRContext* context,
                                     FactManager* fact_manager) const {
  (void)(context);
  (void)(fact_manager);
}

protobufs::Transformation TransformationCopyObject::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_copy_object() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
