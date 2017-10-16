
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

#ifndef LIBSPIRV_OPT_INSTRUCTION_LIST_H_
#define LIBSPIRV_OPT_INSTRUCTION_LIST_H_

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

#include "instruction.h"
#include "operand.h"
#include "util/ilist.h"

#include "spirv-tools/libspirv.h"
#include "spirv/1.2/spirv.h"

namespace spvtools {
namespace ir {

// This class is intended to be the container for Instructions.  This container
// owns the instructions that are in it.  When removing an Instruction from the
// list, the caller is assumming responsibility for deleting the storage.
class InstructionList : public utils::IntrusiveList<Instruction> {
 public:
  InstructionList() = default;
  InstructionList(InstructionList&& that)
      : utils::IntrusiveList<Instruction>(std::move(that)) {}
  InstructionList& operator=(InstructionList&& that) {
    auto p = static_cast<utils::IntrusiveList<Instruction>*>(this);
    *p = std::move(that);
    return *this;
  }

  class iterator : public utils::IntrusiveList<Instruction>::iterator {
   public:
    iterator(const utils::IntrusiveList<Instruction>::iterator& i)
        : utils::IntrusiveList<Instruction>::iterator(&*i) {}
    iterator(Instruction* i) : utils::IntrusiveList<Instruction>::iterator(i) {}

    // The nodes in |list| will be moved to the list that |this| points to.  The
    // positions of the nodes will be immediately before the element pointed to
    // by the iterator.  The return value will be an iterator pointing to the
    // first of the newly inserted elements.  Ownership if the elements in
    // |list| is now pass on to the new list that contains them.
    iterator InsertBefore(std::vector<std::unique_ptr<Instruction>>* list);

    // The node |i| will be inserted immediatly before |this|. The return value
    // will be an iterator pointing to the newly inserted node.  The owner of
    // |*i| becomes the list that contains it.
    iterator InsertBefore(std::unique_ptr<Instruction> i);
  };

  // Destroy this list and any instructions in the list.
  virtual ~InstructionList();
};

}  // namespace ir
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUCTION_LIST_H_
