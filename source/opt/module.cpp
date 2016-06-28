// Copyright (c) 2016 Google Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

#include "module.h"
#include "reflect.h"

namespace spvtools {
namespace ir {

std::vector<Instruction*> Module::types() {
  std::vector<Instruction*> insts;
  for (uint32_t i = 0; i < types_values_.size(); ++i) {
    if (IsTypeInst(types_values_[i].opcode()))
      insts.push_back(&types_values_[i]);
  }
  return insts;
};

void Module::ForEachInst(const std::function<void(Instruction*)>& f) {
  for (auto& i : capabilities_) f(&i);
  for (auto& i : extensions_) f(&i);
  for (auto& i : ext_inst_imports_) f(&i);
  f(&memory_model_);
  for (auto& i : entry_points_) f(&i);
  for (auto& i : execution_modes_) f(&i);
  for (auto& i : debugs_) f(&i);
  for (auto& i : annotations_) f(&i);
  for (auto& i : types_values_) f(&i);
  for (auto& i : functions_) i.ForEachInst(f);
}

void Module::ToBinary(std::vector<uint32_t>* binary, bool skip_nop) const {
  binary->push_back(header_.magic_number);
  binary->push_back(header_.version);
  // TODO(antiagainst): should we change the generator number?
  binary->push_back(header_.generator);
  binary->push_back(header_.bound);
  binary->push_back(header_.reserved);

  // TODO(antiagainst): wow, looks like a duplication of the above.
  for (const auto& c : capabilities_) c.ToBinary(binary, skip_nop);
  for (const auto& e : extensions_) e.ToBinary(binary, skip_nop);
  for (const auto& e : ext_inst_imports_) e.ToBinary(binary, skip_nop);
  memory_model_.ToBinary(binary, skip_nop);
  for (const auto& e : entry_points_) e.ToBinary(binary, skip_nop);
  for (const auto& e : execution_modes_) e.ToBinary(binary, skip_nop);
  for (const auto& d : debugs_) d.ToBinary(binary, skip_nop);
  for (const auto& a : annotations_) a.ToBinary(binary, skip_nop);
  for (const auto& t : types_values_) t.ToBinary(binary, skip_nop);
  for (const auto& f : functions_) f.ToBinary(binary, skip_nop);
}

}  // namespace ir
}  // namespace spvtools
