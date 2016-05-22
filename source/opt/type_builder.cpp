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

#include <cassert>

#include "type_builder.h"

namespace spvtools {
namespace opt {
namespace type {

const Type* TypeBuilder::CreateType(const ir::Inst& inst) {
  const uint32_t id = inst.result_id();
  assert(type_map_->count(id) == 0 && "id taken by another type");

  auto& type = (*type_map_)[id];
  switch (inst.opcode()) {
    case SpvOpTypeVoid:
      type.reset(new Void());
      break;
    case SpvOpTypeBool:
      type.reset(new Bool());
      break;
    case SpvOpTypeInt:
      type.reset(new Integer(inst.GetOperandWord(0), inst.GetOperandWord(1)));
      break;
    case SpvOpTypeFloat:
      type.reset(new Float(inst.GetOperandWord(0)));
      break;
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeArray: {
      type.reset(
          new Vector(GetType(inst.GetOperandWord(0)), inst.GetOperandWord(1)));
    } break;
    case SpvOpTypeStruct: {
      const uint32_t num_operand_words = inst.NumOperandWord();
      std::vector<const Type*> element_types;
      for (uint32_t i = 0; i < num_operand_words; ++i) {
        element_types.push_back(GetType(inst.GetOperandWord(i)));
      }
      type.reset(new Struct(element_types));
    } break;
    case SpvOpTypeFunction: {
      const uint32_t num_operand_words = inst.NumOperandWord();
      const Type* return_type = GetType(inst.GetOperandWord(0));
      std::vector<const Type*> param_types;
      for (uint32_t i = 1; i < num_operand_words; ++i) {
        param_types.push_back(GetType(inst.GetOperandWord(i)));
      }
      type.reset(new Function(return_type, param_types));
    } break;
    case SpvOpTypePointer: {
      type.reset(
          new Pointer(GetType(inst.GetOperandWord(1)),
                      static_cast<SpvStorageClass>(inst.GetOperandWord(0))));
    } break;
    case SpvOpTypeImage:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeRuntimeArray:
    case SpvOpTypeOpaque:
    case SpvOpTypeEvent:
    case SpvOpTypeDeviceEvent:
    case SpvOpTypeReserveId:
    case SpvOpTypeQueue:
    case SpvOpTypePipe:
    case SpvOpTypeForwardPointer:
    case SpvOpTypePipeStorage:
    case SpvOpTypeNamedBarrier:
      assert(0 && "unhandled type");
      break;
    default:
      assert(0 && "expected type-declaring instruction");
      break;
  }

  return type.get();
}

const Type* TypeBuilder::GetType(uint32_t id) const {
  assert(type_map_->count(id) && "id for element type not found");
  return (*type_map_)[id].get();
}

}  // namespace type
}  // namespace opt
}  // namespace spvtools
