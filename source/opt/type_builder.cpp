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

#include <algorithm>
#include <cassert>

#include "reflect.h"
#include "type_builder.h"

namespace spvtools {
namespace opt {
namespace type {

Type* TypeBuilder::CreateType(const ir::Inst& inst) {
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
      type.reset(new Integer(inst.GetSingleWordOperand(0),
                             inst.GetSingleWordOperand(1)));
      break;
    case SpvOpTypeFloat:
      type.reset(new Float(inst.GetSingleWordOperand(0)));
      break;
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeArray: {
      type.reset(new Vector(GetType(inst.GetSingleWordOperand(0)),
                            inst.GetSingleWordOperand(1)));
    } break;
    case SpvOpTypeStruct: {
      std::vector<Type*> element_types;
      for (uint32_t i = 0; i < inst.NumOperands(); ++i) {
        element_types.push_back(GetType(inst.GetSingleWordOperand(i)));
      }
      type.reset(new Struct(element_types));
    } break;
    case SpvOpTypeFunction: {
      Type* return_type = GetType(inst.GetSingleWordOperand(0));
      std::vector<Type*> param_types;
      for (uint32_t i = 1; i < inst.NumOperands(); ++i) {
        param_types.push_back(GetType(inst.GetSingleWordOperand(i)));
      }
      type.reset(new Function(return_type, param_types));
    } break;
    case SpvOpTypePointer: {
      type.reset(new Pointer(
          GetType(inst.GetSingleWordOperand(1)),
          static_cast<SpvStorageClass>(inst.GetSingleWordOperand(0))));
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

void TypeBuilder::AttachDecoration(const ir::Inst& inst) {
  const SpvOp opcode = inst.opcode();
  if (!ir::IsAnnotationInst(opcode)) return;
  const uint32_t id = inst.GetSingleWordPayload(0);
  // Do nothing if the id to be decorated is not for a known type.
  if (!type_map_->count(id)) return;

  Type* target_type = (*type_map_)[id].get();
  switch (opcode) {
    case SpvOpDecorate: {
      const auto count = inst.NumPayloads();
      std::vector<uint32_t> data;
      for (uint32_t i = 1; i < count; ++i) {
        data.push_back(inst.GetSingleWordPayload(i));
      }
      target_type->AddDecoration(std::move(data));
    } break;
    case SpvOpMemberDecorate: {
      const auto count = inst.NumPayloads();
      const uint32_t index = inst.GetSingleWordPayload(1);
      std::vector<uint32_t> data;
      for (uint32_t i = 2; i < count; ++i) {
        data.push_back(inst.GetSingleWordPayload(i));
      }
      if (Struct* st = target_type->AsStruct()) {
        st->AddMemeberDecoration(index, std::move(data));
      } else {
        assert(0 && "OpMemberDecorate on non-struct type");
      }
    } break;
    case SpvOpDecorationGroup:
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
      assert(0 && "unhandled decoration");
      break;
    default:
      assert(0 && "unreachable");
      break;
  }
}

std::vector<std::vector<uint32_t>> TypeBuilder::GroupSameTypes() const {
  std::vector<std::vector<uint32_t>> groups;
  const auto size = type_map_->size();

  std::vector<uint32_t> ids;
  for (const auto& t : *type_map_) ids.push_back(t.first);
  std::sort(ids.begin(), ids.end());
  std::vector<bool> id_done(size, false);

  // TODO(antiagainst): Well, this is not optimal. Try to improve it. Maybe
  // using some kind of hasing mechanism?
  for (uint32_t i = 0; i < size; ++i) {
    if (id_done[i]) continue;
    std::vector<uint32_t> group = {ids[i]};
    for (uint32_t j = i + 1; j < size; ++j) {
      if (id_done[j]) continue;
      Type* itype = (*type_map_)[ids[i]].get();
      Type* jtype = (*type_map_)[ids[j]].get();
      if (itype->IsSame(jtype)) {
        group.push_back(ids[j]);
        id_done[j] = true;
      }
    }
    groups.push_back(std::move(group));
  }

  return groups;
}

Type* TypeBuilder::GetType(uint32_t id) const {
  assert(type_map_->count(id) && "id for element type not found");
  return (*type_map_)[id].get();
}

}  // namespace type
}  // namespace opt
}  // namespace spvtools
