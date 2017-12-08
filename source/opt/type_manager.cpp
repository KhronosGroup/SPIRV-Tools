// Copyright (c) 2016 Google Inc.
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

#include "type_manager.h"

#include <cassert>
#include <cstring>
#include <utility>

#include "ir_context.h"
#include "log.h"
#include "make_unique.h"
#include "reflect.h"

namespace spvtools {
namespace opt {
namespace analysis {

TypeManager::TypeManager(const MessageConsumer& consumer,
                         spvtools::ir::IRContext* c)
    : consumer_(consumer), context_(c) {
  AnalyzeTypes(*c->module());
}

Type* TypeManager::GetType(uint32_t id) const {
  auto iter = id_to_type_.find(id);
  if (iter != id_to_type_.end()) return (*iter).second.get();
  return nullptr;
}

std::pair<Type*, std::unique_ptr<Pointer>> TypeManager::GetTypeAndPointerType(
    uint32_t id, SpvStorageClass sc) const {
  Type* type = GetType(id);
  if (type) {
    return std::make_pair(type, MakeUnique<analysis::Pointer>(type, sc));
  } else {
    return std::make_pair(type, std::unique_ptr<analysis::Pointer>());
  }
}

uint32_t TypeManager::GetId(const Type* type) const {
  auto iter = type_to_id_.find(type);
  if (iter != type_to_id_.end()) return (*iter).second;
  return 0;
}

ForwardPointer* TypeManager::GetForwardPointer(uint32_t index) const {
  if (index >= forward_pointers_.size()) return nullptr;
  return forward_pointers_.at(index).get();
}

void TypeManager::AnalyzeTypes(const spvtools::ir::Module& module) {
  for (const auto* inst : module.GetTypes()) RecordIfTypeDefinition(*inst);
  for (const auto& inst : module.annotations()) AttachIfTypeDecoration(inst);
}

void TypeManager::RemoveId(uint32_t id) {
  auto iter = id_to_type_.find(id);
  if (iter == id_to_type_.end()) return;

  auto& type = iter->second;
  if (!type->IsUniqueType(true)) {
    // Search for an equivalent type to re-map.
    bool found = false;
    for (auto& pair : id_to_type_) {
      if (pair.first != id && *pair.second == *type) {
        // Equivalent ambiguous type, re-map type.
        type_to_id_.erase(type.get());
        type_to_id_[pair.second.get()] = pair.first;
        found = true;
        break;
      }
      // No equivalent ambiguous type, remove mapping.
      if (!found) type_to_id_.erase(type.get());
    }
  } else {
    // Unique type, so just erase the entry.
    type_to_id_.erase(type.get());
  }

  // Erase the entry for |id|.
  id_to_type_.erase(iter);
}

uint32_t TypeManager::GetTypeInstruction(const Type* type) {
  uint32_t id = GetId(type);
  if (id != 0) return id;

  std::unique_ptr<ir::Instruction> typeInst;
  id = context()->TakeNextId();
  RegisterType(id, *type);
  switch (type->kind()) {
#define DefineParameterlessCase(kind)                                          \
  case Type::k##kind:                                                          \
    typeInst.reset(new ir::Instruction(context(), SpvOpType##kind, 0, id,      \
                                       std::initializer_list<ir::Operand>{})); \
    break;
    DefineParameterlessCase(Void);
    DefineParameterlessCase(Bool);
    DefineParameterlessCase(Sampler);
    DefineParameterlessCase(Event);
    DefineParameterlessCase(DeviceEvent);
    DefineParameterlessCase(ReserveId);
    DefineParameterlessCase(Queue);
    DefineParameterlessCase(PipeStorage);
    DefineParameterlessCase(NamedBarrier);
#undef DefineParameterlessCase
    case Type::kInteger:
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypeInt, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {type->AsInteger()->width()}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(type->AsInteger()->IsSigned() ? 1u : 0u)}}}));
      break;
    case Type::kFloat:
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypeFloat, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {type->AsFloat()->width()}}}));
      break;
    case Type::kVector: {
      uint32_t subtype = GetTypeInstruction(type->AsVector()->element_type());
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeVector, 0, id,
                              std::initializer_list<ir::Operand>{
                                  {SPV_OPERAND_TYPE_ID, {subtype}},
                                  {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                                   {type->AsVector()->element_count()}}}));
      break;
    }
    case Type::kMatrix: {
      uint32_t subtype = GetTypeInstruction(type->AsMatrix()->element_type());
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeMatrix, 0, id,
                              std::initializer_list<ir::Operand>{
                                  {SPV_OPERAND_TYPE_ID, {subtype}},
                                  {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                                   {type->AsMatrix()->element_count()}}}));
      break;
    }
    case Type::kImage: {
      const Image* image = type->AsImage();
      uint32_t subtype = GetTypeInstruction(image->sampled_type());
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypeImage, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_ID, {subtype}},
              {SPV_OPERAND_TYPE_DIMENSIONALITY,
               {static_cast<uint32_t>(image->dim())}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {image->depth()}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(image->is_arrayed() ? 1u : 0u)}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(image->is_multisampled() ? 1u : 0u)}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {image->sampled()}},
              {SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT,
               {static_cast<uint32_t>(image->format())}},
              {SPV_OPERAND_TYPE_ACCESS_QUALIFIER,
               {static_cast<uint32_t>(image->access_qualifier())}}}));
      break;
    }
    case Type::kSampledImage: {
      uint32_t subtype =
          GetTypeInstruction(type->AsSampledImage()->image_type());
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeSampledImage, 0, id,
                              std::initializer_list<ir::Operand>{
                                  {SPV_OPERAND_TYPE_ID, {subtype}}}));
      break;
    }
    case Type::kArray: {
      uint32_t subtype = GetTypeInstruction(type->AsArray()->element_type());
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypeArray, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_ID, {subtype}},
              {SPV_OPERAND_TYPE_ID, {type->AsArray()->LengthId()}}}));
      break;
    }
    case Type::kRuntimeArray: {
      uint32_t subtype =
          GetTypeInstruction(type->AsRuntimeArray()->element_type());
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeRuntimeArray, 0, id,
                              std::initializer_list<ir::Operand>{
                                  {SPV_OPERAND_TYPE_ID, {subtype}}}));
      break;
    }
    case Type::kStruct: {
      std::vector<ir::Operand> ops;
      const Struct* structTy = type->AsStruct();
      for (auto ty : structTy->element_types()) {
        ops.push_back(
            ir::Operand(SPV_OPERAND_TYPE_ID, {GetTypeInstruction(ty)}));
      }
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeStruct, 0, id, ops));
      break;
    }
    case Type::kOpaque: {
      const Opaque* opaque = type->AsOpaque();
      size_t size = opaque->name().size();
      // Convert to null-terminated packed UTF-8 string.
      std::vector<uint32_t> words(size / 4 + 1, 0);
      char* dst = reinterpret_cast<char*>(words.data());
      strncpy(dst, opaque->name().c_str(), size);
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeOpaque, 0, id,
                              std::initializer_list<ir::Operand>{
                                  {SPV_OPERAND_TYPE_LITERAL_STRING, words}}));
      break;
    }
    case Type::kPointer: {
      const Pointer* pointer = type->AsPointer();
      uint32_t subtype = GetTypeInstruction(pointer->pointee_type());
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypePointer, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_STORAGE_CLASS,
               {static_cast<uint32_t>(pointer->storage_class())}},
              {SPV_OPERAND_TYPE_ID, {subtype}}}));
      break;
    }
    case Type::kFunction: {
      std::vector<ir::Operand> ops;
      const Function* function = type->AsFunction();
      ops.push_back(ir::Operand(SPV_OPERAND_TYPE_ID,
                                {GetTypeInstruction(function->return_type())}));
      for (auto ty : function->param_types()) {
        ops.push_back(
            ir::Operand(SPV_OPERAND_TYPE_ID, {GetTypeInstruction(ty)}));
      }
      typeInst.reset(
          new ir::Instruction(context(), SpvOpTypeFunction, 0, id, ops));
      break;
    }
    case Type::kPipe:
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypePipe, 0, id,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_ACCESS_QUALIFIER,
               {static_cast<uint32_t>(type->AsPipe()->access_qualifier())}}}));
      break;
    case Type::kForwardPointer:
      typeInst.reset(new ir::Instruction(
          context(), SpvOpTypeForwardPointer, 0, 0,
          std::initializer_list<ir::Operand>{
              {SPV_OPERAND_TYPE_ID, {type->AsForwardPointer()->target_id()}},
              {SPV_OPERAND_TYPE_STORAGE_CLASS,
               {static_cast<uint32_t>(
                   type->AsForwardPointer()->storage_class())}}}));
      break;
    default:
      assert(false && "Unexpected type");
      break;
  }
  context()->AddType(std::move(typeInst));
  context()->get_def_use_mgr()->AnalyzeInstDefUse(
      &*--context()->types_values_end());
  AttachDecorations(id, type);

  return id;
}

void TypeManager::AttachDecorations(uint32_t id, const Type* type) {
  for (auto vec : type->decorations()) {
    CreateDecoration(id, vec);
  }
  if (const Struct* structTy = type->AsStruct()) {
    for (auto pair : structTy->element_decorations()) {
      uint32_t element = pair.first;
      for (auto vec : pair.second) {
        CreateDecoration(id, vec, element);
      }
    }
  }
}

void TypeManager::CreateDecoration(uint32_t target,
                                   const std::vector<uint32_t>& decoration,
                                   uint32_t element) {
  std::vector<ir::Operand> ops;
  ops.push_back(ir::Operand(SPV_OPERAND_TYPE_ID, {target}));
  if (element != 0) {
    ops.push_back(ir::Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, {element}));
  }
  ops.push_back(ir::Operand(SPV_OPERAND_TYPE_DECORATION, {decoration[0]}));
  for (size_t i = 1; i < decoration.size(); ++i) {
    ops.push_back(
        ir::Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, {decoration[i]}));
  }
  context()->AddAnnotationInst(MakeUnique<ir::Instruction>(
      context(), (element == 0 ? SpvOpDecorate : SpvOpMemberDecorate), 0, 0,
      ops));
  ir::Instruction* inst = &*--context()->annotation_end();
  context()->get_def_use_mgr()->AnalyzeInstUse(inst);
}

void TypeManager::RegisterType(uint32_t id, const Type& type) {
  auto& t = id_to_type_[id];
  t.reset(type.Clone().release());
  if (GetId(t.get()) == 0) {
    type_to_id_[t.get()] = id;
  }
}

Type* TypeManager::RecordIfTypeDefinition(
    const spvtools::ir::Instruction& inst) {
  if (!spvtools::ir::IsTypeInst(inst.opcode())) return nullptr;

  Type* type = nullptr;
  switch (inst.opcode()) {
    case SpvOpTypeVoid:
      type = new Void();
      break;
    case SpvOpTypeBool:
      type = new Bool();
      break;
    case SpvOpTypeInt:
      type = new Integer(inst.GetSingleWordInOperand(0),
                         inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeFloat:
      type = new Float(inst.GetSingleWordInOperand(0));
      break;
    case SpvOpTypeVector:
      type = new Vector(GetType(inst.GetSingleWordInOperand(0)),
                        inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeMatrix:
      type = new Matrix(GetType(inst.GetSingleWordInOperand(0)),
                        inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeImage: {
      const SpvAccessQualifier access =
          inst.NumInOperands() < 8
              ? SpvAccessQualifierReadOnly
              : static_cast<SpvAccessQualifier>(inst.GetSingleWordInOperand(7));
      type = new Image(
          GetType(inst.GetSingleWordInOperand(0)),
          static_cast<SpvDim>(inst.GetSingleWordInOperand(1)),
          inst.GetSingleWordInOperand(2), inst.GetSingleWordInOperand(3) == 1,
          inst.GetSingleWordInOperand(4) == 1, inst.GetSingleWordInOperand(5),
          static_cast<SpvImageFormat>(inst.GetSingleWordInOperand(6)), access);
    } break;
    case SpvOpTypeSampler:
      type = new Sampler();
      break;
    case SpvOpTypeSampledImage:
      type = new SampledImage(GetType(inst.GetSingleWordInOperand(0)));
      break;
    case SpvOpTypeArray:
      type = new Array(GetType(inst.GetSingleWordInOperand(0)),
                       inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeRuntimeArray:
      type = new RuntimeArray(GetType(inst.GetSingleWordInOperand(0)));
      break;
    case SpvOpTypeStruct: {
      std::vector<Type*> element_types;
      for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
        element_types.push_back(GetType(inst.GetSingleWordInOperand(i)));
      }
      type = new Struct(element_types);
    } break;
    case SpvOpTypeOpaque: {
      const uint32_t* data = inst.GetInOperand(0).words.data();
      type = new Opaque(reinterpret_cast<const char*>(data));
    } break;
    case SpvOpTypePointer: {
      auto* ptr = new Pointer(
          GetType(inst.GetSingleWordInOperand(1)),
          static_cast<SpvStorageClass>(inst.GetSingleWordInOperand(0)));
      // Let's see if somebody forward references this pointer.
      for (auto* fp : unresolved_forward_pointers_) {
        if (fp->target_id() == inst.result_id()) {
          fp->SetTargetPointer(ptr);
          unresolved_forward_pointers_.erase(fp);
          break;
        }
      }
      type = ptr;
    } break;
    case SpvOpTypeFunction: {
      Type* return_type = GetType(inst.GetSingleWordInOperand(0));
      std::vector<Type*> param_types;
      for (uint32_t i = 1; i < inst.NumInOperands(); ++i) {
        param_types.push_back(GetType(inst.GetSingleWordInOperand(i)));
      }
      type = new Function(return_type, param_types);
    } break;
    case SpvOpTypeEvent:
      type = new Event();
      break;
    case SpvOpTypeDeviceEvent:
      type = new DeviceEvent();
      break;
    case SpvOpTypeReserveId:
      type = new ReserveId();
      break;
    case SpvOpTypeQueue:
      type = new Queue();
      break;
    case SpvOpTypePipe:
      type = new Pipe(
          static_cast<SpvAccessQualifier>(inst.GetSingleWordInOperand(0)));
      break;
    case SpvOpTypeForwardPointer: {
      // Handling of forward pointers is different from the other types.
      auto* fp = new ForwardPointer(
          inst.GetSingleWordInOperand(0),
          static_cast<SpvStorageClass>(inst.GetSingleWordInOperand(1)));
      forward_pointers_.emplace_back(fp);
      unresolved_forward_pointers_.insert(fp);
      return fp;
    }
    case SpvOpTypePipeStorage:
      type = new PipeStorage();
      break;
    case SpvOpTypeNamedBarrier:
      type = new NamedBarrier();
      break;
    default:
      SPIRV_UNIMPLEMENTED(consumer_, "unhandled type");
      break;
  }

  uint32_t id = inst.result_id();
  if (id == 0) {
    SPIRV_ASSERT(consumer_, inst.opcode() == SpvOpTypeForwardPointer,
                 "instruction without result id found");
  } else {
    SPIRV_ASSERT(consumer_, type != nullptr,
                 "type should not be nullptr at this point");
    id_to_type_[id].reset(type);
    type_to_id_[type] = id;
  }
  return type;
}

void TypeManager::AttachIfTypeDecoration(const ir::Instruction& inst) {
  const SpvOp opcode = inst.opcode();
  if (!ir::IsAnnotationInst(opcode)) return;
  const uint32_t id = inst.GetSingleWordOperand(0);
  // Do nothing if the id to be decorated is not for a known type.
  if (!id_to_type_.count(id)) return;

  Type* target_type = id_to_type_[id].get();
  switch (opcode) {
    case SpvOpDecorate: {
      const auto count = inst.NumOperands();
      std::vector<uint32_t> data;
      for (uint32_t i = 1; i < count; ++i) {
        data.push_back(inst.GetSingleWordOperand(i));
      }
      target_type->AddDecoration(std::move(data));
    } break;
    case SpvOpMemberDecorate: {
      const auto count = inst.NumOperands();
      const uint32_t index = inst.GetSingleWordOperand(1);
      std::vector<uint32_t> data;
      for (uint32_t i = 2; i < count; ++i) {
        data.push_back(inst.GetSingleWordOperand(i));
      }
      if (Struct* st = target_type->AsStruct()) {
        st->AddMemberDecoration(index, std::move(data));
      } else {
        SPIRV_UNIMPLEMENTED(consumer_, "OpMemberDecorate non-struct type");
      }
    } break;
    case SpvOpDecorationGroup:
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
      SPIRV_UNIMPLEMENTED(consumer_, "unhandled decoration");
      break;
    default:
      SPIRV_UNREACHABLE(consumer_);
      break;
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
