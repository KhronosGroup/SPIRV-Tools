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

#include "instruction.h"
#include "ir_context.h"

#include <initializer_list>

#include "ir_context.h"
#include "reflect.h"

namespace spvtools {
namespace ir {

namespace {
// Indices used to get particular operands out of instructions using InOperand.
const uint32_t kTypeImageDimIndex = 1;
const uint32_t kLoadBaseIndex = 0;
const uint32_t kVariableStorageClassIndex = 0;
const uint32_t kTypeImageSampledIndex = 5;
}  // namespace

Instruction::Instruction(IRContext* c)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(SpvOpNop),
      type_id_(0),
      result_id_(0),
      unique_id_(c->TakeNextUniqueId()) {}

Instruction::Instruction(IRContext* c, SpvOp op)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      type_id_(0),
      result_id_(0),
      unique_id_(c->TakeNextUniqueId()) {}

Instruction::Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
                         std::vector<Instruction>&& dbg_line)
    : context_(c),
      opcode_(static_cast<SpvOp>(inst.opcode)),
      type_id_(inst.type_id),
      result_id_(inst.result_id),
      unique_id_(c->TakeNextUniqueId()),
      dbg_line_insts_(std::move(dbg_line)) {
  assert((!IsDebugLineInst(opcode_) || dbg_line.empty()) &&
         "Op(No)Line attaching to Op(No)Line found");
  for (uint32_t i = 0; i < inst.num_operands; ++i) {
    const auto& current_payload = inst.operands[i];
    std::vector<uint32_t> words(
        inst.words + current_payload.offset,
        inst.words + current_payload.offset + current_payload.num_words);
    operands_.emplace_back(current_payload.type, std::move(words));
  }
}

Instruction::Instruction(IRContext* c, SpvOp op, uint32_t ty_id,
                         uint32_t res_id,
                         const std::vector<Operand>& in_operands)
    : utils::IntrusiveNodeBase<Instruction>(),
      context_(c),
      opcode_(op),
      type_id_(ty_id),
      result_id_(res_id),
      unique_id_(c->TakeNextUniqueId()),
      operands_() {
  if (type_id_ != 0) {
    operands_.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_TYPE_ID,
                           std::initializer_list<uint32_t>{type_id_});
  }
  if (result_id_ != 0) {
    operands_.emplace_back(spv_operand_type_t::SPV_OPERAND_TYPE_RESULT_ID,
                           std::initializer_list<uint32_t>{result_id_});
  }
  operands_.insert(operands_.end(), in_operands.begin(), in_operands.end());
}

Instruction::Instruction(Instruction&& that)
    : utils::IntrusiveNodeBase<Instruction>(),
      opcode_(that.opcode_),
      type_id_(that.type_id_),
      result_id_(that.result_id_),
      unique_id_(that.unique_id_),
      operands_(std::move(that.operands_)),
      dbg_line_insts_(std::move(that.dbg_line_insts_)) {}

Instruction& Instruction::operator=(Instruction&& that) {
  opcode_ = that.opcode_;
  type_id_ = that.type_id_;
  result_id_ = that.result_id_;
  unique_id_ = that.unique_id_;
  operands_ = std::move(that.operands_);
  dbg_line_insts_ = std::move(that.dbg_line_insts_);
  return *this;
}

Instruction* Instruction::Clone(IRContext* c) const {
  Instruction* clone = new Instruction(c);
  clone->opcode_ = opcode_;
  clone->type_id_ = type_id_;
  clone->result_id_ = result_id_;
  clone->unique_id_ = c->TakeNextUniqueId();
  clone->operands_ = operands_;
  clone->dbg_line_insts_ = dbg_line_insts_;
  return clone;
}

uint32_t Instruction::GetSingleWordOperand(uint32_t index) const {
  const auto& words = GetOperand(index).words;
  assert(words.size() == 1 && "expected the operand only taking one word");
  return words.front();
}

uint32_t Instruction::NumInOperandWords() const {
  uint32_t size = 0;
  for (uint32_t i = TypeResultIdCount(); i < operands_.size(); ++i)
    size += static_cast<uint32_t>(operands_[i].words.size());
  return size;
}

void Instruction::ToBinaryWithoutAttachedDebugInsts(
    std::vector<uint32_t>* binary) const {
  const uint32_t num_words = 1 + NumOperandWords();
  binary->push_back((num_words << 16) | static_cast<uint16_t>(opcode_));
  for (const auto& operand : operands_)
    binary->insert(binary->end(), operand.words.begin(), operand.words.end());
}

void Instruction::ReplaceOperands(const std::vector<Operand>& new_operands) {
  operands_.clear();
  operands_.insert(operands_.begin(), new_operands.begin(), new_operands.end());
  operands_.shrink_to_fit();
}

bool Instruction::IsReadOnlyLoad() const {
  if (IsLoad()) {
    ir::Instruction* address_def = GetBaseAddress();
    if (!address_def || address_def->opcode() != SpvOpVariable) {
      return false;
    }
    return address_def->IsReadOnlyVariable();
  }
  return false;
}

Instruction* Instruction::GetBaseAddress() const {
  assert(IsLoad() &&
         "GetBaseAddress should only be called on load instructions.");
  uint32_t base = GetSingleWordInOperand(kLoadBaseIndex);
  ir::Instruction* base_inst = context()->get_def_use_mgr()->GetDef(base);
  bool done = false;
  while (!done) {
    switch (base_inst->opcode()) {
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpPtrAccessChain:
      case SpvOpInBoundsPtrAccessChain:
      case SpvOpImageTexelPointer:
      case SpvOpCopyObject:
        // All of these instructions have the base pointer use a base pointer
        // in in-operand 0.
        base = base_inst->GetSingleWordInOperand(0);
        base_inst = context()->get_def_use_mgr()->GetDef(base);
        break;
      default:
        done = true;
        break;
    }
  }
  return base_inst;
}

bool Instruction::IsReadOnlyVariable() const {
  if (context()->module()->HasCapability(SpvCapabilityShader))
    return IsReadOnlyVariableShaders();
  else
    return IsReadOnlyVariableKernel();
}

bool Instruction::IsVulkanStorageImage() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  ir::Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) == SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage image.
  auto s = base_type->GetSingleWordInOperand(kTypeImageSampledIndex);
  return s != 1;
}

bool Instruction::IsVulkanSampledImage() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  ir::Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) == SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we know for sure that it is,
  // then return true.
  auto s = base_type->GetSingleWordInOperand(kTypeImageSampledIndex);
  return s == 1;
}

bool Instruction::IsVulkanStorageTexelBuffer() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniformConstant) {
    return false;
  }

  ir::Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeImage) {
    return false;
  }

  if (base_type->GetSingleWordInOperand(kTypeImageDimIndex) != SpvDimBuffer) {
    return false;
  }

  // Check if the image is sampled.  If we do not know for sure that it is,
  // then assume it is a storage texel buffer.
  return base_type->GetSingleWordInOperand(kTypeImageSampledIndex) != 1;
}

bool Instruction::IsVulkanStorageBuffer() const {
  // Is there a difference between a "Storage buffer" and a "dynamic storage
  // buffer" in SPIR-V and do we care about the difference?
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  ir::Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));

  if (base_type->opcode() != SpvOpTypeStruct) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class == SpvStorageClassUniform) {
    bool is_buffer_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), SpvDecorationBufferBlock,
        [&is_buffer_block](const ir::Instruction&) { is_buffer_block = true; });
    return is_buffer_block;
  } else if (storage_class == SpvStorageClassStorageBuffer) {
    bool is_block = false;
    context()->get_decoration_mgr()->ForEachDecoration(
        base_type->result_id(), SpvDecorationBlock,
        [&is_block](const ir::Instruction&) { is_block = true; });
    return is_block;
  }
  return false;
}

bool Instruction::IsVulkanUniformBuffer() const {
  if (opcode() != SpvOpTypePointer) {
    return false;
  }

  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  if (storage_class != SpvStorageClassUniform) {
    return false;
  }

  ir::Instruction* base_type =
      context()->get_def_use_mgr()->GetDef(GetSingleWordInOperand(1));
  if (base_type->opcode() != SpvOpTypeStruct) {
    return false;
  }

  bool is_block = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      base_type->result_id(), SpvDecorationBlock,
      [&is_block](const ir::Instruction&) { is_block = true; });
  return is_block;
}

bool Instruction::IsReadOnlyVariableShaders() const {
  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  Instruction* type_def = context()->get_def_use_mgr()->GetDef(type_id());

  switch (storage_class) {
    case SpvStorageClassUniformConstant:
      if (!type_def->IsVulkanStorageImage() &&
          !type_def->IsVulkanStorageTexelBuffer()) {
        return true;
      }
      break;
    case SpvStorageClassUniform:
      if (!type_def->IsVulkanStorageBuffer()) {
        return true;
      }
      break;
    case SpvStorageClassPushConstant:
    case SpvStorageClassInput:
      return true;
    default:
      break;
  }

  bool is_nonwritable = false;
  context()->get_decoration_mgr()->ForEachDecoration(
      result_id(), SpvDecorationNonWritable,
      [&is_nonwritable](const Instruction&) { is_nonwritable = true; });
  return is_nonwritable;
}

bool Instruction::IsReadOnlyVariableKernel() const {
  uint32_t storage_class = GetSingleWordInOperand(kVariableStorageClassIndex);
  return storage_class == SpvStorageClassUniformConstant;
}
}  // namespace ir
}  // namespace spvtools
