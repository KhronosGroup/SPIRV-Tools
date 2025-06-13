// Copyright (c) 2025 LunarG Inc.
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

#include "source/opt/remap_ids_pass.h"

#include <algorithm>
#include <limits>

namespace spvtools {
namespace opt {

Pass::Status RemapIdsPass::Process() {
  // initialize the new id map
  new_id_.resize(GetBound(), unused_);

  // scan the ids and set to unmapped
  ScanIds();

  // create new ids for types and consts
  RemapTypeAndConst();

  // create new ids for names
  RemapNames();

  // create new ids for functions
  RemapFunctions();

  // create new_ids for everything else
  RemapRemainders();

  // apply mapping
  ApplyMap();

  // update bound in the header
  UpdateBound();

  return Status::SuccessWithChange;
}

void RemapIdsPass::ScanIds() {
  get_module()->ForEachInst(
      [this](Instruction* inst) {
        // look for types and consts
        if (IsTypeOp(inst->opcode()) || IsConstOp(inst->opcode())) {
          type_and_const_ids_.push_back(inst->result_id());
          SetNewId(inst->result_id(), unmapped_);
        }
        // look for names.
        else if (inst->opcode() == spv::Op::OpName) {
          // store name string in map so that we can compute the hash later
          auto const name = inst->GetOperand(1).AsString();
          auto const target = inst->GetSingleWordInOperand(0);
          name_ids_[name] = target;
          SetNewId(target, unmapped_);
        }
        // look for function ids
        else if (inst->opcode() == spv::Op::OpFunction) {
          auto const res_id = inst->result_id();
          function_ids_.push_back(res_id);
          SetNewId(res_id, unmapped_);
        }
        // look for remaining result ids
        else if (inst->HasResultId()) {
          auto const res_id = inst->result_id();
          remainder_ids.push_back(res_id);
          SetNewId(res_id, unmapped_);
        }
      },
      true);
}

void RemapIdsPass::RemapTypeAndConst() {
  // remap type ids
  static constexpr std::uint32_t soft_type_id_limit = 3011;  // small prime.
  static constexpr std::uint32_t first_mapped_id = 8;  // offset into ID space
  for (auto const id : type_and_const_ids_) {
    // compute the hash value
    auto const hash_value =
        (HashTypeAndConst(id) % soft_type_id_limit) + first_mapped_id;

    if (IsOldIdUnmapped(id)) {
      SetNewId(id, NextUnusedNewId(hash_value));
    }
  }
}

// Hash types to canonical values.  This can return ID collisions (it's a bit
// inevitable): it's up to the caller to handle that gracefully.
spv::Id RemapIdsPass::HashTypeAndConst(spv::Id const id) const {
  spv::Id value = 0;

  auto const inst = get_def_use_mgr()->GetDef(id);
  auto const op_code = inst->opcode();
  switch (op_code) {
    case spv::Op::OpTypeVoid:
      value = 0;
      break;
    case spv::Op::OpTypeBool:
      value = 1;
      break;
    case spv::Op::OpTypeInt: {
      auto const signedness = inst->GetSingleWordOperand(2);
      value = 3 + signedness;
      break;
    }
    case spv::Op::OpTypeFloat:
      value = 5;
      break;
    case spv::Op::OpTypeVector: {
      auto const component_type = inst->GetSingleWordOperand(1);
      auto const component_count = inst->GetSingleWordOperand(2);
      value = 6 + HashTypeAndConst(component_type) * (component_count - 1);
      break;
    }
    case spv::Op::OpTypeMatrix: {
      auto const column_type = inst->GetSingleWordOperand(1);
      auto const column_count = inst->GetSingleWordOperand(2);
      value = 30 + HashTypeAndConst(column_type) * (column_count - 1);
      break;
    }
    case spv::Op::OpTypeImage: {
      // TODO: Why isn't the format used to compute the hash value?
      auto const sampled_type = inst->GetSingleWordOperand(1);
      auto const dim = inst->GetSingleWordOperand(2);
      auto const depth = inst->GetSingleWordOperand(3);
      auto const arrayed = inst->GetSingleWordOperand(4);
      auto const ms = inst->GetSingleWordOperand(5);
      auto const sampled = inst->GetSingleWordOperand(6);
      value = 120 + HashTypeAndConst(sampled_type) + dim + depth * 8 * 16 +
              arrayed * 4 * 16 + ms * 2 * 16 + sampled * 1 * 16;
      break;
    }
    case spv::Op::OpTypeSampler:
      value = 500;
      break;
    case spv::Op::OpTypeSampledImage:
      value = 502;
      break;
    case spv::Op::OpTypeArray: {
      auto const element_type = inst->GetSingleWordOperand(1);
      auto const length = inst->GetSingleWordOperand(2);
      value = 501 + HashTypeAndConst(element_type) * length;
      break;
    }
    case spv::Op::OpTypeRuntimeArray: {
      auto const element_type = inst->GetSingleWordOperand(1);
      value = 5000 + HashTypeAndConst(element_type);
      break;
    }
    case spv::Op::OpTypeStruct:
      value = 10000;
      for (uint32_t w = 1; w < inst->NumOperandWords(); ++w) {
        value += (w + 1) * HashTypeAndConst(inst->GetSingleWordOperand(w));
      }
      break;
    case spv::Op::OpTypeOpaque: {
      // TODO: name is a literal that may have more than one word.
      auto const name = inst->GetSingleWordOperand(1);
      value = 6000 + name;
      break;
    }
    case spv::Op::OpTypePointer: {
      auto const type = inst->GetSingleWordOperand(2);
      value = 100000 + HashTypeAndConst(type);
      break;
    }
    case spv::Op::OpTypeFunction:
      value = 200000;
      for (uint32_t w = 1; w < inst->NumOperandWords(); ++w) {
        value += (w + 1) * HashTypeAndConst(inst->GetSingleWordOperand(w));
      }
      break;
    case spv::Op::OpTypeEvent:
      value = 300000;
      break;
    case spv::Op::OpTypeDeviceEvent:
      value = 300001;
      break;
    case spv::Op::OpTypeReserveId:
      value = 300002;
      break;
    case spv::Op::OpTypeQueue:
      value = 300003;
      break;
    case spv::Op::OpTypePipe:
      value = 300004;
      break;
    case spv::Op::OpConstantTrue:
      value = 300007;
      break;
    case spv::Op::OpConstantFalse:
      value = 300008;
      break;
    case spv::Op::OpTypeRayQueryKHR:
      value = 300009;
      break;
    case spv::Op::OpTypeAccelerationStructureKHR:
      value = 300010;
      break;
    case spv::Op::OpConstantComposite: {
      auto const result_type = inst->GetSingleWordOperand(0);
      value = 300011 + HashTypeAndConst(result_type);
      for (uint32_t w = 2; w < inst->NumOperandWords(); ++w) {
        value += (w + 1) * HashTypeAndConst(inst->GetSingleWordOperand(w));
      }
      break;
    }
    case spv::Op::OpConstant: {
      auto const result_type = inst->GetSingleWordOperand(0);
      value = 400011 + HashTypeAndConst(result_type);
      auto const literal = inst->GetOperand(2);
      for (uint32_t w = 0; w < literal.words.size(); ++w) {
        value += (w + 3) * literal.words[w];
      }
      break;
    }
    case spv::Op::OpConstantNull: {
      auto const result_type = inst->GetSingleWordOperand(0);
      value = 500009 + HashTypeAndConst(result_type);
      break;
    }
    case spv::Op::OpConstantSampler: {
      auto const result_type = inst->GetSingleWordOperand(0);
      value = 600011 + HashTypeAndConst(result_type);
      for (uint32_t w = 2; w < inst->NumOperandWords(); ++w) {
        value += (w + 1) * inst->GetSingleWordOperand(w);
      }
      break;
    }
    default:
      context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0},
                            "unknown type opcode");
      break;
  }

  return value;
}

void RemapIdsPass::RemapNames() {
  static constexpr std::uint32_t soft_type_id_limit = 3011;  // small prime.
  static constexpr std::uint32_t first_mapped_id =
      3019;  // offset into ID space

  for (auto const& [name, target] : name_ids_) {
    spv::Id hash_value = 1911;
    for (const char c : name) {
      hash_value = hash_value * 1009 + c;
    }

    if (IsOldIdUnmapped(target)) {
      SetNewId(target, NextUnusedNewId(hash_value % soft_type_id_limit +
                                       first_mapped_id));
    }
  }
}

void RemapIdsPass::RemapFunctions() {
  static constexpr std::uint32_t soft_type_id_limit = 19071;  // small prime.
  static constexpr std::uint32_t first_mapped_id =
      6203;  // offset into ID space
  // Window size for context-sensitive canonicalization values
  // Empirical best size from a single data set.  TODO: Would be a good tunable.
  // We essentially perform a little convolution around each instruction,
  // to capture the flavor of nearby code, to hopefully match to similar
  // code in other modules.
  static const int32_t window_size = 2;

  for (auto const func_id : function_ids_) {
    // store the instructions and opcode hash values in vectors so that the
    // window of instructions can be easily accessed and avoid having to
    // recompute the hash value repeatedly in overlapping windows
    std::vector<Instruction*> insts;
    std::vector<uint32_t> opcode_hashvals;
    auto const func = context()->GetFunction(func_id);
    func->WhileEachInst([&](Instruction* inst) {
      insts.emplace_back(inst);
      opcode_hashvals.emplace_back(HashOpCode(inst));
      return true;
    });

    // perform the convolution over the window of instructions
    assert(insts.size() < (size_t)std::numeric_limits<int32_t>::max());
    for (int32_t i = 0; i < (int32_t)insts.size(); ++i) {
      auto const inst = insts[i];
      if (!inst->HasResultId()) {
        continue;
      }

      auto const old_id = inst->result_id();
      if (!IsOldIdUnmapped(old_id)) {
        continue;
      }

      int32_t const lower_bound = std::max(0, i - window_size);
      int32_t const upper_bound =
          std::min((int32_t)insts.size() - 1, i + window_size);
      spv::Id hash_value = func_id * 17;  // small prime
      // convolve preceding instructions
      for (int32_t j = i - 1; j >= lower_bound; --j) {
        // don't convolve outside of the function
        auto const local_inst = insts[j];
        if (local_inst->opcode() == spv::Op::OpFunction) {
          break;
        }

        hash_value = hash_value * 30103 +
                     opcode_hashvals[j];  // 30103 = semiarbitrary prime
      }
      // convolve following instructions
      for (int32_t j = i; j <= upper_bound; ++j) {
        // don't convolve outside of the function
        auto const local_inst = insts[j];
        if (local_inst->opcode() == spv::Op::OpFunctionEnd) {
          break;
        }

        hash_value = hash_value * 30103 +
                     opcode_hashvals[j];  // 30103 = semiarbitrary prime
      }

      SetNewId(old_id, NextUnusedNewId(hash_value % soft_type_id_limit +
                                       first_mapped_id));
    }
  }
}

spv::Id RemapIdsPass::HashOpCode(Instruction const* const inst) const {
  auto const op_code = inst->opcode();
  std::uint32_t offset = 0;
  if (op_code == spv::Op::OpExtInst) {
    // offset is literal instruction
    offset = inst->GetSingleWordOperand(3);
  }

  return (std::uint32_t)op_code * 19 + offset;  // 19 = small prime
}

// Assign remaining ids sequentially from remaining holes in the new id space.
void RemapIdsPass::RemapRemainders() {
  spv::Id new_id = 1;
  for (auto const old_id : remainder_ids) {
    if (IsOldIdUnmapped(old_id)) {
      SetNewId(old_id, new_id = NextUnusedNewId(new_id));
    }
  }
}

void RemapIdsPass::ApplyMap() {
  context()->module()->ForEachInst(
      [this](Instruction* inst) {
        for (auto operand = inst->begin(); operand != inst->end(); ++operand) {
          const auto type = operand->type;
          if (spvIsIdType(type)) {
            uint32_t& id = operand->words[0];
            uint32_t const new_id = GetNewId(id);
            if (new_id == unused_) {
              continue;
            }

            assert(new_id != unmapped_ && "new_id should not be unmapped_");

            if (id != new_id) {
              id = new_id;
              if (type == SPV_OPERAND_TYPE_RESULT_ID) {
                inst->SetResultId(new_id);
              } else if (type == SPV_OPERAND_TYPE_TYPE_ID) {
                inst->SetResultType(new_id);
              }
            }
          }
        }
      },
      true);
}

// Return true if this opcode defines a type
bool RemapIdsPass::IsTypeOp(spv::Op const opCode) const {
  bool is_type_op = false;
  switch (opCode) {
    case spv::Op::OpTypeVoid:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeImage:
    case spv::Op::OpTypeSampler:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypeOpaque:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeEvent:
    case spv::Op::OpTypeDeviceEvent:
    case spv::Op::OpTypeReserveId:
    case spv::Op::OpTypeQueue:
    case spv::Op::OpTypeSampledImage:
    case spv::Op::OpTypePipe:
      is_type_op = true;
      break;
    default:
      break;
  }
  return is_type_op;
}

// Return true if this opcode defines a constant
bool RemapIdsPass::IsConstOp(spv::Op const opCode) const {
  bool is_const_op = false;
  switch (opCode) {
    case spv::Op::OpConstantSampler:
      context()->consumer()(SPV_MSG_ERROR, "", {0, 0, 0},
                            "unimplemented constant type");
      is_const_op = true;
      break;
    case spv::Op::OpConstantNull:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstant:
      is_const_op = true;
      break;
    default:
      break;
  }
  return is_const_op;
}

spv::Id RemapIdsPass::GetBound() const {
  return context()->module()->id_bound();
}

void RemapIdsPass::UpdateBound() {
  context()->module()->SetIdBound(context()->module()->ComputeIdBound());

  context()->ResetFeatureManager();
}

spv::Id RemapIdsPass::NextUnusedNewId(spv::Id id) const {
  // search for an unused id
  while (IsNewIdClaimed(id)) {
    ++id;
  }

  return id;
}

void RemapIdsPass::SetNewId(spv::Id const old_id, spv::Id const new_id) {
  assert(old_id < GetBound() && "don't remap an ID that is out of bounds");

  if (old_id >= new_id_.size()) {
    new_id_.resize(old_id + 1, unused_);
  }

  if (new_id != unmapped_ && new_id != unused_) {
    assert(!IsOldIdUnused(old_id) && "don't remap unused IDs");
    assert(IsOldIdUnmapped(old_id) && "don't remap already mapped IDs");
    assert(!IsNewIdClaimed(new_id) &&
           "don't remap to an ID that is already claimed");

    ClaimNewId(new_id);
  }

  new_id_[old_id] = new_id;
}

std::string RemapIdsPass::IdAsString(spv::Id const id) const {
  if (id == unused_) {
    return "unused";
  } else if (id == unmapped_) {
    return "unmapped";
  } else {
    return std::to_string(id);
  }
}

void RemapIdsPass::PrintNewIds() const {
  for (spv::Id id = 0; id < new_id_.size(); ++id) {
    auto const message =
        "new id[" + IdAsString(id) + "]: " + IdAsString(new_id_[id]);
    context()->consumer()(SPV_MSG_INFO, "", {0, 0, 0}, message.c_str());
  }
}

}  // namespace opt
}  // namespace spvtools