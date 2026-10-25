// Copyright (c) 2025 Lee Gao
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

#include "source/opt/mali_optimization_barrier_pass.h"

#include <cstdio>
#include <memory>
#include <vector>

#include "source/opt/ir_builder.h"

#define LOG_(level, fmt, ...)                                         \
  do {                                                                \
    char buffer[256];                                                 \
    snprintf(buffer, sizeof(buffer), "OptimizationBarrierPass: " fmt, \
             ##__VA_ARGS__);                                          \
    consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer);        \
  } while (0)

#define LOGD(fmt, ...) LOG_(SPV_MSG_DEBUG, fmt, ##__VA_ARGS__)
#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) LOG_(SPV_MSG_ERROR, fmt, ##__VA_ARGS__)

namespace spvtools {
namespace opt {

namespace {

bool IsShiftOp(spv::Op opcode) {
  return opcode == spv::Op::OpShiftLeftLogical ||
         opcode == spv::Op::OpShiftRightLogical ||
         opcode == spv::Op::OpShiftRightArithmetic;
}

}  // namespace

Pass::Status MaliOptimizationBarrierPass::Process() {
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    return Status::SuccessWithoutChange;
  }

  bool modified = false;
  uint32_t scalar_modifications = 0;
  uint32_t vector_modifications = 0;

  // Iterate over all functions in the module.
  for (auto& func : *get_module()) {
    if (func.IsDeclaration()) {
      continue;
    }

    for (auto& bb : func) {
      for (auto inst_itr = bb.begin(); inst_itr != bb.end(); ++inst_itr) {
        Instruction* inst = &*inst_itr;
        // Look for a shift by a constant amount, which may trigger the Mali
        // constant-folding bug.
        if (!IsShiftOp(inst->opcode())) continue;
        if (inst->NumInOperands() < 2) continue;

        const uint32_t shift_amount_id = inst->GetSingleWordInOperand(1);
        const analysis::Constant* shift_constant =
            context()->get_constant_mgr()->FindDeclaredConstant(shift_amount_id);
        if (shift_constant == nullptr) continue;

        const analysis::Type* shift_type = shift_constant->type();
        if (shift_type == nullptr) continue;
        const bool is_scalar = shift_type->AsInteger() != nullptr;
        const bool is_vector =
            shift_type->AsVector() != nullptr &&
            shift_type->AsVector()->element_type()->AsInteger() != nullptr;
        if (!is_scalar && !is_vector) continue;

        const uint32_t original_result_type_id = inst->type_id();
        const uint32_t original_result_id = inst->result_id();

        // The zero must have the same type (and therefore the same integer
        // width and signedness) as the shift result, otherwise the
        // OpBitFieldInsert barrier is invalid for 64-bit and vector shifts.
        const uint32_t const_0_id =
            is_scalar ? GetOrCreateConstantZero(original_result_type_id)
                      : GetOrCreateConstantZeroVector(original_result_type_id);
        if (const_0_id == 0) {
          LOGE("Failed to get or create a zero constant for %s",
               inst->PrettyPrint().c_str());
          return Status::Failure;
        }

        // uint 0's for the OpBitFieldInsert offset and count operands.
        const uint32_t const_0_uint_id =
            context()->get_constant_mgr()->GetUIntConstId(0);
        if (const_0_uint_id == 0) {
          LOGE("Failed to get or create %%uint_0");
          return Status::Failure;
        }

        // Allocate a temp result id for the shift so that the barrier can
        // reuse the original result id without creating a self-reference.
        const uint32_t temp_result_id = context()->TakeNextId();
        if (temp_result_id == 0) {
          LOGE("Failed to allocate new temp result_id for %s",
               inst->PrettyPrint().c_str());
          return Status::Failure;
        }
        inst->SetResultId(temp_result_id);

        // Create an OpBitFieldInsert no-op instruction to block constant
        // folding optimizations that may be broken on some drivers:
        // %original_result_id = OpBitFieldInsert %type %temp_result_id 0 0 0
        InstructionBuilder builder(context(), inst->NextNode());
        auto noop_barrier_inst = std::make_unique<Instruction>(
            context(), spv::Op::OpBitFieldInsert, original_result_type_id,
            original_result_id,
            Instruction::OperandList{
                {SPV_OPERAND_TYPE_ID, {temp_result_id}},     // base
                {SPV_OPERAND_TYPE_ID, {const_0_id}},         // insert
                {SPV_OPERAND_TYPE_ID, {const_0_uint_id}},    // offset
                {SPV_OPERAND_TYPE_ID, {const_0_uint_id}}});  // count
        Instruction* barrier =
            builder.AddInstruction(std::move(noop_barrier_inst));
        if (barrier == nullptr) {
          LOGE("Failed to insert OpBitFieldInsert barrier for %s",
               inst->PrettyPrint().c_str());
          return Status::Failure;
        }

        // Keep the def-use chains consistent after renaming the shift result.
        get_def_use_mgr()->UpdateDefUse(inst);
        get_def_use_mgr()->UpdateDefUse(barrier);

        modified = true;
        if (is_scalar) {
          scalar_modifications += 1;
        } else {
          vector_modifications += 1;
        }
      }
    }
  }

  if (modified) {
    LOG("Added optimization barriers to all functions in the module, "
        "scalar barriers = %d, vector barriers = %d",
        scalar_modifications, vector_modifications);
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

uint32_t MaliOptimizationBarrierPass::GetOrCreateConstantZero(
    const uint32_t type_id) const {
  analysis::Type* type = context()->get_type_mgr()->GetType(type_id);
  if (type == nullptr || type->AsInteger() == nullptr) {
    return 0;
  }
  return context()->get_constant_mgr()->GetNullConstId(type);
}

uint32_t MaliOptimizationBarrierPass::GetOrCreateConstantZeroVector(
    const uint32_t type_id) const {
  analysis::Type* type = context()->get_type_mgr()->GetType(type_id);
  if (type == nullptr) {
    return 0;
  }

  const auto* vec_type = type->AsVector();
  if (vec_type == nullptr ||
      vec_type->element_type()->AsInteger() == nullptr) {
    return 0;
  }

  return context()->get_constant_mgr()->GetNullConstId(type);
}

}  // namespace opt
}  // namespace spvtools
