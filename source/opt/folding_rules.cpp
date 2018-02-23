// Copyright (c) 2018 Google LLC
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

#include "folding_rules.h"
#include "latest_version_glsl_std_450_header.h"

namespace spvtools {
namespace opt {

namespace {
const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;
const uint32_t kExtInstSetIdInIdx = 0;
const uint32_t kExtInstInstructionInIdx = 1;
const uint32_t kFMixXIdInIdx = 2;
const uint32_t kFMixYIdInIdx = 3;

// Returns the element width of |type|.
uint32_t ElementWidth(const analysis::Type* type) {
  if (const analysis::Vector* vec_type = type->AsVector()) {
    return ElementWidth(vec_type->element_type());
  } else if (const analysis::Float* float_type = type->AsFloat()) {
    return float_type->width();
  } else {
    assert(type->AsInteger());
    return type->AsInteger()->width();
  }
}

// Returns true if |type| is Float or a vector of Float.
bool HasFloatingPoint(const analysis::Type* type) {
  if (type->AsFloat()) {
    return true;
  } else if (const analysis::Vector* vec_type = type->AsVector()) {
    return vec_type->element_type()->AsFloat() != nullptr;
  }

  return false;
}

// Returns false if |val| is NaN, infinite or subnormal.
template <typename T>
bool IsValidResult(T val) {
  int classified = std::fpclassify(val);
  switch (classified) {
    case FP_NAN:
    case FP_INFINITE:
    case FP_SUBNORMAL:
      return false;
    default:
      return true;
  }
}

// Returns the negation of |c|. |c| must be a 32 or 64 bit floating point
// constant.
uint32_t NegateFloatingPointConstant(analysis::ConstantManager* const_mgr,
                                     const analysis::Constant* c) {
  assert(c);
  assert(c->type()->AsFloat());
  uint32_t width = c->type()->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    spvutils::FloatProxy<double> result(c->GetDouble() * -1.0);
    words = result.GetWords();
  } else {
    spvutils::FloatProxy<float> result(c->GetFloat() * -1.0f);
    words = result.GetWords();
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

std::vector<uint32_t> ExtractInts(uint64_t val) {
  std::vector<uint32_t> words;
  words.push_back(static_cast<uint32_t>(val));
  words.push_back(static_cast<uint32_t>(val >> 32));
  return words;
}

// Negates the integer constant |c|. Returns the id of the defining instruction.
uint32_t NegateIntegerConstant(analysis::ConstantManager* const_mgr,
                               const analysis::Constant* c) {
  assert(c);
  assert(c->type()->AsInteger());
  uint32_t width = c->type()->AsInteger()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    uint64_t uval = static_cast<uint64_t>(0 - c->GetU64());
    words = ExtractInts(uval);
  } else {
    words.push_back(static_cast<uint32_t>(0 - c->GetU32()));
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

// Negates the vector constant |c|. Returns the id of the defining instruction.
uint32_t NegateVectorConstant(analysis::ConstantManager* const_mgr,
                              const analysis::Constant* c) {
  assert(const_mgr && c);
  assert(c->type()->AsVector());
  if (c->AsNullConstant()) {
    // 0.0 vs -0.0 shouldn't matter.
    return const_mgr->GetDefiningInstruction(c)->result_id();
  } else {
    const analysis::Type* component_type =
        c->AsVectorConstant()->component_type();
    std::vector<uint32_t> words;
    for (auto& comp : c->AsVectorConstant()->GetComponents()) {
      if (component_type->AsFloat()) {
        words.push_back(NegateFloatingPointConstant(const_mgr, comp));
      } else {
        assert(component_type->AsInteger());
        words.push_back(NegateIntegerConstant(const_mgr, comp));
      }
    }

    const analysis::Constant* negated_const =
        const_mgr->GetConstant(c->type(), std::move(words));
    return const_mgr->GetDefiningInstruction(negated_const)->result_id();
  }
}

// Negates |c|. Returns the id of the defining instruction.
uint32_t NegateConstant(analysis::ConstantManager* const_mgr,
                        const analysis::Constant* c) {
  if (c->type()->AsVector()) {
    return NegateVectorConstant(const_mgr, c);
  } else if (c->type()->AsFloat()) {
    return NegateFloatingPointConstant(const_mgr, c);
  } else {
    assert(c->type()->AsInteger());
    return NegateIntegerConstant(const_mgr, c);
  }
}

// Takes the reciprocal of |c|. |c|'s type must be Float or a vector of Float.
// Returns 0 if the reciprocal is NaN, infinite or subnormal.
uint32_t Reciprocal(analysis::ConstantManager* const_mgr,
                    const analysis::Constant* c) {
  assert(const_mgr && c);
  assert(c->type()->AsFloat());

  uint32_t width = c->type()->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
  if (width == 64) {
    spvutils::FloatProxy<double> result(1.0 / c->GetDouble());
    if (!IsValidResult(result.getAsFloat())) return 0;
    words = result.GetWords();
  } else {
    spvutils::FloatProxy<float> result(1.0f / c->GetFloat());
    if (!IsValidResult(result.getAsFloat())) return 0;
    words = result.GetWords();
  }

  const analysis::Constant* negated_const =
      const_mgr->GetConstant(c->type(), std::move(words));
  return const_mgr->GetDefiningInstruction(negated_const)->result_id();
}

// Replaces fdiv where second operand is constant with fmul.
FoldingRule ReciprocalFDiv() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv);
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (!inst->IsFloatingPointFoldingAllowed()) return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    if (constants[1] != nullptr) {
      uint32_t id = 0;
      if (const analysis::VectorConstant* vector_const =
              constants[1]->AsVectorConstant()) {
        std::vector<uint32_t> neg_ids;
        for (auto& comp : vector_const->GetComponents()) {
          id = Reciprocal(const_mgr, comp);
          if (id == 0) return false;
          neg_ids.push_back(id);
        }
        const analysis::Constant* negated_const =
            const_mgr->GetConstant(constants[1]->type(), std::move(neg_ids));
        id = const_mgr->GetDefiningInstruction(negated_const)->result_id();
      } else {
        id = Reciprocal(const_mgr, constants[1]);
        if (id == 0) return false;
      }
      inst->SetOpcode(SpvOpFMul);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0u)}},
           {SPV_OPERAND_TYPE_ID, {id}}});
      return true;
    }

    return false;
  };
};

// Elides consecutive negate instructions.
FoldingRule MergeNegateArithmetic() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    ir::IRContext* context = inst->context();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    ir::Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (op_inst->opcode() == inst->opcode()) {
      // Elide negates.
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {op_inst->GetSingleWordInOperand(0u)}}});
      return true;
    }

    return false;
  };
}

// Merges negate into a mul or div operation if that operation contains a
// constant operand.
FoldingRule MergeNegateMulDivArithmetic() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    ir::Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    SpvOp opcode = op_inst->opcode();
    if (opcode == SpvOpFMul || opcode == SpvOpFDiv || opcode == SpvOpIMul ||
        opcode == SpvOpSDiv || opcode == SpvOpUDiv) {
      std::vector<const analysis::Constant*> op_constants =
          const_mgr->GetOperandConstants(op_inst);
      // Merge negate into mul or div if one operand is constant.
      if (op_constants[0] || op_constants[1]) {
        bool zero_is_variable = op_constants[0] == nullptr;
        const analysis::Constant* c =
            zero_is_variable ? op_constants[1] : op_constants[0];
        uint32_t neg_id = NegateConstant(const_mgr, c);
        uint32_t non_const_id = zero_is_variable
                                    ? op_inst->GetSingleWordInOperand(0u)
                                    : op_inst->GetSingleWordInOperand(1u);
        // Change this instruction to a mul/div.
        inst->SetOpcode(op_inst->opcode());
        if (opcode == SpvOpFDiv || opcode == SpvOpUDiv || opcode == SpvOpSDiv) {
          uint32_t op0 = zero_is_variable ? non_const_id : neg_id;
          uint32_t op1 = zero_is_variable ? neg_id : non_const_id;
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {op0}}, {SPV_OPERAND_TYPE_ID, {op1}}});
        } else {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                               {SPV_OPERAND_TYPE_ID, {neg_id}}});
        }
        return true;
      }
    }

    return false;
  };
}

// Merges negate into a add or sub operation if that operation contains a
// constant operand.
FoldingRule MergeNegateAddSubArithmetic() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFNegate || inst->opcode() == SpvOpSNegate);
    (void)constants;
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    ir::Instruction* op_inst =
        context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    if (HasFloatingPoint(type) && !op_inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    if (op_inst->opcode() == SpvOpFAdd || op_inst->opcode() == SpvOpFSub ||
        op_inst->opcode() == SpvOpIAdd || op_inst->opcode() == SpvOpISub) {
      std::vector<const analysis::Constant*> op_constants =
          const_mgr->GetOperandConstants(op_inst);
      if (op_constants[0] || op_constants[1]) {
        bool zero_is_variable = op_constants[0] == nullptr;
        bool is_add = (op_inst->opcode() == SpvOpFAdd) ||
                      (op_inst->opcode() == SpvOpIAdd);
        bool swap_operands = !(!zero_is_variable && is_add);
        bool negate_const = is_add;
        const analysis::Constant* c =
            zero_is_variable ? op_constants[1] : op_constants[0];
        uint32_t const_id = 0;
        if (negate_const) {
          const_id = NegateConstant(const_mgr, c);
        } else {
          const_id = zero_is_variable ? op_inst->GetSingleWordInOperand(1u)
                                      : op_inst->GetSingleWordInOperand(0u);
        }

        // Swap operands if necessary and make the instruction a subtraction.
        uint32_t op0 =
            zero_is_variable ? op_inst->GetSingleWordInOperand(0u) : const_id;
        uint32_t op1 =
            zero_is_variable ? const_id : op_inst->GetSingleWordInOperand(1u);
        if (swap_operands) std::swap(op0, op1);
        inst->SetOpcode(HasFloatingPoint(type) ? SpvOpFSub : SpvOpISub);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID, {op0}}, {SPV_OPERAND_TYPE_ID, {op1}}});
        return true;
      }
    }

    return false;
  };
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Float.
uint32_t PerformFloatingPointOperation(analysis::ConstantManager* const_mgr,
                                       SpvOp opcode,
                                       const analysis::Constant* input1,
                                       const analysis::Constant* input2) {
  const analysis::Type* type = input1->type();
  assert(type->AsFloat());
  uint32_t width = type->AsFloat()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
#define FOLD_OP(op)                                 \
  if (width == 64) {                                \
    spvutils::FloatProxy<double> val =              \
        input1->GetDouble() op input2->GetDouble(); \
    double dval = val.getAsFloat();                 \
    if (!IsValidResult(dval)) return 0;             \
    words = val.GetWords();                         \
  } else {                                          \
    spvutils::FloatProxy<float> val =               \
        input1->GetFloat() op input2->GetFloat();   \
    float fval = val.getAsFloat();                  \
    if (!IsValidResult(fval)) return 0;             \
    words = val.GetWords();                         \
  }
  switch (opcode) {
    case SpvOpFMul:
      FOLD_OP(*);
      break;
    case SpvOpFDiv:
      FOLD_OP(/);
      break;
    case SpvOpFAdd:
      FOLD_OP(+);
      break;
    case SpvOpFSub:
      FOLD_OP(-);
      break;
    default:
      assert(false && "Unexpected operation");
      break;
  }
#undef FOLD_OP
  const analysis::Constant* merged_const = const_mgr->GetConstant(type, words);
  return const_mgr->GetDefiningInstruction(merged_const)->result_id();
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Integers.
uint32_t PerformIntegerOperation(analysis::ConstantManager* const_mgr,
                                 SpvOp opcode, const analysis::Constant* input1,
                                 const analysis::Constant* input2) {
  assert(input1->type()->AsInteger());
  const analysis::Integer* type = input1->type()->AsInteger();
  uint32_t width = type->AsInteger()->width();
  assert(width == 32 || width == 64);
  std::vector<uint32_t> words;
#define FOLD_OP(op)                                        \
  if (width == 64) {                                       \
    if (type->IsSigned()) {                                \
      int64_t val = input1->GetS64() op input2->GetS64();  \
      words = ExtractInts(static_cast<uint64_t>(val));     \
    } else {                                               \
      uint64_t val = input1->GetU64() op input2->GetU64(); \
      words = ExtractInts(val);                            \
    }                                                      \
  } else {                                                 \
    if (type->IsSigned()) {                                \
      int32_t val = input1->GetS32() op input2->GetS32();  \
      words.push_back(static_cast<uint32_t>(val));         \
    } else {                                               \
      uint32_t val = input1->GetU32() op input2->GetU32(); \
      words.push_back(val);                                \
    }                                                      \
  }
  switch (opcode) {
    case SpvOpIMul:
      FOLD_OP(*);
      break;
    case SpvOpSDiv:
    case SpvOpUDiv:
      // To avoid losing precision we won't perform division that would result
      // in a remainder. Unfortunate code duplication results.
      if (input2->AsIntConstant()->IsZero()) return 0;
      if (width == 64) {
        if (type->IsSigned()) {
          if (input1->GetS64() % input2->GetS64() != 0) return 0;
          int64_t val = input1->GetS64() / input2->GetS64();
          words = ExtractInts(static_cast<uint64_t>(val));
        } else {
          if (input1->GetU64() % input2->GetU64() != 0) return 0;
          uint64_t val = input1->GetU64() / input2->GetU64();
          words = ExtractInts(val);
        }
      } else {
        if (type->IsSigned()) {
          if (input1->GetS32() % input2->GetS32() != 0) return 0;
          int32_t val = input1->GetS32() / input2->GetS32();
          words.push_back(static_cast<uint32_t>(val));
        } else {
          if (input1->GetU32() % input2->GetU32() != 0) return 0;
          uint32_t val = input1->GetU32() / input2->GetU32();
          words.push_back(val);
        }
      }
      break;
    case SpvOpIAdd:
      FOLD_OP(+);
      break;
    case SpvOpISub:
      FOLD_OP(-);
      break;
    default:
      assert(false && "Unexpected operation");
      break;
  }
#undef FOLD_OP
  const analysis::Constant* merged_const = const_mgr->GetConstant(type, words);
  return const_mgr->GetDefiningInstruction(merged_const)->result_id();
}

// Performs |input1| |opcode| |input2| and returns the merged constant result
// id. Returns 0 if the result is not a valid value. The input types must be
// Integers, Floats or Vectors of such.
uint32_t PerformOperation(analysis::ConstantManager* const_mgr, SpvOp opcode,
                          const analysis::Constant* input1,
                          const analysis::Constant* input2) {
  assert(input1->type() == input2->type());
  const analysis::Type* type = input1->type();
  std::vector<uint32_t> words;
  if (const analysis::Vector* vector_type = type->AsVector()) {
    const analysis::Type* ele_type = vector_type->element_type();
    for (uint32_t i = 0; i != vector_type->element_count(); ++i) {
      uint32_t id = 0;
      const analysis::Constant* input1_comp =
          input1->AsVectorConstant()->GetComponents()[i];
      const analysis::Constant* input2_comp =
          input2->AsVectorConstant()->GetComponents()[i];
      if (ele_type->AsFloat()) {
        id = PerformFloatingPointOperation(const_mgr, opcode, input1_comp,
                                           input2_comp);
      } else {
        assert(ele_type->AsInteger());
        id = PerformIntegerOperation(const_mgr, opcode, input1_comp,
                                     input2_comp);
      }
      if (id == 0) return 0;
      words.push_back(id);
    }
    const analysis::Constant* merged_const =
        const_mgr->GetConstant(type, words);
    return const_mgr->GetDefiningInstruction(merged_const)->result_id();
  } else if (type->AsFloat()) {
    return PerformFloatingPointOperation(const_mgr, opcode, input1, input2);
  } else {
    assert(type->AsInteger());
    return PerformIntegerOperation(const_mgr, opcode, input1, input2);
  }
}

// Merges consecutive multiplies where each contains one constant operand.
FoldingRule MergeMulMulArithmetic() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul || inst->opcode() == SpvOpIMul);
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    // Determine the constant input and the variable input in |inst|.
    const analysis::Constant* const_input1 = nullptr;
    ir::Instruction* other_inst = nullptr;
    if (constants[0]) {
      const_input1 = constants[0];
      other_inst =
          context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1u));
    } else {
      const_input1 = constants[1];
      other_inst =
          context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    }

    if (!const_input1) return false;
    if (HasFloatingPoint(type) && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == inst->opcode()) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      if (other_constants[0] || other_constants[1]) {
        bool other_first_is_variable = other_constants[0] == nullptr;
        const analysis::Constant* const_input2 =
            other_first_is_variable ? other_constants[1] : other_constants[0];
        uint32_t merged_id = PerformOperation(const_mgr, inst->opcode(),
                                              const_input1, const_input2);
        if (merged_id == 0) return false;

        uint32_t non_const_id = other_first_is_variable
                                    ? other_inst->GetSingleWordInOperand(0u)
                                    : other_inst->GetSingleWordInOperand(1u);
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                             {SPV_OPERAND_TYPE_ID, {merged_id}}});
        return true;
      }
    }

    return false;
  };
}

// Merges divides into subsequent multiplies if each instruction contains one
// constant operand.
FoldingRule MergeMulDivArithmetic() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul || inst->opcode() == SpvOpIMul);
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    const analysis::Type* type =
        context->get_type_mgr()->GetType(inst->type_id());
    if (HasFloatingPoint(type) && !inst->IsFloatingPointFoldingAllowed())
      return false;

    uint32_t width = ElementWidth(type);
    if (width != 32 && width != 64) return false;

    const analysis::Constant* const_input1 = nullptr;
    ir::Instruction* other_inst = nullptr;
    if (constants[0]) {
      const_input1 = constants[0];
      other_inst =
          context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1u));
    } else {
      const_input1 = constants[1];
      other_inst =
          context->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0u));
    }

    if (!const_input1) return false;
    if (HasFloatingPoint(type) && !other_inst->IsFloatingPointFoldingAllowed())
      return false;

    if (other_inst->opcode() == SpvOpFDiv ||
        other_inst->opcode() == SpvOpSDiv ||
        other_inst->opcode() == SpvOpUDiv) {
      std::vector<const analysis::Constant*> other_constants =
          const_mgr->GetOperandConstants(other_inst);
      if (other_constants[0] || other_constants[1]) {
        bool other_first_is_variable = other_constants[0] == nullptr;
        const analysis::Constant* const_input2 =
            other_first_is_variable ? other_constants[1] : other_constants[0];
        // If the variable value is the second operand of the divide, multiply
        // the constants together. Otherwise divide the constants.
        uint32_t merged_id = PerformOperation(
            const_mgr,
            other_first_is_variable ? other_inst->opcode() : inst->opcode(),
            const_input1, const_input2);
        if (merged_id == 0) return false;

        uint32_t non_const_id = other_first_is_variable
                                    ? other_inst->GetSingleWordInOperand(0u)
                                    : other_inst->GetSingleWordInOperand(1u);

        // If the variable value is on the second operand of the div, then this
        // operation is a div. Otherwise it should be a multiply.
        inst->SetOpcode(other_first_is_variable ? inst->opcode()
                                                : other_inst->opcode());
        if (other_first_is_variable) {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {non_const_id}},
                               {SPV_OPERAND_TYPE_ID, {merged_id}}});
        } else {
          inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {merged_id}},
                               {SPV_OPERAND_TYPE_ID, {non_const_id}}});
        }
        return true;
      }
    }

    return false;
  };
}

FoldingRule IntMultipleBy1() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpIMul && "Wrong opcode.  Should be OpIMul.");
    for (uint32_t i = 0; i < 2; i++) {
      if (constants[i] == nullptr) {
        continue;
      }
      const analysis::IntConstant* int_constant = constants[i]->AsIntConstant();
      if (int_constant) {
        uint32_t width = ElementWidth(int_constant->type());
        if (width != 32 && width != 64) return false;
        bool is_one = (width == 32) ? int_constant->GetU32BitValue() == 1u
                                    : int_constant->GetU64BitValue() == 1ull;
        if (is_one) {
          inst->SetOpcode(SpvOpCopyObject);
          inst->SetInOperands(
              {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1 - i)}}});
          return true;
        }
      }
    }
    return false;
  };
}

FoldingRule CompositeConstructFeedingExtract() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    // If the input to an OpCompositeExtract is an OpCompositeConstruct,
    // then we can simply use the appropriate element in the construction.
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    analysis::TypeManager* type_mgr = inst->context()->get_type_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    ir::Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeConstruct) {
      return false;
    }

    std::vector<ir::Operand> operands;
    analysis::Type* composite_type = type_mgr->GetType(cinst->type_id());
    if (composite_type->AsVector() == nullptr) {
      // Get the element being extracted from the OpCompositeConstruct
      // Since it is not a vector, it is simple to extract the single element.
      uint32_t element_index = inst->GetSingleWordInOperand(1);
      uint32_t element_id = cinst->GetSingleWordInOperand(element_index);
      operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});

      // Add the remaining indices for extraction.
      for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {inst->GetSingleWordInOperand(i)}});
      }

    } else {
      // With vectors we have to handle the case where it is concatenating
      // vectors.
      assert(inst->NumInOperands() == 2 &&
             "Expecting a vector of scalar values.");

      uint32_t element_index = inst->GetSingleWordInOperand(1);
      for (uint32_t construct_index = 0;
           construct_index < cinst->NumInOperands(); ++construct_index) {
        uint32_t element_id = cinst->GetSingleWordInOperand(construct_index);
        ir::Instruction* element_def = def_use_mgr->GetDef(element_id);
        analysis::Vector* element_type =
            type_mgr->GetType(element_def->type_id())->AsVector();
        if (element_type) {
          uint32_t vector_size = element_type->element_count();
          if (vector_size < element_index) {
            // The element we want comes after this vector.
            element_index -= vector_size;
          } else {
            // We want an element of this vector.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            operands.push_back(
                {SPV_OPERAND_TYPE_LITERAL_INTEGER, {element_index}});
            break;
          }
        } else {
          if (element_index == 0) {
            // This is a scalar, and we this is the element we are extracting.
            operands.push_back({SPV_OPERAND_TYPE_ID, {element_id}});
            break;
          } else {
            // Skip over this scalar value.
            --element_index;
          }
        }
      }
    }

    // If there were no extra indices, then we have the final object.  No need
    // to extract even more.
    if (operands.size() == 1) {
      inst->SetOpcode(SpvOpCopyObject);
    }

    inst->SetInOperands(std::move(operands));
    return true;
  };
}

FoldingRule CompositeExtractFeedingConstruct() {
  // If the OpCompositeConstruct is simply putting back together elements that
  // where extracted from the same souce, we can simlpy reuse the source.
  //
  // This is a common code pattern because of the way that scalar replacement
  // works.
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeConstruct &&
           "Wrong opcode.  Should be OpCompositeConstruct.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    uint32_t original_id = 0;

    // Check each element to make sure they are:
    // - extractions
    // - extracting the same position they are inserting
    // - all extract from the same id.
    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      uint32_t element_id = inst->GetSingleWordInOperand(i);
      ir::Instruction* element_inst = def_use_mgr->GetDef(element_id);

      if (element_inst->opcode() != SpvOpCompositeExtract) {
        return false;
      }

      if (element_inst->NumInOperands() != 2) {
        return false;
      }

      if (element_inst->GetSingleWordInOperand(1) != i) {
        return false;
      }

      if (i == 0) {
        original_id =
            element_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
      } else if (original_id != element_inst->GetSingleWordInOperand(
                                    kExtractCompositeIdInIdx)) {
        return false;
      }
    }

    // The last check it to see that the object being extracted from is the
    // correct type.
    ir::Instruction* original_inst = def_use_mgr->GetDef(original_id);
    if (original_inst->type_id() != inst->type_id()) {
      return false;
    }

    // Simplify by using the original object.
    inst->SetOpcode(SpvOpCopyObject);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {original_id}}});
    return true;
  };
}

FoldingRule InsertFeedingExtract() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    assert(inst->opcode() == SpvOpCompositeExtract &&
           "Wrong opcode.  Should be OpCompositeExtract.");
    analysis::DefUseManager* def_use_mgr = inst->context()->get_def_use_mgr();
    uint32_t cid = inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
    ir::Instruction* cinst = def_use_mgr->GetDef(cid);

    if (cinst->opcode() != SpvOpCompositeInsert) {
      return false;
    }

    // Find the first position where the list of insert and extract indicies
    // differ, if at all.
    uint32_t i;
    for (i = 1; i < inst->NumInOperands(); ++i) {
      if (i + 1 >= cinst->NumInOperands()) {
        break;
      }

      if (inst->GetSingleWordInOperand(i) !=
          cinst->GetSingleWordInOperand(i + 1)) {
        break;
      }
    }

    // We are extracting the element that was inserted.
    if (i == inst->NumInOperands() && i + 1 == cinst->NumInOperands()) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID,
            {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}}});
      return true;
    }

    // Extracting the value that was inserted along with values for the base
    // composite.  Cannot do anything.
    if (i == inst->NumInOperands()) {
      return false;
    }

    // Extracting an element of the value that was inserted.  Extract from
    // that value directly.
    if (i + 1 == cinst->NumInOperands()) {
      std::vector<ir::Operand> operands;
      operands.push_back(
          {SPV_OPERAND_TYPE_ID,
           {cinst->GetSingleWordInOperand(kInsertObjectIdInIdx)}});
      for (; i < inst->NumInOperands(); ++i) {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {cinst->GetSingleWordInOperand(i)}});
      }
      inst->SetInOperands(std::move(operands));
      return true;
    }

    // Extracting a value that is disjoint from the element being inserted.
    // Rewrite the extract to use the composite input to the insert.
    std::vector<ir::Operand> operands;
    operands.push_back(
        {SPV_OPERAND_TYPE_ID,
         {cinst->GetSingleWordInOperand(kInsertCompositeIdInIdx)}});
    for (i = 1; i < inst->NumInOperands(); ++i) {
      operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {inst->GetSingleWordInOperand(i)}});
    }
    inst->SetInOperands(std::move(operands));
    return true;
  };
}

FoldingRule RedundantPhi() {
  // An OpPhi instruction where all values are the same or the result of the phi
  // itself, can be replaced by the value itself.
  return
      [](ir::Instruction* inst, const std::vector<const analysis::Constant*>&) {
        assert(inst->opcode() == SpvOpPhi && "Wrong opcode.  Should be OpPhi.");

        ir::IRContext* context = inst->context();
        analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

        uint32_t incoming_value = 0;

        for (uint32_t i = 0; i < inst->NumInOperands(); i += 2) {
          uint32_t op_id = inst->GetSingleWordInOperand(i);
          if (op_id == inst->result_id()) {
            continue;
          }

          ir::Instruction* op_inst = def_use_mgr->GetDef(op_id);
          if (op_inst->opcode() == SpvOpUndef) {
            // TODO: We should be able to still use op_id if we know that
            // the definition of op_id dominates |inst|.
            return false;
          }

          if (incoming_value == 0) {
            incoming_value = op_id;
          } else if (op_id != incoming_value) {
            // Found two possible value.  Can't simplify.
            return false;
          }
        }

        if (incoming_value == 0) {
          // Code looks invalid.  Don't do anything.
          return false;
        }

        // We have a single incoming value.  Simplify using that value.
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {incoming_value}}});
        return true;
      };
}

FoldingRule RedundantSelect() {
  // An OpSelect instruction where both values are the same or the condition is
  // constant can be replaced by one of the values
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpSelect &&
           "Wrong opcode.  Should be OpSelect.");
    assert(inst->NumInOperands() == 3);
    assert(constants.size() == 3);

    const analysis::BoolConstant* bc =
        constants[0] ? constants[0]->AsBoolConstant() : nullptr;
    uint32_t true_id = inst->GetSingleWordInOperand(1);
    uint32_t false_id = inst->GetSingleWordInOperand(2);

    if (bc) {
      // Select condition is constant, result is known
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {bc->value() ? true_id : false_id}}});
      return true;
    } else if (true_id == false_id) {
      // Both results are the same, condition doesn't matter
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {true_id}}});
      return true;
    } else {
      return false;
    }
  };
}

enum class FloatConstantKind { Unknown, Zero, One };

FloatConstantKind getFloatConstantKind(const analysis::Constant* constant) {
  if (constant == nullptr) {
    return FloatConstantKind::Unknown;
  }

  if (const analysis::VectorConstant* vc = constant->AsVectorConstant()) {
    const std::vector<const analysis::Constant*>& components =
        vc->GetComponents();
    assert(!components.empty());

    FloatConstantKind kind = getFloatConstantKind(components[0]);

    for (size_t i = 1; i < components.size(); ++i) {
      if (getFloatConstantKind(components[i]) != kind) {
        return FloatConstantKind::Unknown;
      }
    }

    return kind;
  } else if (const analysis::FloatConstant* fc = constant->AsFloatConstant()) {
    double value = (fc->type()->AsFloat()->width() == 64) ? fc->GetDoubleValue()
                                                          : fc->GetFloatValue();

    if (value == 0.0) {
      return FloatConstantKind::Zero;
    } else if (value == 1.0) {
      return FloatConstantKind::One;
    } else {
      return FloatConstantKind::Unknown;
    }
  } else {
    return FloatConstantKind::Unknown;
  }
}

FoldingRule RedundantFAdd() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFAdd && "Wrong opcode.  Should be OpFAdd.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::Zero ? 1 : 0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFSub() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFSub && "Wrong opcode.  Should be OpFSub.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpFNegate);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(1)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFMul() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFMul && "Wrong opcode.  Should be OpFMul.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero || kind1 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::Zero ? 0 : 1)}}});
      return true;
    }

    if (kind0 == FloatConstantKind::One || kind1 == FloatConstantKind::One) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands({{SPV_OPERAND_TYPE_ID,
                            {inst->GetSingleWordInOperand(
                                kind0 == FloatConstantKind::One ? 1 : 0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFDiv() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpFDiv && "Wrong opcode.  Should be OpFDiv.");
    assert(constants.size() == 2);

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    FloatConstantKind kind0 = getFloatConstantKind(constants[0]);
    FloatConstantKind kind1 = getFloatConstantKind(constants[1]);

    if (kind0 == FloatConstantKind::Zero) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    if (kind1 == FloatConstantKind::One) {
      inst->SetOpcode(SpvOpCopyObject);
      inst->SetInOperands(
          {{SPV_OPERAND_TYPE_ID, {inst->GetSingleWordInOperand(0)}}});
      return true;
    }

    return false;
  };
}

FoldingRule RedundantFMix() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants) {
    assert(inst->opcode() == SpvOpExtInst &&
           "Wrong opcode.  Should be OpExtInst.");

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return false;
    }

    uint32_t instSetId =
        inst->context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();

    if (inst->GetSingleWordInOperand(kExtInstSetIdInIdx) == instSetId &&
        inst->GetSingleWordInOperand(kExtInstInstructionInIdx) ==
            GLSLstd450FMix) {
      assert(constants.size() == 5);

      FloatConstantKind kind4 = getFloatConstantKind(constants[4]);

      if (kind4 == FloatConstantKind::Zero || kind4 == FloatConstantKind::One) {
        inst->SetOpcode(SpvOpCopyObject);
        inst->SetInOperands(
            {{SPV_OPERAND_TYPE_ID,
              {inst->GetSingleWordInOperand(kind4 == FloatConstantKind::Zero
                                                ? kFMixXIdInIdx
                                                : kFMixYIdInIdx)}}});
        return true;
      }
    }

    return false;
  };
}

}  // namespace

spvtools::opt::FoldingRules::FoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[SpvOpCompositeConstruct].push_back(CompositeExtractFeedingConstruct());

  rules_[SpvOpCompositeExtract].push_back(InsertFeedingExtract());
  rules_[SpvOpCompositeExtract].push_back(CompositeConstructFeedingExtract());

  rules_[SpvOpExtInst].push_back(RedundantFMix());

  rules_[SpvOpFAdd].push_back(RedundantFAdd());

  rules_[SpvOpFDiv].push_back(RedundantFDiv());
  rules_[SpvOpFDiv].push_back(ReciprocalFDiv());

  rules_[SpvOpFMul].push_back(RedundantFMul());
  rules_[SpvOpFMul].push_back(MergeMulMulArithmetic());
  rules_[SpvOpFMul].push_back(MergeMulDivArithmetic());

  rules_[SpvOpFNegate].push_back(MergeNegateArithmetic());
  rules_[SpvOpFNegate].push_back(MergeNegateAddSubArithmetic());
  rules_[SpvOpFNegate].push_back(MergeNegateMulDivArithmetic());

  rules_[SpvOpFSub].push_back(RedundantFSub());

  rules_[SpvOpIMul].push_back(IntMultipleBy1());
  rules_[SpvOpIMul].push_back(MergeMulMulArithmetic());
  rules_[SpvOpIMul].push_back(MergeMulDivArithmetic());

  rules_[SpvOpSNegate].push_back(MergeNegateArithmetic());
  rules_[SpvOpSNegate].push_back(MergeNegateMulDivArithmetic());
  rules_[SpvOpSNegate].push_back(MergeNegateAddSubArithmetic());

  rules_[SpvOpPhi].push_back(RedundantPhi());

  rules_[SpvOpSelect].push_back(RedundantSelect());
}

}  // namespace opt
}  // namespace spvtools
