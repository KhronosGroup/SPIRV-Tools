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

#include "const_folding_rules.h"

namespace spvtools {
namespace opt {

namespace {
const uint32_t kExtractCompositeIdInIdx = 0;

// Returns a vector that contains the two 32-bit integers that result from
// splitting |a| in two.  The first entry in vector are the low order bit if
// |a|.
inline std::vector<uint32_t> ExtractInts(uint64_t a) {
  std::vector<uint32_t> result;
  result.push_back(static_cast<uint32_t>(a));
  result.push_back(static_cast<uint32_t>(a >> 32));
  return result;
}

// Folds an OpcompositeExtract where input is a composite constant.
ConstantFoldingRule FoldExtractWithConstants() {
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    const analysis::Constant* c = constants[kExtractCompositeIdInIdx];
    if (c == nullptr) {
      return nullptr;
    }

    for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
      uint32_t element_index = inst->GetSingleWordInOperand(i);
      if (c->AsNullConstant()) {
        // Return Null for the return type.
        ir::IRContext* context = inst->context();
        analysis::ConstantManager* const_mgr = context->get_constant_mgr();
        analysis::TypeManager* type_mgr = context->get_type_mgr();
        return const_mgr->GetConstant(type_mgr->GetType(inst->type_id()), {});
      }

      auto cc = c->AsCompositeConstant();
      assert(cc != nullptr);
      auto components = cc->GetComponents();
      c = components[element_index];
    }
    return c;
  };
}

ConstantFoldingRule FoldCompositeWithConstants() {
  // Folds an OpCompositeConstruct where all of the inputs are constants to a
  // constant.  A new constant is created if necessary.
  return [](ir::Instruction* inst,
            const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* new_type = type_mgr->GetType(inst->type_id());

    std::vector<uint32_t> ids;
    for (const analysis::Constant* element_const : constants) {
      if (element_const == nullptr) {
        return nullptr;
      }
      uint32_t element_id = const_mgr->FindDeclaredConstant(element_const);
      if (element_id == 0) {
        return nullptr;
      }
      ids.push_back(element_id);
    }
    return const_mgr->GetConstant(new_type, ids);
  };
}

// The interface for a function that returns the result of applying a scalar
// floating-point binary operation on |a| and |b|.  The type of the return value
// will be |type|.  The input constants must also be of type |type|.
using FloatScalarFoldingRule = std::function<const analysis::Constant*(
    const analysis::Type* result_type, const analysis::Constant* a,
    const analysis::Constant* b, analysis::ConstantManager*)>;

// Returns an std::vector containing the elements of |constant|.  The type of
// |constant| must be |Vector|.
std::vector<const analysis::Constant*> GetVectorComponents(
    const analysis::Constant* constant, analysis::ConstantManager* const_mgr) {
  std::vector<const analysis::Constant*> components;
  const analysis::VectorConstant* a = constant->AsVectorConstant();
  const analysis::Vector* vector_type = constant->type()->AsVector();
  assert(vector_type != nullptr);
  if (a != nullptr) {
    for (uint32_t i = 0; i < vector_type->element_count(); ++i) {
      components.push_back(a->GetComponents()[i]);
    }
  } else {
    const analysis::Type* element_type = vector_type->element_type();
    const analysis::Constant* element_null_const =
        const_mgr->GetConstant(element_type, {});
    for (uint32_t i = 0; i < vector_type->element_count(); ++i) {
      components.push_back(element_null_const);
    }
  }
  return components;
}

// Returns a |ConstantFoldingRule| that folds floating point scalars using
// |scalar_rule| and vectors of floating point by applying |scalar_rule| to the
// elements of the vector.  The |ConstantFoldingRule| that is returned assumes
// that |constants| contains 2 entries.  If they are not |nullptr|, then their
// type is either |Float| or a |Vector| whose element type is |Float|.
ConstantFoldingRule FoldFloatingPointOp(FloatScalarFoldingRule scalar_rule) {
  return [scalar_rule](ir::Instruction* inst,
                       const std::vector<const analysis::Constant*>& constants)
             -> const analysis::Constant* {
    ir::IRContext* context = inst->context();
    analysis::ConstantManager* const_mgr = context->get_constant_mgr();
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    const analysis::Type* result_type = type_mgr->GetType(inst->type_id());
    const analysis::Vector* vector_type = result_type->AsVector();

    if (!inst->IsFloatingPointFoldingAllowed()) {
      return nullptr;
    }

    if (constants[0] == nullptr || constants[1] == nullptr) {
      return nullptr;
    }

    if (vector_type != nullptr) {
      std::vector<const analysis::Constant*> a_components;
      std::vector<const analysis::Constant*> b_components;
      std::vector<const analysis::Constant*> results_components;

      a_components = GetVectorComponents(constants[0], const_mgr);
      b_components = GetVectorComponents(constants[1], const_mgr);

      // Fold each component of the vector.
      for (uint32_t i = 0; i < a_components.size(); ++i) {
        results_components.push_back(scalar_rule(vector_type->element_type(),
                                                 a_components[i],
                                                 b_components[i], const_mgr));
        if (results_components[i] == nullptr) {
          return nullptr;
        }
      }

      // Build the constant object and return it.
      std::vector<uint32_t> ids;
      for (const analysis::Constant* member : results_components) {
        ids.push_back(const_mgr->GetDefiningInstruction(member)->result_id());
      }
      return const_mgr->GetConstant(vector_type, ids);
    } else {
      return scalar_rule(result_type, constants[0], constants[1], const_mgr);
    }
  };
}

// Returns the floating point value of |c|.  The constant |c| must have type
// |Float|, and width |32|.
float GetFloatFromConst(const analysis::Constant* c) {
  assert(c->type()->AsFloat() != nullptr &&
         c->type()->AsFloat()->width() == 32);
  const analysis::FloatConstant* fc = c->AsFloatConstant();
  if (fc) {
    return fc->GetFloatValue();
  } else {
    assert(c->AsNullConstant() && "c must be a float point constant.");
    return 0.0f;
  }
}

// Returns the double value of |c|.  The constant |c| must have type
// |Float|, and width |64|.
double GetDoubleFromConst(const analysis::Constant* c) {
  assert(c->type()->AsFloat() != nullptr &&
         c->type()->AsFloat()->width() == 64);
  const analysis::FloatConstant* fc = c->AsFloatConstant();
  if (fc) {
    return fc->GetDoubleValue();
  } else {
    assert(c->AsNullConstant() && "c must be a float point constant.");
    return 0.0;
  }
}

// This macro defines a |FloatScalarFoldingRule| that applies |op|.  The
// operator |op| must work for both float and double, and use syntax "f1 op f2".
#define FOLD_FPARITH_OP(op)                                               \
  [](const analysis::Type* result_type, const analysis::Constant* a,      \
     const analysis::Constant* b,                                         \
     analysis::ConstantManager* const_mgr) -> const analysis::Constant* { \
    assert(result_type != nullptr && a != nullptr && b != nullptr);       \
    assert(result_type == a->type() && result_type == b->type());         \
    const analysis::Float* float_type = result_type->AsFloat();           \
    assert(float_type != nullptr);                                        \
    if (float_type->width() == 32) {                                      \
      float fa = GetFloatFromConst(a);                                    \
      float fb = GetFloatFromConst(b);                                    \
      spvutils::FloatProxy<float> result(fa op fb);                       \
      std::vector<uint32_t> words = {result.data()};                      \
      return const_mgr->GetConstant(result_type, words);                  \
    } else if (float_type->width() == 64) {                               \
      double fa = GetDoubleFromConst(a);                                  \
      double fb = GetDoubleFromConst(b);                                  \
      spvutils::FloatProxy<double> result(fa op fb);                      \
      std::vector<uint32_t> words(ExtractInts(result.data()));            \
      return const_mgr->GetConstant(result_type, words);                  \
    }                                                                     \
    return nullptr;                                                       \
  }

// Define the folding rules for subtraction, addition, multiplication, and
// division for floating point values.
ConstantFoldingRule FoldFSub() {
  return FoldFloatingPointOp(FOLD_FPARITH_OP(-));
}
ConstantFoldingRule FoldFAdd() {
  return FoldFloatingPointOp(FOLD_FPARITH_OP(+));
}
ConstantFoldingRule FoldFMul() {
  return FoldFloatingPointOp(FOLD_FPARITH_OP(*));
}
ConstantFoldingRule FoldFDiv() {
  return FoldFloatingPointOp(FOLD_FPARITH_OP(/));
}

bool CompareFloatingPoint(bool op_result, bool op_unordered,
                          bool need_ordered) {
  if (need_ordered) {
    // operands are ordered and Operand 1 is |op| Operand 2
    return !op_unordered && op_result;
  } else {
    // operands are unordered or Operand 1 is |op| Operand 2
    return op_unordered || op_result;
  }
}

// This macro defines a |FloatScalarFoldingRule| that applies |op|.  The
// operator |op| must work for both float and double, and use syntax "f1 op f2".
#define FOLD_FPCMP_OP(op, ord)                                            \
  [](const analysis::Type* result_type, const analysis::Constant* a,      \
     const analysis::Constant* b,                                         \
     analysis::ConstantManager* const_mgr) -> const analysis::Constant* { \
    assert(result_type != nullptr && a != nullptr && b != nullptr);       \
    assert(result_type->AsBool());                                        \
    assert(a->type() == b->type());                                       \
    const analysis::Float* float_type = a->type()->AsFloat();             \
    assert(float_type != nullptr);                                        \
    if (float_type->width() == 32) {                                      \
      float fa = GetFloatFromConst(a);                                    \
      float fb = GetFloatFromConst(b);                                    \
      bool result = CompareFloatingPoint(                                 \
          fa op fb, std::isnan(fa) || std::isnan(fb), ord);               \
      std::vector<uint32_t> words = {uint32_t(result)};                   \
      return const_mgr->GetConstant(result_type, words);                  \
    } else if (float_type->width() == 64) {                               \
      double fa = GetDoubleFromConst(a);                                  \
      double fb = GetDoubleFromConst(b);                                  \
      bool result = CompareFloatingPoint(                                 \
          fa op fb, std::isnan(fa) || std::isnan(fb), ord);               \
      std::vector<uint32_t> words = {uint32_t(result)};                   \
      return const_mgr->GetConstant(result_type, words);                  \
    }                                                                     \
    return nullptr;                                                       \
  }

// Define the folding rules for ordered and unordered comparison for floating
// point values.
ConstantFoldingRule FoldFOrdEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(==, true));
}
ConstantFoldingRule FoldFUnordEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(==, false));
}
ConstantFoldingRule FoldFOrdNotEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(!=, true));
}
ConstantFoldingRule FoldFUnordNotEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(!=, false));
}
ConstantFoldingRule FoldFOrdLessThan() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(<, true));
}
ConstantFoldingRule FoldFUnordLessThan() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(<, false));
}
ConstantFoldingRule FoldFOrdGreaterThan() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(>, true));
}
ConstantFoldingRule FoldFUnordGreaterThan() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(>, false));
}
ConstantFoldingRule FoldFOrdLessThanEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(<=, true));
}
ConstantFoldingRule FoldFUnordLessThanEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(<=, false));
}
ConstantFoldingRule FoldFOrdGreaterThanEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(>=, true));
}
ConstantFoldingRule FoldFUnordGreaterThanEqual() {
  return FoldFloatingPointOp(FOLD_FPCMP_OP(>=, false));
}
}  // namespace

spvtools::opt::ConstantFoldingRules::ConstantFoldingRules() {
  // Add all folding rules to the list for the opcodes to which they apply.
  // Note that the order in which rules are added to the list matters. If a rule
  // applies to the instruction, the rest of the rules will not be attempted.
  // Take that into consideration.

  rules_[SpvOpCompositeConstruct].push_back(FoldCompositeWithConstants());

  rules_[SpvOpCompositeExtract].push_back(FoldExtractWithConstants());

  rules_[SpvOpFAdd].push_back(FoldFAdd());
  rules_[SpvOpFDiv].push_back(FoldFDiv());
  rules_[SpvOpFMul].push_back(FoldFMul());
  rules_[SpvOpFSub].push_back(FoldFSub());

  rules_[SpvOpFOrdEqual].push_back(FoldFOrdEqual());
  rules_[SpvOpFUnordEqual].push_back(FoldFUnordEqual());
  rules_[SpvOpFOrdNotEqual].push_back(FoldFOrdNotEqual());
  rules_[SpvOpFUnordNotEqual].push_back(FoldFUnordNotEqual());
  rules_[SpvOpFOrdLessThan].push_back(FoldFOrdLessThan());
  rules_[SpvOpFUnordLessThan].push_back(FoldFUnordLessThan());
  rules_[SpvOpFOrdGreaterThan].push_back(FoldFOrdGreaterThan());
  rules_[SpvOpFUnordGreaterThan].push_back(FoldFUnordGreaterThan());
  rules_[SpvOpFOrdLessThanEqual].push_back(FoldFOrdLessThanEqual());
  rules_[SpvOpFUnordLessThanEqual].push_back(FoldFUnordLessThanEqual());
  rules_[SpvOpFOrdGreaterThanEqual].push_back(FoldFOrdGreaterThanEqual());
  rules_[SpvOpFUnordGreaterThanEqual].push_back(FoldFUnordGreaterThanEqual());
}
}  // namespace opt
}  // namespace spvtools
