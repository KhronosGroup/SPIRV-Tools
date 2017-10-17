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

#include "fold.h"
#include "def_use_manager.h"

#include <cassert>
#include <vector>

namespace spvtools {
namespace opt {

namespace {

// Returns the single-word result from performing the given unary operation on
// the operand value which is passed in as a 32-bit word.
uint32_t UnaryOperate(SpvOp opcode, uint32_t operand) {
  switch (opcode) {
    // Arthimetics
    case SpvOp::SpvOpSNegate:
      return -static_cast<int32_t>(operand);
    case SpvOp::SpvOpNot:
      return ~operand;
    case SpvOp::SpvOpLogicalNot:
      return !static_cast<bool>(operand);
    default:
      assert(false &&
             "Unsupported unary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

// Returns the single-word result from performing the given binary operation on
// the operand values which are passed in as two 32-bit word.
uint32_t BinaryOperate(SpvOp opcode, uint32_t a, uint32_t b) {
  switch (opcode) {
    // Arthimetics
    case SpvOp::SpvOpIAdd:
      return a + b;
    case SpvOp::SpvOpISub:
      return a - b;
    case SpvOp::SpvOpIMul:
      return a * b;
    case SpvOp::SpvOpUDiv:
      assert(b != 0);
      return a / b;
    case SpvOp::SpvOpSDiv:
      assert(b != 0u);
      return (static_cast<int32_t>(a)) / (static_cast<int32_t>(b));
    case SpvOp::SpvOpSRem: {
      // The sign of non-zero result comes from the first operand: a. This is
      // guaranteed by C++11 rules for integer division operator. The division
      // result is rounded toward zero, so the result of '%' has the sign of
      // the first operand.
      assert(b != 0u);
      return static_cast<int32_t>(a) % static_cast<int32_t>(b);
    }
    case SpvOp::SpvOpSMod: {
      // The sign of non-zero result comes from the second operand: b
      assert(b != 0u);
      int32_t rem = BinaryOperate(SpvOp::SpvOpSRem, a, b);
      int32_t b_prim = static_cast<int32_t>(b);
      return (rem + b_prim) % b_prim;
    }
    case SpvOp::SpvOpUMod:
      assert(b != 0u);
      return (a % b);

    // Shifting
    case SpvOp::SpvOpShiftRightLogical: {
      return a >> b;
    }
    case SpvOp::SpvOpShiftRightArithmetic:
      return (static_cast<int32_t>(a)) >> b;
    case SpvOp::SpvOpShiftLeftLogical:
      return a << b;

    // Bitwise operations
    case SpvOp::SpvOpBitwiseOr:
      return a | b;
    case SpvOp::SpvOpBitwiseAnd:
      return a & b;
    case SpvOp::SpvOpBitwiseXor:
      return a ^ b;

    // Logical
    case SpvOp::SpvOpLogicalEqual:
      return (static_cast<bool>(a)) == (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalNotEqual:
      return (static_cast<bool>(a)) != (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalOr:
      return (static_cast<bool>(a)) || (static_cast<bool>(b));
    case SpvOp::SpvOpLogicalAnd:
      return (static_cast<bool>(a)) && (static_cast<bool>(b));

    // Comparison
    case SpvOp::SpvOpIEqual:
      return a == b;
    case SpvOp::SpvOpINotEqual:
      return a != b;
    case SpvOp::SpvOpULessThan:
      return a < b;
    case SpvOp::SpvOpSLessThan:
      return (static_cast<int32_t>(a)) < (static_cast<int32_t>(b));
    case SpvOp::SpvOpUGreaterThan:
      return a > b;
    case SpvOp::SpvOpSGreaterThan:
      return (static_cast<int32_t>(a)) > (static_cast<int32_t>(b));
    case SpvOp::SpvOpULessThanEqual:
      return a <= b;
    case SpvOp::SpvOpSLessThanEqual:
      return (static_cast<int32_t>(a)) <= (static_cast<int32_t>(b));
    case SpvOp::SpvOpUGreaterThanEqual:
      return a >= b;
    case SpvOp::SpvOpSGreaterThanEqual:
      return (static_cast<int32_t>(a)) >= (static_cast<int32_t>(b));
    default:
      assert(false &&
             "Unsupported binary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

// Returns the single-word result from performing the given ternary operation
// on the operand values which are passed in as three 32-bit word.
uint32_t TernaryOperate(SpvOp opcode, uint32_t a, uint32_t b, uint32_t c) {
  switch (opcode) {
    case SpvOp::SpvOpSelect:
      return (static_cast<bool>(a)) ? b : c;
    default:
      assert(false &&
             "Unsupported ternary operation for OpSpecConstantOp instruction");
      return 0u;
  }
}

// Returns the single-word result from performing the given operation on the
// operand words. This only works with 32-bit operations and uses boolean
// convention that 0u is false, and anything else is boolean true.
// TODO(qining): Support operands other than 32-bit wide.
uint32_t OperateWords(SpvOp opcode,
                      const std::vector<uint32_t>& operand_words) {
  switch (operand_words.size()) {
    case 1:
      return UnaryOperate(opcode, operand_words.front());
    case 2:
      return BinaryOperate(opcode, operand_words.front(), operand_words.back());
    case 3:
      return TernaryOperate(opcode, operand_words[0], operand_words[1],
                            operand_words[2]);
    default:
      assert(false && "Invalid number of operands");
      return 0;
  }
}

}  // namespace

// Returns the result of performing an operation on scalar constant operands.
// This function extracts the operand values as 32 bit words and returns the
// result in 32 bit word. Scalar constants with longer than 32-bit width are
// not accepted in this function.
uint32_t FoldScalars(SpvOp opcode,
                     const std::vector<analysis::Constant*>& operands) {
  std::vector<uint32_t> operand_values_in_raw_words;
  for (analysis::Constant* operand : operands) {
    if (analysis::ScalarConstant* scalar = operand->AsScalarConstant()) {
      const auto& scalar_words = scalar->words();
      assert(scalar_words.size() == 1 &&
             "Scalar constants with longer than 32-bit width are not allowed "
             "in FoldScalars()");
      operand_values_in_raw_words.push_back(scalar_words.front());
    } else if (operand->AsNullConstant()) {
      operand_values_in_raw_words.push_back(0u);
    } else {
      assert(false &&
             "FoldScalars() only accepts ScalarConst or NullConst type of "
             "constant");
    }
  }
  return OperateWords(opcode, operand_values_in_raw_words);
}

// Returns the result of performing an operation over constant vectors. This
// function iterates through the given vector type constant operands and
// calculates the result for each element of the result vector to return.
// Vectors with longer than 32-bit scalar components are not accepted in this
// function.
std::vector<uint32_t> FoldVectors(
    SpvOp opcode, uint32_t num_dims,
    const std::vector<analysis::Constant*>& operands) {
  std::vector<uint32_t> result;
  for (uint32_t d = 0; d < num_dims; d++) {
    std::vector<uint32_t> operand_values_for_one_dimension;
    for (analysis::Constant* operand : operands) {
      if (analysis::VectorConstant* vector_operand =
              operand->AsVectorConstant()) {
        // Extract the raw value of the scalar component constants
        // in 32-bit words here. The reason of not using FoldScalars() here
        // is that we do not create temporary null constants as components
        // when the vector operand is a NullConstant because Constant creation
        // may need extra checks for the validity and that is not manageed in
        // here.
        if (const analysis::ScalarConstant* scalar_component =
                vector_operand->GetComponents().at(d)->AsScalarConstant()) {
          const auto& scalar_words = scalar_component->words();
          assert(
              scalar_words.size() == 1 &&
              "Vector components with longer than 32-bit width are not allowed "
              "in FoldVectors()");
          operand_values_for_one_dimension.push_back(scalar_words.front());
        } else if (operand->AsNullConstant()) {
          operand_values_for_one_dimension.push_back(0u);
        } else {
          assert(false &&
                 "VectorConst should only has ScalarConst or NullConst as "
                 "components");
        }
      } else if (operand->AsNullConstant()) {
        operand_values_for_one_dimension.push_back(0u);
      } else {
        assert(false &&
               "FoldVectors() only accepts VectorConst or NullConst type of "
               "constant");
      }
    }
    result.push_back(OperateWords(opcode, operand_values_for_one_dimension));
  }
  return result;
}

}  // namespace opt
}  // namespace spvtools
