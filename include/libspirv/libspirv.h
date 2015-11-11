// Copyright (c) 2015 The Khronos Group Inc.
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

#ifndef LIBSPIRV_LIBSPIRV_LIBSPIRV_H_
#define LIBSPIRV_LIBSPIRV_LIBSPIRV_H_

#include <headers/GLSL.std.450.h>
#include <headers/OpenCL.std.h>
#include <headers/spirv.h>
#include <headers/spirv_operands.hpp>

#ifdef __cplusplus
using namespace spv;
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Helpers

#define spvIsInBitfield(value, bitfield) (value == (value & bitfield))

#define SPV_BIT(shift) (1 << (shift))

#define SPV_FORCE_16_BIT_ENUM(name) _##name = 0x7fff
#define SPV_FORCE_32_BIT_ENUM(name) _##name = 0x7fffffff

// Enumerations

typedef enum spv_result_t {
  SPV_SUCCESS = 0,
  SPV_UNSUPPORTED = 1,
  SPV_END_OF_STREAM = 2,
  SPV_WARNING = 3,
  SPV_FAILED_MATCH = 4,
  SPV_ERROR_INTERNAL = -1,
  SPV_ERROR_OUT_OF_MEMORY = -2,
  SPV_ERROR_INVALID_POINTER = -3,
  SPV_ERROR_INVALID_BINARY = -4,
  SPV_ERROR_INVALID_TEXT = -5,
  SPV_ERROR_INVALID_TABLE = -6,
  SPV_ERROR_INVALID_VALUE = -7,
  SPV_ERROR_INVALID_DIAGNOSTIC = -8,
  SPV_ERROR_INVALID_LOOKUP = -9,
  SPV_ERROR_INVALID_ID = -10,
  SPV_FORCE_32_BIT_ENUM(spv_result_t)
} spv_result_t;

typedef enum spv_endianness_t {
  SPV_ENDIANNESS_LITTLE,
  SPV_ENDIANNESS_BIG,
  SPV_FORCE_32_BIT_ENUM(spv_endianness_t)
} spv_endianness_t;

// The kinds of operands that an instruction may have.
//
// Some operand types are "concrete".  The binary parser uses a concrete
// operand type to describe an operand of a parsed instruction.
//
// The assembler uses all operand types.  In addition to determining what
// kind of value an operand may be, non-concrete operand types capture the
// fact that an operand might be optional (may be absent, or present exactly
// once), or might occure zero or more times.
//
// Sometimes we also need to be able to express the fact that an operand
// is a member of an optional tuple of values.  In that case the first member
// would be optional, and the subsequent members would be required.
typedef enum spv_operand_type_t {
  // A sentinel value.
  SPV_OPERAND_TYPE_NONE = 0,

#define FIRST_CONCRETE(ENUM) ENUM, SPV_OPERAND_TYPE_FIRST_CONCRETE_TYPE = ENUM
#define LAST_CONCRETE(ENUM) ENUM, SPV_OPERAND_TYPE_LAST_CONCRETE_TYPE = ENUM

  // Set 1:  Operands that are IDs.
  FIRST_CONCRETE(SPV_OPERAND_TYPE_ID),
  SPV_OPERAND_TYPE_TYPE_ID,
  SPV_OPERAND_TYPE_RESULT_ID,
  SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,  // SPIR-V Sec 3.25
  SPV_OPERAND_TYPE_SCOPE_ID,             // SPIR-V Sec 3.27

  // TODO(dneto): Remove these old names.
  SPV_OPERAND_TYPE_MEMORY_SEMANTICS = SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID,
  SPV_OPERAND_TYPE_EXECUTION_SCOPE = SPV_OPERAND_TYPE_SCOPE_ID,

  // Set 2:  Operands that are literal numbers.
  SPV_OPERAND_TYPE_LITERAL_INTEGER,  // Always unsigned 32-bits.
  // The Instruction argument to OpExtInst. It's an unsigned 32-bit literal
  // number indicating which instruction to use from an extended instruction
  // set.
  SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
  // The Opcode argument to OpSpecConstantOp. It determines the operation
  // to be performed on constant operands to compute a specialization constant
  // result.
  SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER,
  // A literal number whose format and size are determined by a previous operand
  // in the same instruction.  It's a signed integer, an unsigned integer, or a
  // floating point number.  It also has a specified bit width.  The width
  // may be larger than 32, which would require such a typed literal value to
  // occupy multiple SPIR-V words.
  SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,

  // Set 3:  The literal string operand type.
  SPV_OPERAND_TYPE_LITERAL_STRING,

  // Set 4:  Operands that are a single word enumerated value.
  SPV_OPERAND_TYPE_SOURCE_LANGUAGE,               // SPIR-V Sec 3.2
  SPV_OPERAND_TYPE_EXECUTION_MODEL,               // SPIR-V Sec 3.3
  SPV_OPERAND_TYPE_ADDRESSING_MODEL,              // SPIR-V Sec 3.4
  SPV_OPERAND_TYPE_MEMORY_MODEL,                  // SPIR-V Sec 3.5
  SPV_OPERAND_TYPE_EXECUTION_MODE,                // SPIR-V Sec 3.6
  SPV_OPERAND_TYPE_STORAGE_CLASS,                 // SPIR-V Sec 3.7
  SPV_OPERAND_TYPE_DIMENSIONALITY,                // SPIR-V Sec 3.8
  SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE,       // SPIR-V Sec 3.9
  SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE,           // SPIR-V Sec 3.10
  SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT,          // SPIR-V Sec 3.11
  SPV_OPERAND_TYPE_IMAGE_CHANNEL_ORDER,           // SPIR-V Sec 3.12
  SPV_OPERAND_TYPE_IMAGE_CHANNEL_DATA_TYPE,       // SPIR-V Sec 3.13
  SPV_OPERAND_TYPE_FP_ROUNDING_MODE,              // SPIR-V Sec 3.16
  SPV_OPERAND_TYPE_LINKAGE_TYPE,                  // SPIR-V Sec 3.17
  SPV_OPERAND_TYPE_ACCESS_QUALIFIER,              // SPIR-V Sec 3.18
  SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE,  // SPIR-V Sec 3.19
  SPV_OPERAND_TYPE_DECORATION,                    // SPIR-V Sec 3.20
  SPV_OPERAND_TYPE_BUILT_IN,                      // SPIR-V Sec 3.21
  SPV_OPERAND_TYPE_GROUP_OPERATION,               // SPIR-V Sec 3.28
  SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS,              // SPIR-V Sec 3.29
  SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO,         // SPIR-V Sec 3.30
  SPV_OPERAND_TYPE_CAPABILITY,                    // SPIR-V Sec 3.31

  // Set 5:  Operands that are a single word bitmask.
  // Sometimes a set bit indicates the instruction requires still more operands.
  SPV_OPERAND_TYPE_IMAGE,                         // SPIR-V Sec 3.14
  SPV_OPERAND_TYPE_FP_FAST_MATH_MODE,             // SPIR-V Sec 3.15
  SPV_OPERAND_TYPE_SELECTION_CONTROL,             // SPIR-V Sec 3.22
  SPV_OPERAND_TYPE_LOOP_CONTROL,                  // SPIR-V Sec 3.23
  SPV_OPERAND_TYPE_FUNCTION_CONTROL,              // SPIR-V Sec 3.24
  LAST_CONCRETE(SPV_OPERAND_TYPE_MEMORY_ACCESS),  // SPIR-V Sec 3.26
#undef FIRST_CONCRETE
#undef LAST_CONCRETE

// The remaining operand types are only used internally by the assembler.
// There are two categories:
//    Optional : expands to 0 or 1 operand, like ? in regular expressions.
//    Variable : expands to 0, 1 or many operands or pairs of operands.
//               This is similar to * in regular expressions.

// Macros for defining bounds on optional and variable operand types.
// Any variable operand type is also optional.
#define FIRST_OPTIONAL(ENUM) ENUM, SPV_OPERAND_TYPE_FIRST_OPTIONAL_TYPE = ENUM
#define FIRST_VARIABLE(ENUM) ENUM, SPV_OPERAND_TYPE_FIRST_VARIABLE_TYPE = ENUM
#define LAST_VARIABLE(ENUM)                         \
  ENUM, SPV_OPERAND_TYPE_LAST_VARIABLE_TYPE = ENUM, \
        SPV_OPERAND_TYPE_LAST_OPTIONAL_TYPE = ENUM

  // An optional operand represents zero or one logical operands.
  // In an instruction definition, this may only appear at the end of the
  // operand types.
  FIRST_OPTIONAL(SPV_OPERAND_TYPE_OPTIONAL_ID),
  // An optional image operand type.
  SPV_OPERAND_TYPE_OPTIONAL_IMAGE,
  // An optional memory access type.
  SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS,
  // An optional literal integer.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER,
  // An optional literal number, which may be either integer or floating point.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_NUMBER,
  // Like SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, but optional, and integral.
  SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER,
  // An optional literal string.
  SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING,
  // An optional execution mode.
  SPV_OPERAND_TYPE_OPTIONAL_EXECUTION_MODE,
  // An optional context-independent value, or CIV.  CIVs are tokens that we can
  // assemble regardless of where they occur -- literals, IDs, immediate
  // integers, etc.
  SPV_OPERAND_TYPE_OPTIONAL_CIV,

  // A variable operand represents zero or more logical operands.
  // In an instruction definition, this may only appear at the end of the
  // operand types.
  FIRST_VARIABLE(SPV_OPERAND_TYPE_VARIABLE_ID),
  SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER,
  // A sequence of zero or more pairs of (typed literal integer, Id).
  // Expands to zero or more:
  //  (SPV_OPERAND_TYPE_TYPED_LITERAL_INTEGER, SPV_OPERAND_TYPE_ID)
  // where the literal number must always be an integer of some sort.
  SPV_OPERAND_TYPE_VARIABLE_LITERAL_INTEGER_ID,
  // A sequence of zero or more pairs of (Id, Literal integer)
  SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL_INTEGER,
  // A sequence of zero or more execution modes
  LAST_VARIABLE(SPV_OPERAND_TYPE_VARIABLE_EXECUTION_MODE),

  // This is a sentinel value, and does not represent an operand type.
  // It should come last.
  SPV_OPERAND_TYPE_NUM_OPERAND_TYPES,

  SPV_FORCE_32_BIT_ENUM(spv_operand_type_t)
} spv_operand_type_t;

typedef enum spv_ext_inst_type_t {
  SPV_EXT_INST_TYPE_NONE = 0,
  SPV_EXT_INST_TYPE_GLSL_STD_450,
  SPV_EXT_INST_TYPE_OPENCL_STD,

  SPV_FORCE_32_BIT_ENUM(spv_ext_inst_type_t)
} spv_ext_inst_type_t;

typedef enum spv_binary_to_text_options_t {
  SPV_BINARY_TO_TEXT_OPTION_NONE = SPV_BIT(0),
  SPV_BINARY_TO_TEXT_OPTION_PRINT = SPV_BIT(1),
  SPV_BINARY_TO_TEXT_OPTION_COLOR = SPV_BIT(2),
  SPV_FORCE_32_BIT_ENUM(spv_binary_to_text_options_t)
} spv_binary_to_text_options_t;

typedef enum spv_validate_options_t {
  SPV_VALIDATE_BASIC_BIT = SPV_BIT(0),
  SPV_VALIDATE_LAYOUT_BIT = SPV_BIT(1),
  SPV_VALIDATE_ID_BIT = SPV_BIT(2),
  SPV_VALIDATE_RULES_BIT = SPV_BIT(3),
  SPV_VALIDATE_ALL = SPV_VALIDATE_BASIC_BIT | SPV_VALIDATE_LAYOUT_BIT |
                     SPV_VALIDATE_ID_BIT | SPV_VALIDATE_RULES_BIT,
  SPV_FORCE_32_BIT_ENUM(spv_validation_options_t)
} spv_validate_options_t;

// Structures

typedef struct spv_const_binary_t {
  const uint32_t* code;
  const size_t wordCount;
} spv_const_binary_t;

typedef struct spv_binary_t {
  uint32_t* code;
  size_t wordCount;
} spv_binary_t;

typedef struct spv_text_t {
  const char* str;
  size_t length;
} spv_text_t;

typedef struct spv_position_t {
  size_t line;
  size_t column;
  size_t index;
} spv_position_t;

typedef struct spv_diagnostic_t {
  spv_position_t position;
  char* error;
  bool isTextSource;
} spv_diagnostic_t;

// Type Definitions

typedef spv_const_binary_t* spv_const_binary;
typedef spv_binary_t* spv_binary;
typedef spv_text_t* spv_text;
typedef spv_position_t* spv_position;
typedef spv_diagnostic_t* spv_diagnostic;

// Platform API

// Encodes the given SPIR-V assembly text to its binary representation. The
// length parameter specifies the number of bytes for text. Encoded binary will
// be stored into *binary. Any error will be written into *diagnostic.
spv_result_t spvTextToBinary(const char* text, const size_t length,
                             spv_binary* binary, spv_diagnostic* diagnostic);

// @brief Frees an allocated text stream. This is a no-op if the text parameter
// is a null pointer.
void spvTextDestroy(spv_text text);

// Decodes the given SPIR-V binary representation to its assembly text. The
// word_count parameter specifies the number of words for binary. The options
// parameter is a bit field of spv_binary_to_text_options_t. Decoded text will
// be stored into *text. Any error will be written into *diagnostic.
spv_result_t spvBinaryToText(const uint32_t* binary, const size_t word_count,
                             const uint32_t options, spv_text* text,
                             spv_diagnostic* diagnostic);

// Frees a binary stream from memory. This is a no-op if binary is a null
// pointer.
void spvBinaryDestroy(spv_binary binary);

// Validates a SPIR-V binary for correctness. The options parameter is a bit
// field of spv_validation_options_t.
spv_result_t spvValidate(const spv_const_binary binary, const uint32_t options,
                         spv_diagnostic* pDiagnostic);

// Creates a diagnostic object. The position parameter specifies the location in
// the text/binary stream. The message parameter, copied into the diagnostic
// object, contains the error message to display.
spv_diagnostic spvDiagnosticCreate(const spv_position position,
                                   const char* message);

/// Destroys a diagnostic object.
void spvDiagnosticDestroy(spv_diagnostic diagnostic);

// Prints the diagnostic to stderr.
spv_result_t spvDiagnosticPrint(const spv_diagnostic diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LIBSPIRV_LIBSPIRV_LIBSPIRV_H_
