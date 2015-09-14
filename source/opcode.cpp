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

#include <libspirv/libspirv.h>
#include "binary.h"
#include "opcode.h"

#include <assert.h>
#include <string.h>

namespace {

// Descriptions of each opcode.  Each entry describes the format of the
// instruction that follows a particular opcode.
//
// Most fields are initialized statically by including an automatically
// generated file.
// The operandTypes fields are initialized during spvOpcodeInitialize().
//
// TODO(dneto): Some of the macros are quite unreadable.  We could make
// good use of constexpr functions, but some compilers don't support that yet.
spv_opcode_desc_t opcodeTableEntries[] = {
#define EmptyList {}
#define List(...) {__VA_ARGS__}
#define Capability(X) Capability##X
#define CapabilityNone -1
#define Instruction(Name,HasResult,HasType,NumLogicalOperands,NumCapabilities,CapabilityRequired,IsVariable,LogicalArgsList) \
  { #Name, \
    Op##Name, \
    ((CapabilityRequired != CapabilityNone ? SPV_OPCODE_FLAGS_CAPABILITIES : 0)), \
    uint32_t(CapabilityRequired), \
    0, {}, /* Filled in later. Operand list, including result id and type id, if needed */ \
    HasResult, \
    HasType, \
    LogicalArgsList },
#include "opcode.inc"
#undef EmptyList
#undef List
#undef Capability
#undef CapabilityNone
#undef Instruction
};

// Has the opcodeTableEntries table been fully elaborated?
// That is, are the operandTypes fields initialized?
bool opcodeTableInitialized = false;

// Opcode API

// Converts the given operand class enum (from the SPIR-V document generation
// logic) to the operand type required by the parser.
// This only applies to logical operands.
spv_operand_type_t convertOperandClassToType(spv::Op opcode,
                                             spv::OperandClass operandClass) {
  // The spec document generator uses OptionalOperandLiteral for several kinds
  // of repeating values.  Our parser needs more specific information about
  // what is being repeated.
  if (operandClass == OperandOptionalLiteral) {
    switch (opcode) {
      case spv::OpLoad:
      case spv::OpStore:
      case spv::OpCopyMemory:
      case spv::OpCopyMemorySized:
        return SPV_OPERAND_TYPE_VARIABLE_MEMORY_ACCESS;
      case spv::OpExecutionMode:
        return SPV_OPERAND_TYPE_VARIABLE_EXECUTION_MODE;
      default:
        break;
    }
  }

  switch(operandClass) {
    case OperandNone: return SPV_OPERAND_TYPE_NONE;
    case OperandId: return SPV_OPERAND_TYPE_ID;
    case OperandOptionalId: return SPV_OPERAND_TYPE_OPTIONAL_ID;
    case OperandOptionalImage: return SPV_OPERAND_TYPE_OPTIONAL_IMAGE; // TODO(dneto): This is variable.
    case OperandVariableIds: return SPV_OPERAND_TYPE_VARIABLE_ID;
    case OperandOptionalLiteral: return SPV_OPERAND_TYPE_OPTIONAL_LITERAL;
    case OperandOptionalLiteralString: return SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING;
    case OperandVariableLiterals: return SPV_OPERAND_TYPE_VARIABLE_LITERAL;
    case OperandLiteralNumber: return SPV_OPERAND_TYPE_LITERAL_NUMBER;
    case OperandLiteralString: return SPV_OPERAND_TYPE_LITERAL_STRING;
    case OperandSource: return SPV_OPERAND_TYPE_SOURCE_LANGUAGE;
    case OperandExecutionModel: return SPV_OPERAND_TYPE_EXECUTION_MODEL;
    case OperandAddressing: return SPV_OPERAND_TYPE_ADDRESSING_MODEL;
    case OperandMemory: return SPV_OPERAND_TYPE_MEMORY_MODEL;
    case OperandExecutionMode: return SPV_OPERAND_TYPE_EXECUTION_MODE;
    case OperandStorage: return SPV_OPERAND_TYPE_STORAGE_CLASS;
    case OperandDimensionality: return SPV_OPERAND_TYPE_DIMENSIONALITY;
    case OperandSamplerAddressingMode: return SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE;
    case OperandSamplerFilterMode: return SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE;
    case OperandSamplerImageFormat: return SPV_OPERAND_TYPE_NONE; //TODO
    case OperandImageChannelOrder: return SPV_OPERAND_TYPE_NONE; //TODO
    case OperandImageChannelDataType: return SPV_OPERAND_TYPE_NONE; //TODO
    case OperandImageOperands: return SPV_OPERAND_TYPE_NONE; //TODO
    case OperandFPFastMath: return SPV_OPERAND_TYPE_FP_FAST_MATH_MODE;
    case OperandFPRoundingMode: return SPV_OPERAND_TYPE_FP_ROUNDING_MODE;
    case OperandLinkageType: return SPV_OPERAND_TYPE_LINKAGE_TYPE;
    case OperandAccessQualifier: return SPV_OPERAND_TYPE_ACCESS_QUALIFIER;
    case OperandFuncParamAttr: return SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE;
    case OperandDecoration: return SPV_OPERAND_TYPE_DECORATION;
    case OperandBuiltIn: return SPV_OPERAND_TYPE_BUILT_IN;
    case OperandSelect: return SPV_OPERAND_TYPE_SELECTION_CONTROL;
    case OperandLoop: return SPV_OPERAND_TYPE_LOOP_CONTROL;
    case OperandFunction: return SPV_OPERAND_TYPE_FUNCTION_CONTROL;
    case OperandMemorySemantics: return SPV_OPERAND_TYPE_MEMORY_SEMANTICS;
    case OperandMemoryAccess:
      // This case does not occur in Rev 31.
      // We expect that it will become SPV_OPERAND_TYPE_VARIABLE_MEMORY_ACCESS,
      // and we can remove the special casing above for memory operation
      // instructions.
      break;
    case OperandScope: return SPV_OPERAND_TYPE_EXECUTION_SCOPE;
    case OperandGroupOperation: return SPV_OPERAND_TYPE_GROUP_OPERATION;
    case OperandKernelEnqueueFlags: return SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS;
    case OperandKernelProfilingInfo: return SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO;
    case OperandCapability: return SPV_OPERAND_TYPE_CAPABILITY;

    // Used by GroupMemberDecorate
    case OperandVariableIdLiteral: return SPV_OPERAND_TYPE_VARIABLE_ID_LITERAL;

    // Used by Switch
    case OperandVariableLiteralId: return SPV_OPERAND_TYPE_VARIABLE_LITERAL_ID;

    // These exceptional cases shouldn't occur.
    case OperandCount:
    default:
      break;
  }
  assert(0 && "Unexpected operand class");
  return SPV_OPERAND_TYPE_NONE;
}

}  // anonymous namespace

// Finish populating the opcodeTableEntries array.
void spvOpcodeTableInitialize() {
  // Compute the operandTypes field for each entry.
  for (auto &opcode : opcodeTableEntries) {
    opcode.numTypes = 0;
    // Type ID always comes first, if present.
    if (opcode.hasType)
      opcode.operandTypes[opcode.numTypes++] = SPV_OPERAND_TYPE_ID;
    // Result ID always comes next, if present
    if (opcode.hasResult)
      opcode.operandTypes[opcode.numTypes++] = SPV_OPERAND_TYPE_RESULT_ID;
    const uint16_t maxNumOperands =
        sizeof(opcode.operandTypes) / sizeof(opcode.operandTypes[0]);
    const uint16_t maxNumClasses =
        sizeof(opcode.operandClass) / sizeof(opcode.operandClass[0]);
    for (uint16_t classIndex = 0;
         opcode.numTypes < maxNumOperands && classIndex < maxNumClasses;
         classIndex++) {
      const OperandClass operandClass = opcode.operandClass[classIndex];
      opcode.operandTypes[opcode.numTypes++] =
          convertOperandClassToType(opcode.opcode, operandClass);
      // The OperandNone value is not explicitly represented in the .inc file.
      // However, it is the zero value, and is created via implicit value
      // initialization.
      if (operandClass == OperandNone) {
        opcode.numTypes--;
        break;
      }
    }
    // We should have written the terminating SPV_OPERAND_TYPE_NONE entry, but
    // also without overflowing.
    assert((opcode.numTypes < maxNumOperands) &&
           "Operand class list is too long.  Expand "
           "spv_opcode_desc_t.operandClass");
  }
  opcodeTableInitialized = true;
}

const char *spvGeneratorStr(uint32_t generator) {
  switch (generator) {
    case SPV_GENERATOR_KHRONOS:
      return "Khronos";
    case SPV_GENERATOR_VALVE:
      return "Valve";
    case SPV_GENERATOR_LUNARG:
      return "LunarG";
    case SPV_GENERATOR_CODEPLAY:
      return "Codeplay Software Ltd.";
    default:
      return "Unknown";
  }
}

uint32_t spvOpcodeMake(uint16_t wordCount, Op opcode) {
  return ((uint32_t)opcode) | (((uint32_t)wordCount) << 16);
}

void spvOpcodeSplit(const uint32_t word, uint16_t *pWordCount, Op *pOpcode) {
  if (pWordCount) {
    *pWordCount = (uint16_t)((0xffff0000 & word) >> 16);
  }
  if (pOpcode) {
    *pOpcode = (Op)(0x0000ffff & word);
  }
}

spv_result_t spvOpcodeTableGet(spv_opcode_table *pInstTable) {
  if (!pInstTable) return SPV_ERROR_INVALID_POINTER;

  static spv_opcode_table_t table = {
      sizeof(opcodeTableEntries) / sizeof(spv_opcode_desc_t),
      opcodeTableEntries};

  // TODO(dneto): Consider thread safety of initialization.
  // That is, ordering effects of the flag vs. the table updates.
  if (!opcodeTableInitialized) spvOpcodeTableInitialize();

  *pInstTable = &table;

  return SPV_SUCCESS;
}

spv_result_t spvOpcodeTableNameLookup(const spv_opcode_table table,
                                      const char *name,
                                      spv_opcode_desc *pEntry) {
  if (!name || !pEntry) return SPV_ERROR_INVALID_POINTER;
  if (!table) return SPV_ERROR_INVALID_TABLE;

  // TODO: This lookup of the Opcode table is suboptimal! Binary sort would be
  // preferable but the table requires sorting on the Opcode name, but it's
  // static
  // const initialized and matches the order of the spec.
  const size_t nameLength = strlen(name);
  for (uint64_t opcodeIndex = 0; opcodeIndex < table->count; ++opcodeIndex) {
    if (nameLength == strlen(table->entries[opcodeIndex].name) &&
        !strncmp(name, table->entries[opcodeIndex].name, nameLength)) {
      // NOTE: Found out Opcode!
      *pEntry = &table->entries[opcodeIndex];
      return SPV_SUCCESS;
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

spv_result_t spvOpcodeTableValueLookup(const spv_opcode_table table,
                                       const Op opcode,
                                       spv_opcode_desc *pEntry) {
  if (!table) return SPV_ERROR_INVALID_TABLE;
  if (!pEntry) return SPV_ERROR_INVALID_POINTER;

  // TODO: As above this lookup is not optimal.
  for (uint64_t opcodeIndex = 0; opcodeIndex < table->count; ++opcodeIndex) {
    if (opcode == table->entries[opcodeIndex].opcode) {
      // NOTE: Found the Opcode!
      *pEntry = &table->entries[opcodeIndex];
      return SPV_SUCCESS;
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

int16_t spvOpcodeResultIdIndex(spv_opcode_desc entry) {
  for (int16_t i = 0; i < entry->numTypes; ++i) {
    if (SPV_OPERAND_TYPE_RESULT_ID == entry->operandTypes[i]) return i;
  }
  return SPV_OPERAND_INVALID_RESULT_ID_INDEX;
}

int32_t spvOpcodeRequiresCapabilities(spv_opcode_desc entry) {
  return SPV_OPCODE_FLAGS_CAPABILITIES ==
         (SPV_OPCODE_FLAGS_CAPABILITIES & entry->flags);
}

void spvInstructionCopy(const uint32_t *words, const Op opcode,
                        const uint16_t wordCount, const spv_endianness_t endian,
                        spv_instruction_t *pInst) {
  pInst->opcode = opcode;
  pInst->wordCount = wordCount;
  for (uint16_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    pInst->words[wordIndex] = spvFixWord(words[wordIndex], endian);
    if (!wordIndex) {
      uint16_t thisWordCount;
      Op thisOpcode;
      spvOpcodeSplit(pInst->words[wordIndex], &thisWordCount, &thisOpcode);
      assert(opcode == thisOpcode && wordCount == thisWordCount &&
             "Endianness failed!");
    }
  }
}

const char *spvOpcodeString(const Op opcode) {
#define CASE(OPCODE) \
  case OPCODE:       \
    return #OPCODE;
  switch (opcode) {
    CASE(OpNop)
    CASE(OpSource)
    CASE(OpSourceExtension)
    CASE(OpExtension)
    CASE(OpExtInstImport)
    CASE(OpMemoryModel)
    CASE(OpEntryPoint)
    CASE(OpExecutionMode)
    CASE(OpTypeVoid)
    CASE(OpTypeBool)
    CASE(OpTypeInt)
    CASE(OpTypeFloat)
    CASE(OpTypeVector)
    CASE(OpTypeMatrix)
    CASE(OpTypeSampler)
    CASE(OpTypeArray)
    CASE(OpTypeRuntimeArray)
    CASE(OpTypeStruct)
    CASE(OpTypeOpaque)
    CASE(OpTypePointer)
    CASE(OpTypeFunction)
    CASE(OpTypeEvent)
    CASE(OpTypeDeviceEvent)
    CASE(OpTypeReserveId)
    CASE(OpTypeQueue)
    CASE(OpTypePipe)
    CASE(OpConstantTrue)
    CASE(OpConstantFalse)
    CASE(OpConstant)
    CASE(OpConstantComposite)
    CASE(OpConstantSampler)
    CASE(OpConstantNull)
    CASE(OpSpecConstantTrue)
    CASE(OpSpecConstantFalse)
    CASE(OpSpecConstant)
    CASE(OpSpecConstantComposite)
    CASE(OpVariable)
    CASE(OpFunction)
    CASE(OpFunctionParameter)
    CASE(OpFunctionEnd)
    CASE(OpFunctionCall)
    CASE(OpExtInst)
    CASE(OpUndef)
    CASE(OpLoad)
    CASE(OpStore)
    CASE(OpPhi)
    CASE(OpDecorationGroup)
    CASE(OpDecorate)
    CASE(OpMemberDecorate)
    CASE(OpGroupDecorate)
    CASE(OpGroupMemberDecorate)
    CASE(OpName)
    CASE(OpMemberName)
    CASE(OpString)
    CASE(OpLine)
    CASE(OpVectorExtractDynamic)
    CASE(OpVectorInsertDynamic)
    CASE(OpVectorShuffle)
    CASE(OpCompositeConstruct)
    CASE(OpCompositeExtract)
    CASE(OpCompositeInsert)
    CASE(OpCopyObject)
    CASE(OpCopyMemory)
    CASE(OpCopyMemorySized)
    CASE(OpAccessChain)
    CASE(OpInBoundsAccessChain)
    CASE(OpSNegate)
    CASE(OpFNegate)
    CASE(OpNot)
    CASE(OpAny)
    CASE(OpAll)
    CASE(OpConvertFToU)
    CASE(OpConvertFToS)
    CASE(OpConvertSToF)
    CASE(OpConvertUToF)
    CASE(OpUConvert)
    CASE(OpSConvert)
    CASE(OpFConvert)
    CASE(OpConvertPtrToU)
    CASE(OpConvertUToPtr)
    CASE(OpPtrCastToGeneric)
    CASE(OpGenericCastToPtr)
    CASE(OpBitcast)
    CASE(OpTranspose)
    CASE(OpIsNan)
    CASE(OpIsInf)
    CASE(OpIsFinite)
    CASE(OpIsNormal)
    CASE(OpSignBitSet)
    CASE(OpLessOrGreater)
    CASE(OpOrdered)
    CASE(OpUnordered)
    CASE(OpArrayLength)
    CASE(OpIAdd)
    CASE(OpFAdd)
    CASE(OpISub)
    CASE(OpFSub)
    CASE(OpIMul)
    CASE(OpFMul)
    CASE(OpUDiv)
    CASE(OpSDiv)
    CASE(OpFDiv)
    CASE(OpUMod)
    CASE(OpSRem)
    CASE(OpSMod)
    CASE(OpFRem)
    CASE(OpFMod)
    CASE(OpVectorTimesScalar)
    CASE(OpMatrixTimesScalar)
    CASE(OpVectorTimesMatrix)
    CASE(OpMatrixTimesVector)
    CASE(OpMatrixTimesMatrix)
    CASE(OpOuterProduct)
    CASE(OpDot)
    CASE(OpShiftRightLogical)
    CASE(OpShiftRightArithmetic)
    CASE(OpShiftLeftLogical)
    CASE(OpLogicalOr)
    CASE(OpLogicalAnd)
    CASE(OpBitwiseOr)
    CASE(OpBitwiseXor)
    CASE(OpBitwiseAnd)
    CASE(OpSelect)
    CASE(OpIEqual)
    CASE(OpFOrdEqual)
    CASE(OpFUnordEqual)
    CASE(OpINotEqual)
    CASE(OpFOrdNotEqual)
    CASE(OpFUnordNotEqual)
    CASE(OpULessThan)
    CASE(OpSLessThan)
    CASE(OpFOrdLessThan)
    CASE(OpFUnordLessThan)
    CASE(OpUGreaterThan)
    CASE(OpSGreaterThan)
    CASE(OpFOrdGreaterThan)
    CASE(OpFUnordGreaterThan)
    CASE(OpULessThanEqual)
    CASE(OpSLessThanEqual)
    CASE(OpFOrdLessThanEqual)
    CASE(OpFUnordLessThanEqual)
    CASE(OpUGreaterThanEqual)
    CASE(OpSGreaterThanEqual)
    CASE(OpFOrdGreaterThanEqual)
    CASE(OpFUnordGreaterThanEqual)
    CASE(OpDPdx)
    CASE(OpDPdy)
    CASE(OpFwidth)
    CASE(OpDPdxFine)
    CASE(OpDPdyFine)
    CASE(OpFwidthFine)
    CASE(OpDPdxCoarse)
    CASE(OpDPdyCoarse)
    CASE(OpFwidthCoarse)
    CASE(OpEmitVertex)
    CASE(OpEndPrimitive)
    CASE(OpEmitStreamVertex)
    CASE(OpEndStreamPrimitive)
    CASE(OpControlBarrier)
    CASE(OpMemoryBarrier)
    CASE(OpAtomicLoad)
    CASE(OpAtomicStore)
    CASE(OpAtomicExchange)
    CASE(OpAtomicCompareExchange)
    CASE(OpAtomicCompareExchangeWeak)
    CASE(OpAtomicIIncrement)
    CASE(OpAtomicIDecrement)
    CASE(OpAtomicIAdd)
    CASE(OpAtomicISub)
    CASE(OpAtomicUMin)
    CASE(OpAtomicUMax)
    CASE(OpAtomicAnd)
    CASE(OpAtomicOr)
    CASE(OpAtomicXor)
    CASE(OpLoopMerge)
    CASE(OpSelectionMerge)
    CASE(OpLabel)
    CASE(OpBranch)
    CASE(OpBranchConditional)
    CASE(OpSwitch)
    CASE(OpKill)
    CASE(OpReturn)
    CASE(OpReturnValue)
    CASE(OpUnreachable)
    CASE(OpLifetimeStart)
    CASE(OpLifetimeStop)
    CASE(OpAsyncGroupCopy)
    CASE(OpWaitGroupEvents)
    CASE(OpGroupAll)
    CASE(OpGroupAny)
    CASE(OpGroupBroadcast)
    CASE(OpGroupIAdd)
    CASE(OpGroupFAdd)
    CASE(OpGroupFMin)
    CASE(OpGroupUMin)
    CASE(OpGroupSMin)
    CASE(OpGroupFMax)
    CASE(OpGroupUMax)
    CASE(OpGroupSMax)
    CASE(OpGenericCastToPtrExplicit)
    CASE(OpGenericPtrMemSemantics)
    CASE(OpReadPipe)
    CASE(OpWritePipe)
    CASE(OpReservedReadPipe)
    CASE(OpReservedWritePipe)
    CASE(OpReserveReadPipePackets)
    CASE(OpReserveWritePipePackets)
    CASE(OpCommitReadPipe)
    CASE(OpCommitWritePipe)
    CASE(OpIsValidReserveId)
    CASE(OpGetNumPipePackets)
    CASE(OpGetMaxPipePackets)
    CASE(OpGroupReserveReadPipePackets)
    CASE(OpGroupReserveWritePipePackets)
    CASE(OpGroupCommitReadPipe)
    CASE(OpGroupCommitWritePipe)
    CASE(OpEnqueueMarker)
    CASE(OpEnqueueKernel)
    CASE(OpGetKernelNDrangeSubGroupCount)
    CASE(OpGetKernelNDrangeMaxSubGroupSize)
    CASE(OpGetKernelWorkGroupSize)
    CASE(OpGetKernelPreferredWorkGroupSizeMultiple)
    CASE(OpRetainEvent)
    CASE(OpReleaseEvent)
    CASE(OpCreateUserEvent)
    CASE(OpIsValidEvent)
    CASE(OpSetUserEventStatus)
    CASE(OpCaptureEventProfilingInfo)
    CASE(OpGetDefaultQueue)
    CASE(OpBuildNDRange)
    default:
      assert(0 && "Unreachable!");
  }
#undef CASE
  return "unknown";
}

int32_t spvOpcodeIsType(const Op opcode) {
  switch (opcode) {
    case OpTypeVoid:
    case OpTypeBool:
    case OpTypeInt:
    case OpTypeFloat:
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeSampler:
    case OpTypeArray:
    case OpTypeRuntimeArray:
    case OpTypeStruct:
    case OpTypeOpaque:
    case OpTypePointer:
    case OpTypeFunction:
    case OpTypeEvent:
    case OpTypeDeviceEvent:
    case OpTypeReserveId:
    case OpTypeQueue:
    case OpTypePipe:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsScalarType(const Op opcode) {
  switch (opcode) {
    case OpTypeInt:
    case OpTypeFloat:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsConstant(const Op opcode) {
  switch (opcode) {
    case OpConstantTrue:
    case OpConstantFalse:
    case OpConstant:
    case OpConstantComposite:
    case OpConstantSampler:
    // case OpConstantNull:
    case OpConstantNull:
    case OpSpecConstantTrue:
    case OpSpecConstantFalse:
    case OpSpecConstant:
    case OpSpecConstantComposite:
      // case OpSpecConstantOp:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsComposite(const Op opcode) {
  switch (opcode) {
    case OpTypeVector:
    case OpTypeMatrix:
    case OpTypeArray:
    case OpTypeStruct:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeAreTypesEqual(const spv_instruction_t *pTypeInst0,
                               const spv_instruction_t *pTypeInst1) {
  if (pTypeInst0->opcode != pTypeInst1->opcode) return false;
  if (pTypeInst0->words[1] != pTypeInst1->words[1]) return false;
  return true;
}

int32_t spvOpcodeIsPointer(const Op opcode) {
  switch (opcode) {
    case OpVariable:
    case OpAccessChain:
    case OpInBoundsAccessChain:
    case OpFunctionParameter:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsObject(const Op opcode) {
  switch (opcode) {
    case OpConstantTrue:
    case OpConstantFalse:
    case OpConstant:
    case OpConstantComposite:
    // TODO: case OpConstantSampler:
    case OpConstantNull:
    case OpSpecConstantTrue:
    case OpSpecConstantFalse:
    case OpSpecConstant:
    case OpSpecConstantComposite:
    // TODO: case OpSpecConstantOp:
    case OpVariable:
    case OpAccessChain:
    case OpInBoundsAccessChain:
    case OpConvertFToU:
    case OpConvertFToS:
    case OpConvertSToF:
    case OpConvertUToF:
    case OpUConvert:
    case OpSConvert:
    case OpFConvert:
    case OpConvertPtrToU:
    // TODO: case OpConvertUToPtr:
    case OpPtrCastToGeneric:
    // TODO: case OpGenericCastToPtr:
    case OpBitcast:
    // TODO: case OpGenericCastToPtrExplicit:
    case OpSatConvertSToU:
    case OpSatConvertUToS:
    case OpVectorExtractDynamic:
    case OpCompositeConstruct:
    case OpCompositeExtract:
    case OpCopyObject:
    case OpTranspose:
    case OpSNegate:
    case OpFNegate:
    case OpNot:
    case OpIAdd:
    case OpFAdd:
    case OpISub:
    case OpFSub:
    case OpIMul:
    case OpFMul:
    case OpUDiv:
    case OpSDiv:
    case OpFDiv:
    case OpUMod:
    case OpSRem:
    case OpSMod:
    case OpVectorTimesScalar:
    case OpMatrixTimesScalar:
    case OpVectorTimesMatrix:
    case OpMatrixTimesVector:
    case OpMatrixTimesMatrix:
    case OpOuterProduct:
    case OpDot:
    case OpShiftRightLogical:
    case OpShiftRightArithmetic:
    case OpShiftLeftLogical:
    case OpBitwiseOr:
    case OpBitwiseXor:
    case OpBitwiseAnd:
    case OpAny:
    case OpAll:
    case OpIsNan:
    case OpIsInf:
    case OpIsFinite:
    case OpIsNormal:
    case OpSignBitSet:
    case OpLessOrGreater:
    case OpOrdered:
    case OpUnordered:
    case OpLogicalOr:
    case OpLogicalAnd:
    case OpSelect:
    case OpIEqual:
    case OpFOrdEqual:
    case OpFUnordEqual:
    case OpINotEqual:
    case OpFOrdNotEqual:
    case OpFUnordNotEqual:
    case OpULessThan:
    case OpSLessThan:
    case OpFOrdLessThan:
    case OpFUnordLessThan:
    case OpUGreaterThan:
    case OpSGreaterThan:
    case OpFOrdGreaterThan:
    case OpFUnordGreaterThan:
    case OpULessThanEqual:
    case OpSLessThanEqual:
    case OpFOrdLessThanEqual:
    case OpFUnordLessThanEqual:
    case OpUGreaterThanEqual:
    case OpSGreaterThanEqual:
    case OpFOrdGreaterThanEqual:
    case OpFUnordGreaterThanEqual:
    case OpDPdx:
    case OpDPdy:
    case OpFwidth:
    case OpDPdxFine:
    case OpDPdyFine:
    case OpFwidthFine:
    case OpDPdxCoarse:
    case OpDPdyCoarse:
    case OpFwidthCoarse:
    case OpReturnValue:
      return true;
    default:
      return false;
  }
}

int32_t spvOpcodeIsBasicTypeNullable(Op opcode) {
  switch (opcode) {
    case OpTypeBool:
    case OpTypeInt:
    case OpTypeFloat:
    case OpTypePointer:
    case OpTypeEvent:
    case OpTypeDeviceEvent:
    case OpTypeReserveId:
    case OpTypeQueue:
      return true;
    default:
      return false;
  }
}

int32_t spvInstructionIsInBasicBlock(const spv_instruction_t *pFirstInst,
                                     const spv_instruction_t *pInst) {
  while (pFirstInst != pInst) {
    if (OpFunction == pInst->opcode) break;
    pInst--;
  }
  if (OpFunction != pInst->opcode) return false;
  return true;
}

int32_t spvOpcodeIsValue(Op opcode) {
  if (spvOpcodeIsPointer(opcode)) return true;
  if (spvOpcodeIsConstant(opcode)) return true;
  switch (opcode) {
    case OpLoad:
      // TODO: Other Opcode's resulting in a value
      return true;
    default:
      return false;
  }
}
