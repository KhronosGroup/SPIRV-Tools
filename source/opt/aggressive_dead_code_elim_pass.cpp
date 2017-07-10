// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "aggressive_dead_code_elim_pass.h"

#include "iterator.h"

#include "spirv/1.0/GLSL.std.450.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kEntryPointFunctionIdInIdx = 1;
const uint32_t kStorePtrIdInIdx = 0;
const uint32_t kLoadPtrIdInIdx = 0;
const uint32_t kAccessChainPtrIdInIdx = 0;
const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kCopyObjectOperandInIdx = 0;
const uint32_t kExtInstSetIdInIndx = 0;
const uint32_t kExtInstInstructionInIndx = 1;

}  // namespace anonymous

bool AggressiveDCEPass::IsNonPtrAccessChain(const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

ir::Instruction* AggressiveDCEPass::GetPtr(
      ir::Instruction* ip, uint32_t* varId) {
  const SpvOp op = ip->opcode();
  assert(op == SpvOpStore || op == SpvOpLoad);
  *varId = ip->GetSingleWordInOperand(
      op == SpvOpStore ? kStorePtrIdInIdx : kLoadPtrIdInIdx);
  ir::Instruction* ptrInst = def_use_mgr_->GetDef(*varId);
  ir::Instruction* varInst = ptrInst;
  while (varInst->opcode() != SpvOpVariable) {
    if (IsNonPtrAccessChain(varInst->opcode())) {
      *varId = varInst->GetSingleWordInOperand(kAccessChainPtrIdInIdx);
    }
    else {
      assert(varInst->opcode() == SpvOpCopyObject);
      *varId = varInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    }
    varInst = def_use_mgr_->GetDef(*varId);
  }
  return ptrInst;
}

bool AggressiveDCEPass::IsLocalVar(uint32_t varId) {
  const ir::Instruction* varInst = def_use_mgr_->GetDef(varId);
  assert(varInst->opcode() == SpvOpVariable);
  const uint32_t varTypeId = varInst->type_id();
  const ir::Instruction* varTypeInst = def_use_mgr_->GetDef(varTypeId);
  return varTypeInst->GetSingleWordInOperand(kTypePointerStorageClassInIdx) ==
      SpvStorageClassFunction;
}

void AggressiveDCEPass::AddStores(uint32_t ptrId) {
  const analysis::UseList* uses = def_use_mgr_->GetUses(ptrId);
  if (uses == nullptr)
    return;
  for (const auto u : *uses) {
    const SpvOp op = u.inst->opcode();
    switch (op) {
      case SpvOpAccessChain:
      case SpvOpInBoundsAccessChain:
      case SpvOpCopyObject: {
        AddStores(u.inst->result_id());
      } break;
      case SpvOpLoad:
        break;
      // Assume it stores eg frexp, modf
      case SpvOpStore:
      default: {
        if (live_insts_.find(u.inst) == live_insts_.end())
          worklist_.push(u.inst);
      } break;
    }
  }
}

bool AggressiveDCEPass::IsCombinator(uint32_t op) const {
  return combinator_ops_shader_.find(op) != combinator_ops_shader_.end();
}

bool AggressiveDCEPass::IsCombinatorExt(ir::Instruction* inst) const {
  assert(inst->opcode() == SpvOpExtInst);
  if (inst->GetSingleWordInOperand(kExtInstSetIdInIndx) == glsl_std_450_id_) {
    uint32_t op = inst->GetSingleWordInOperand(kExtInstInstructionInIndx);
    return combinator_ops_glsl_std_450_.find(op) !=
        combinator_ops_glsl_std_450_.end();
  }
  else
    return false;
}

bool AggressiveDCEPass::AllExtensionsSupported() {
  uint32_t ecnt = 0;
  for (auto& ei : module_->extensions()) {
    (void) ei;
    ++ecnt;
  }
  return ecnt == 0;
}

void AggressiveDCEPass::KillInstIfTargetDead(ir::Instruction* inst) {
  const uint32_t tId = inst->GetSingleWordInOperand(0);
  const ir::Instruction* tInst = def_use_mgr_->GetDef(tId);
  if (dead_insts_.find(tInst) != dead_insts_.end())
    def_use_mgr_->KillInst(inst);
}

bool AggressiveDCEPass::AggressiveDCE(ir::Function* func) {
  bool modified = false;
  // Add all control flow and instructions with external side effects 
  // to worklist
  // TODO(greg-lunarg): Handle Frexp, Modf more optimally
  // TODO(greg-lunarg): Handle FunctionCall more optimally
  // TODO(greg-lunarg): Handle CopyMemory more optimally
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      uint32_t op = inst.opcode();
      switch (op) {
        case SpvOpStore: {
          uint32_t varId;
          (void) GetPtr(&inst, &varId);
          // non-function-scope stores
          if (!IsLocalVar(varId)) {
            worklist_.push(&inst);
          }
        } break;
        case SpvOpExtInst: {
          // eg. GLSL frexp, modf
          if (!IsCombinatorExt(&inst))
            worklist_.push(&inst);
        } break;
        case SpvOpCopyMemory:
        case SpvOpFunctionCall: {
          return false;
        } break;
        default: {
          // eg. control flow, function call, atomics
          if (!IsCombinator(op))
            worklist_.push(&inst);
        } break;
      }
    }
  }
  // Add OpGroupDecorates to worklist because they are a pain to remove
  // ids from.
  // TODO(greg-lunarg): Handle dead ids in OpGroupDecorate
  for (auto& ai : module_->annotations()) {
    if (ai.opcode() == SpvOpGroupDecorate)
      worklist_.push(&ai);
  }
  // Perform closure on live instruction set. 
  while (!worklist_.empty()) {
    ir::Instruction* liveInst = worklist_.front();
    live_insts_.insert(liveInst);
    // Add all operand instructions if not already live
    liveInst->ForEachInId([this](const uint32_t* iid) {
      ir::Instruction* inInst = def_use_mgr_->GetDef(*iid);
      if (live_insts_.find(inInst) == live_insts_.end())
        worklist_.push(inInst);
    });
    // If local load, add all variable's stores if variable not already live
    if (liveInst->opcode() == SpvOpLoad) {
      uint32_t varId;
      (void) GetPtr(liveInst, &varId);
      if (IsLocalVar(varId)) {
        if (live_local_vars_.find(varId) == live_local_vars_.end()) {
          AddStores(varId);
          live_local_vars_.insert(varId);
        }
      }
    }
    worklist_.pop();
  }
  // Mark all non-live instructions dead
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      if (live_insts_.find(&inst) != live_insts_.end())
        continue;
      dead_insts_.insert(&inst);
    }
  }
  // Remove debug and annotation statements referencing dead instructions.
  // This must be done before killing the instructions, otherwise there are
  // dead objects in the def/use database.
  for (auto& di : module_->debugs()) {
    if (di.opcode() != SpvOpName)
      continue;
    KillInstIfTargetDead(&di);
    modified = true;
  }
  for (auto& ai : module_->annotations()) {
    if (ai.opcode() != SpvOpDecorate && ai.opcode() != SpvOpDecorateId)
      continue;
    KillInstIfTargetDead(&ai);
    modified = true;
  }
  // Kill dead instructions
  for (auto& blk : *func) {
    for (auto& inst : blk) {
      if (dead_insts_.find(&inst) == dead_insts_.end())
        continue;
      def_use_mgr_->KillInst(&inst);
      modified = true;
    }
  }
  return modified;
}

void AggressiveDCEPass::Initialize(ir::Module* module) {

  module_ = module;

  // Initialize id-to-function map
  id2function_.clear();
  for (auto& fn : *module_)
    id2function_[fn.result_id()] = &fn;

  // Clear collections
  worklist_ = std::queue<ir::Instruction*>{};
  live_insts_.clear();
  live_local_vars_.clear();
  dead_insts_.clear();
  combinator_ops_shader_.clear();
  combinator_ops_glsl_std_450_.clear();

  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module_));
}

Pass::Status AggressiveDCEPass::ProcessImpl() {
  // Current functionality assumes shader capability 
  // TODO(greg-lunarg): Handle additional capabilities
  if (!module_->HasCapability(SpvCapabilityShader))
    return Status::SuccessWithoutChange;

  // Current functionality assumes logical addressing only
  // TODO(greg-lunarg): Handle non-logical addressing
  if (module_->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;

  // If any extensions in the module are not explicitly supported,
  // return unmodified. Currently, no extensions are supported.
  // glsl_std_450 extended instructions are allowed.
  // TODO(greg-lunarg): Allow additional extensions
  if (!AllExtensionsSupported())
    return Status::SuccessWithoutChange;

  InitCombinatorSets();

  bool modified = false;
  for (auto& e : module_->entry_points()) {
    ir::Function* fn =
        id2function_[e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx)];
    modified = AggressiveDCE(fn) || modified;
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

AggressiveDCEPass::AggressiveDCEPass()
    : module_(nullptr), def_use_mgr_(nullptr) {}

Pass::Status AggressiveDCEPass::Process(ir::Module* module) {
  Initialize(module);
  return ProcessImpl();
}

void AggressiveDCEPass::InitCombinatorSets() {
  combinator_ops_shader_ = {
    SpvOpNop,
    SpvOpUndef,
    SpvOpVariable,
    SpvOpImageTexelPointer,
    SpvOpLoad,
    SpvOpAccessChain,
    SpvOpInBoundsAccessChain,
    SpvOpArrayLength,
    SpvOpVectorExtractDynamic,
    SpvOpVectorInsertDynamic,
    SpvOpVectorShuffle,
    SpvOpCompositeConstruct,
    SpvOpCompositeExtract,
    SpvOpCompositeInsert,
    SpvOpCopyObject,
    SpvOpTranspose,
    SpvOpSampledImage,
    SpvOpImageSampleImplicitLod,
    SpvOpImageSampleExplicitLod,
    SpvOpImageSampleDrefImplicitLod,
    SpvOpImageSampleDrefExplicitLod,
    SpvOpImageSampleProjImplicitLod,
    SpvOpImageSampleProjExplicitLod,
    SpvOpImageSampleProjDrefImplicitLod,
    SpvOpImageSampleProjDrefExplicitLod,
    SpvOpImageFetch,
    SpvOpImageGather,
    SpvOpImageDrefGather,
    SpvOpImageRead,
    SpvOpImage,
    SpvOpConvertFToU,
    SpvOpConvertFToS,
    SpvOpConvertSToF,
    SpvOpConvertUToF,
    SpvOpUConvert,
    SpvOpSConvert,
    SpvOpFConvert,
    SpvOpQuantizeToF16,
    SpvOpBitcast,
    SpvOpSNegate,
    SpvOpFNegate,
    SpvOpIAdd,
    SpvOpFAdd,
    SpvOpISub,
    SpvOpFSub,
    SpvOpIMul,
    SpvOpFMul,
    SpvOpUDiv,
    SpvOpSDiv,
    SpvOpFDiv,
    SpvOpUMod,
    SpvOpSRem,
    SpvOpSMod,
    SpvOpFRem,
    SpvOpFMod,
    SpvOpVectorTimesScalar,
    SpvOpMatrixTimesScalar,
    SpvOpVectorTimesMatrix,
    SpvOpMatrixTimesVector,
    SpvOpMatrixTimesMatrix,
    SpvOpOuterProduct,
    SpvOpDot,
    SpvOpIAddCarry,
    SpvOpISubBorrow,
    SpvOpUMulExtended,
    SpvOpSMulExtended,
    SpvOpAny,
    SpvOpAll,
    SpvOpIsNan,
    SpvOpIsInf,
    SpvOpLogicalEqual,
    SpvOpLogicalNotEqual,
    SpvOpLogicalOr,
    SpvOpLogicalAnd,
    SpvOpLogicalNot,
    SpvOpSelect,
    SpvOpIEqual,
    SpvOpINotEqual,
    SpvOpUGreaterThan,
    SpvOpSGreaterThan,
    SpvOpUGreaterThanEqual,
    SpvOpSGreaterThanEqual,
    SpvOpULessThan,
    SpvOpSLessThan,
    SpvOpULessThanEqual,
    SpvOpSLessThanEqual,
    SpvOpFOrdEqual,
    SpvOpFUnordEqual,
    SpvOpFOrdNotEqual,
    SpvOpFUnordNotEqual,
    SpvOpFOrdLessThan,
    SpvOpFUnordLessThan,
    SpvOpFOrdGreaterThan,
    SpvOpFUnordGreaterThan,
    SpvOpFOrdLessThanEqual,
    SpvOpFUnordLessThanEqual,
    SpvOpFOrdGreaterThanEqual,
    SpvOpFUnordGreaterThanEqual,
    SpvOpShiftRightLogical,
    SpvOpShiftRightArithmetic,
    SpvOpShiftLeftLogical,
    SpvOpBitwiseOr,
    SpvOpBitwiseXor,
    SpvOpBitwiseAnd,
    SpvOpNot,
    SpvOpBitFieldInsert,
    SpvOpBitFieldSExtract,
    SpvOpBitFieldUExtract,
    SpvOpBitReverse,
    SpvOpBitCount,
    SpvOpDPdx,
    SpvOpDPdy,
    SpvOpFwidth,
    SpvOpDPdxFine,
    SpvOpDPdyFine,
    SpvOpFwidthFine,
    SpvOpDPdxCoarse,
    SpvOpDPdyCoarse,
    SpvOpFwidthCoarse,
    SpvOpPhi,
    SpvOpImageSparseSampleImplicitLod,
    SpvOpImageSparseSampleExplicitLod,
    SpvOpImageSparseSampleDrefImplicitLod,
    SpvOpImageSparseSampleDrefExplicitLod,
    SpvOpImageSparseSampleProjImplicitLod,
    SpvOpImageSparseSampleProjExplicitLod,
    SpvOpImageSparseSampleProjDrefImplicitLod,
    SpvOpImageSparseSampleProjDrefExplicitLod,
    SpvOpImageSparseFetch,
    SpvOpImageSparseGather,
    SpvOpImageSparseDrefGather,
    SpvOpImageSparseTexelsResident,
    SpvOpImageSparseRead,
    SpvOpSizeOf
    // TODO(dneto): Add instructions enabled by ImageQuery
  };

  // Find supported extension instruction set ids
  glsl_std_450_id_ = module_->GetExtInstImportId("GLSL.std.450");

  combinator_ops_glsl_std_450_ = {
    GLSLstd450Round,
    GLSLstd450RoundEven,
    GLSLstd450Trunc,
    GLSLstd450FAbs,
    GLSLstd450SAbs,
    GLSLstd450FSign,
    GLSLstd450SSign,
    GLSLstd450Floor,
    GLSLstd450Ceil,
    GLSLstd450Fract,
    GLSLstd450Radians,
    GLSLstd450Degrees,
    GLSLstd450Sin,
    GLSLstd450Cos,
    GLSLstd450Tan,
    GLSLstd450Asin,
    GLSLstd450Acos,
    GLSLstd450Atan,
    GLSLstd450Sinh,
    GLSLstd450Cosh,
    GLSLstd450Tanh,
    GLSLstd450Asinh,
    GLSLstd450Acosh,
    GLSLstd450Atanh,
    GLSLstd450Atan2,
    GLSLstd450Pow,
    GLSLstd450Exp,
    GLSLstd450Log,
    GLSLstd450Exp2,
    GLSLstd450Log2,
    GLSLstd450Sqrt,
    GLSLstd450InverseSqrt,
    GLSLstd450Determinant,
    GLSLstd450MatrixInverse,
    GLSLstd450ModfStruct,
    GLSLstd450FMin,
    GLSLstd450UMin,
    GLSLstd450SMin,
    GLSLstd450FMax,
    GLSLstd450UMax,
    GLSLstd450SMax,
    GLSLstd450FClamp,
    GLSLstd450UClamp,
    GLSLstd450SClamp,
    GLSLstd450FMix,
    GLSLstd450IMix,
    GLSLstd450Step,
    GLSLstd450SmoothStep,
    GLSLstd450Fma,
    GLSLstd450FrexpStruct,
    GLSLstd450Ldexp,
    GLSLstd450PackSnorm4x8,
    GLSLstd450PackUnorm4x8,
    GLSLstd450PackSnorm2x16,
    GLSLstd450PackUnorm2x16,
    GLSLstd450PackHalf2x16,
    GLSLstd450PackDouble2x32,
    GLSLstd450UnpackSnorm2x16,
    GLSLstd450UnpackUnorm2x16,
    GLSLstd450UnpackHalf2x16,
    GLSLstd450UnpackSnorm4x8,
    GLSLstd450UnpackUnorm4x8,
    GLSLstd450UnpackDouble2x32,
    GLSLstd450Length,
    GLSLstd450Distance,
    GLSLstd450Cross,
    GLSLstd450Normalize,
    GLSLstd450FaceForward,
    GLSLstd450Reflect,
    GLSLstd450Refract,
    GLSLstd450FindILsb,
    GLSLstd450FindSMsb,
    GLSLstd450FindUMsb,
    GLSLstd450InterpolateAtCentroid,
    GLSLstd450InterpolateAtSample,
    GLSLstd450InterpolateAtOffset,
    GLSLstd450NMin,
    GLSLstd450NMax,
    GLSLstd450NClamp
  };
}

}  // namespace opt
}  // namespace spvtools

