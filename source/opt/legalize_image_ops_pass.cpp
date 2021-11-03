// Copyright (c) 2021 Tencent Inc.
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

#include "source/opt/legalize_image_ops_pass.h"

#include <algorithm>
#include <vector>

#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {

Pass::Status LegalizeImageOpsPass::Process() {
  Initialize();
  return ProcessImpl();
}

analysis::Type* LegalizeImageOpsPass::FloatScalarType(uint32_t width) {
  analysis::Float float_ty(width);
  return context()->get_type_mgr()->GetRegisteredType(&float_ty);
}

analysis::Type* LegalizeImageOpsPass::FloatVectorType(uint32_t v_len,
                                                      uint32_t width) {
  analysis::Type* reg_float_ty = FloatScalarType(width);
  analysis::Vector vec_ty(reg_float_ty, v_len);
  return context()->get_type_mgr()->GetRegisteredType(&vec_ty);
}

void LegalizeImageOpsPass::SetRelaxed(Instruction* inst) {
  get_decoration_mgr()->AddDecoration(inst->result_id(),
                                      SpvDecorationRelaxedPrecision);
}

bool LegalizeImageOpsPass::IsSameTypeImaged(Instruction* inst_a,
                                            Instruction* inst_b) {
  if (inst_a == inst_b) return false;
  if (inst_a->opcode() != SpvOp::SpvOpTypeImage) return false;
  if (inst_b->opcode() != SpvOp::SpvOpTypeImage) return false;

  auto type_a =
      context()->get_type_mgr()->GetType(inst_a->GetOperand(1).words[0]);
  auto type_b =
      context()->get_type_mgr()->GetType(inst_b->GetOperand(1).words[0]);

  if (!type_a->IsSame(type_b)) return false;

  for (uint32_t i = 2; i < inst_a->NumOperands(); i++) {
    if (inst_a->GetOperand(i) != inst_b->GetOperand(i)) {
      return false;
    }
  }
  return true;
}

uint32_t LegalizeImageOpsPass::IndexOfType(Instruction* inst) {
  uint32_t index = 0;
  for (auto& cur_inst : context()->types_values()) {
    if (*inst == cur_inst) {
      break;
    }
    index++;
  }
  return index;
}

void LegalizeImageOpsPass::MoveType(Instruction* a,
                                     Module::inst_iterator& insert_point) {
  auto a_it = std::find(context()->types_values_begin(),
                          context()->types_values_end(), *a);
  if (a_it != context()->types_values_end()) {
    insert_point.InsertBefore(std::make_unique<Instruction>(*a));
    a_it.Erase();
  }
}

bool LegalizeImageOpsPass::ConvertOpTypeImage(Instruction* inst) {
  // OpTypeImage (Result, Sampled Type, Dim, Depth, ... )
  uint32_t sampled_ty_id = inst->GetOperand(1).words[0];
  auto sampled_ty = context()->get_type_mgr()->GetType(sampled_ty_id);
  if (sampled_ty->kind() == analysis::Type::Kind::kFloat) {
    if (static_cast<analysis::Float*>(sampled_ty)->width() == 32) return false;
  } else {
    return false;
  }

  auto fp32_ty_id =
      context()->get_type_mgr()->GetTypeInstruction(FloatScalarType(32));
  auto f32_ty = context()->get_def_use_mgr()->GetDef(fp32_ty_id);
  // Covers the case when %float 32 is not defined, 
  // which is rare in real life condictions.
  if (IndexOfType(f32_ty) > IndexOfType(inst)) {
    // OpFloatType is created after OpTypeImage
    // Which causes 'Operand 13[%float] requires a previous definition'.
    // So we need to move OpFloatType to the top of type definitions.
    MoveType(f32_ty, context()->types_values_begin());
  }

  inst->SetOperand(1, {fp32_ty_id});

  SetRelaxed(inst);
  get_def_use_mgr()->AnalyzeInstUse(inst);
  return true;
}

bool LegalizeImageOpsPass::ConvertImageOps(BasicBlock* bb, Instruction* inst) {
  if (inst->opcode() == SpvOpImageWrite) {
    // OpImageWrite (Image, Coordinate, Texel, Image Operands ...)

    auto load_id = inst->GetOperand(0).words[0];
    auto load_inst = get_def_use_mgr()->GetDef(load_id);
    auto load_result_id = load_inst->result_id();

    auto write_id = inst->GetOperand(2).words[0];
    auto write_inst = get_def_use_mgr()->GetDef(write_id);

    auto write_type_id = write_inst->type_id();
    auto write_type = context()->get_type_mgr()->GetType(write_type_id);

    if (write_type->kind() != analysis::Type::Kind::kVector) return false;

    auto vector_ty = static_cast<analysis::Vector*>(write_type);
    auto vector_ele_ty = vector_ty->element_type();

    if (vector_ele_ty->kind() != analysis::Type::Kind::kFloat ||
        static_cast<const analysis::Float*>(vector_ele_ty)->width() != 16) {
      return false;
    }

    auto f32_type_id = context()->get_type_mgr()->GetTypeInstruction(
        FloatVectorType(vector_ty->element_count(), 32));

    // For SpvOpImageWrite:
    // Convert From:
    // %387 = OpCompositeConstruct %v4half %384 %385 %386 %float_1
    // %388 = OpLoad %type_2d_image_0 %OutputTexture
    //        OpImageWrite % 388 % 240 % 387 None
    // To:
    // %387 = OpCompositeConstruct %v4half %384 %385 %386 %float_1 [0]
    // %999 = OpFConvert %v4float %387                             [1]
    // %388 = OpLoad %type_2d_image_0 %OutputTexture               [2]
    //        OpImageWrite %388 %240 %999 None                     [3]

    Instruction* inst0 = write_inst;
    Instruction* inst1 = nullptr;
    Instruction* inst2 = load_inst;
    Instruction* inst3 = inst;
    {
      // %999 = OpFConvert %v4float %387                             [1]
      InstructionBuilder builder(
          context(), inst2,
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
      inst1 =
          builder.AddUnaryOp(f32_type_id, SpvOpFConvert, inst0->result_id());
    }
    
    //        OpImageWrite %388 %240 %999 None                     [3]
    inst3->SetOperand(2, {inst1->result_id()});

    SetRelaxed(inst1);

    return true;
  } else if (image_ops_.count(inst->opcode()) > 0) {
    // Targets OpImageXXX starts with (Result Type, Result Id ...)

    auto result_ty_id = inst->GetOperand(0).words[0];
    auto result_ty = context()->get_type_mgr()->GetType(result_ty_id);

    assert(result_ty->kind() == analysis::Type::Kind::kVector &&
           "Result Type must be a vector of floating-point or integer type.");

    auto vector_ty = static_cast<analysis::Vector*>(result_ty);
    auto vector_ele_ty = vector_ty->element_type();

    if (vector_ele_ty->kind() != analysis::Type::Kind::kFloat ||
        static_cast<const analysis::Float*>(vector_ele_ty)->width() != 16) {
      return false;
    }

    auto f16_type_id = result_ty_id;
    auto f32_type_id = context()->get_type_mgr()->GetTypeInstruction(
        FloatVectorType(vector_ty->element_count(), 32));

    auto result_id = inst->result_id();

    Instruction* cvt_inst = nullptr;
    {
      auto anlysisdefs =
          IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping;
      // OpFConvert (Result Type, Result Id, Float Value)
      if (auto next_inst = inst->NextNode()) {
        InstructionBuilder builder(context(), next_inst, anlysisdefs);
        cvt_inst = builder.AddUnaryOp(f16_type_id, SpvOpFConvert, result_id);
      } else {
        InstructionBuilder builder(context(), bb, anlysisdefs);
        cvt_inst = builder.AddUnaryOp(f16_type_id, SpvOpFConvert, result_id);
      }
    }

    context()->ReplaceAllUsesWith(result_id, cvt_inst->result_id());

    inst->SetResultType(f32_type_id);
    cvt_inst->SetOperand(2, {result_id});

    SetRelaxed(inst);
    SetRelaxed(cvt_inst);

    return true;
  }

  return false;
}

bool LegalizeImageOpsPass::ProcessFunction(Function* func) {
  bool modified = false;
  cfg()->ForEachBlockInReversePostOrder(
      func->entry().get(), [&modified, this](BasicBlock* bb) {
        for (auto ii = bb->begin(); ii != bb->end(); ++ii)
          modified |= ConvertImageOps(bb, &*ii);
      });
  return modified;
}

Pass::Status LegalizeImageOpsPass::ProcessImpl() {
  bool modified = false;

  // Collect all OpTypeImages.
  std::vector<Instruction*> op_type_images;
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() == SpvOp::SpvOpTypeImage) {
      op_type_images.push_back(&inst);
    }
  }

  // Convert F16 OpTypeImage to F32.
  for (auto inst : op_type_images) {
    if (ConvertOpTypeImage(inst)) {
      modified = true;
    }
  }

  // Remove duplicated OpTypeImage declarations.
  if (modified) {
    for (auto& inst_a : op_type_images) {
      if (!inst_a) {
        continue;
      }
      for (auto& inst_b : op_type_images) {
        if (!inst_b) {
          continue;
        }
        if (IsSameTypeImaged(inst_a, inst_b)) {
          context()->ReplaceAllUsesWith(inst_b->result_id(),
                                        inst_a->result_id());
          context()->KillInst(inst_b);
          inst_b = nullptr;
        }
      }
    }
  }

  // Convert F16 OpImageFetch 
  if (modified) {
    Pass::ProcessFunction pfn = [this](Function* fp) {
      return ProcessFunction(fp);
    };

    if (context()->ProcessEntryPointCallTree(pfn)) modified = true;
  }

  auto typesa = std::vector<Instruction*>();
  for (auto& cur_inst : context()->types_values()) {
    typesa.push_back(&cur_inst);
  }

  auto types = context()->module()->GetTypes();

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

void LegalizeImageOpsPass::Initialize() {
  image_ops_ = {SpvOpImageSampleImplicitLod,
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
                SpvOpImageSparseRead};
}

}  // namespace opt
}  // namespace spvtools