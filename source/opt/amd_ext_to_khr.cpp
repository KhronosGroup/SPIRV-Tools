// Copyright (c) 2019 Google LLC.
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

#include "source/opt/amd_ext_to_khr.h"

#include "ir_builder.h"
#include "source/opt/ir_context.h"
#include "spv-amd-shader-ballot.insts.inc"
#include "type_manager.h"

namespace spvtools {
namespace opt {

namespace {

enum ExtOpcodes {
  AmdShaderBallotSwizzleInvocationsAMD = 1,
  AmdShaderBallotSwizzleInvocationsMaskedAMD = 2,
  AmdShaderBallotWriteInvocationAMD = 3,
  AmdShaderBallotMbcntAMD = 4
};

analysis::Type* GetUIntType(IRContext* ctx) {
  analysis::Integer int_type(32, false);
  return ctx->get_type_mgr()->GetRegisteredType(&int_type);
}

// Returns a folding rule that will replace the opcode with |opcode| and add
// the capabilities required.  The folding rule assumes it is folding an
// OpGroup*NonUniformAMD instruction from the SPV_AMD_shader_ballot extension.
FoldingRule ReplaceGroupNonuniformOperationOpCode(SpvOp new_opcode) {
  switch (new_opcode) {
    case SpvOpGroupNonUniformIAdd:
    case SpvOpGroupNonUniformFAdd:
    case SpvOpGroupNonUniformUMin:
    case SpvOpGroupNonUniformSMin:
    case SpvOpGroupNonUniformFMin:
    case SpvOpGroupNonUniformUMax:
    case SpvOpGroupNonUniformSMax:
    case SpvOpGroupNonUniformFMax:
      break;
    default:
      assert(false &&
             "Should replacing with a group non uniform arithmetic operation.");
  }

  return [new_opcode](IRContext* ctx, Instruction* inst,
                      const std::vector<const analysis::Constant*>&) {
    switch (inst->opcode()) {
      case SpvOpGroupIAddNonUniformAMD:
      case SpvOpGroupFAddNonUniformAMD:
      case SpvOpGroupUMinNonUniformAMD:
      case SpvOpGroupSMinNonUniformAMD:
      case SpvOpGroupFMinNonUniformAMD:
      case SpvOpGroupUMaxNonUniformAMD:
      case SpvOpGroupSMaxNonUniformAMD:
      case SpvOpGroupFMaxNonUniformAMD:
        break;
      default:
        assert(false &&
               "Should replacing a group non uniform arithmetic operation.");
    }

    ctx->AddCapability(SpvCapabilityGroupNonUniformArithmetic);
    inst->SetOpcode(new_opcode);
    return true;
  };
}

FoldingRule NotImplementedYet() {
  return [](IRContext*, Instruction*,
            const std::vector<const analysis::Constant*>&) {
    assert(false && "Replacement not implemented yet.");
    return false;
  };
}

// Returns a folding rule that will replace the WriteInvocationAMD extended
// instruction in the SPV_AMD_shader_ballot extension.
//
// The instruction
//
// clang-format off
//    %result = OpExtInst %type %1 WriteInvocationAMD %input_value %write_value %invocation_index
// clang-format on
//
// with
//
//     %id = OpLoad %uint %SubgroupLocalInvocationId
//    %cmp = OpIEqual %bool %id %invocation_index
// %result = OpSelect %type %cmp %write_value %input_value
//
// Also adding the capabilities and builtins that are needed.
FoldingRule ReplaceWriteInvocation() {
  return [](IRContext* ctx, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    uint32_t var_id =
        ctx->GetBuiltinInputVarId(SpvBuiltInSubgroupLocalInvocationId);
    ctx->AddCapability(SpvCapabilitySubgroupBallotKHR);
    ctx->AddExtension("SPV_KHR_shader_ballot");
    assert(var_id != 0 && "Could not get SubgroupLocalInvocationId variable.");
    Instruction* var_inst = ctx->get_def_use_mgr()->GetDef(var_id);
    Instruction* var_ptr_type =
        ctx->get_def_use_mgr()->GetDef(var_inst->type_id());

    InstructionBuilder ir_builder(
        ctx, inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* t =
        ir_builder.AddLoad(var_ptr_type->GetSingleWordInOperand(1), var_id);
    analysis::Bool bool_type;
    uint32_t bool_type_id = ctx->get_type_mgr()->GetTypeInstruction(&bool_type);
    Instruction* cmp =
        ir_builder.AddBinaryOp(bool_type_id, SpvOpIEqual, t->result_id(),
                               inst->GetSingleWordInOperand(4));

    // Build a select.
    inst->SetOpcode(SpvOpSelect);
    Instruction::OperandList new_operands;
    new_operands.push_back({SPV_OPERAND_TYPE_ID, {cmp->result_id()}});
    new_operands.push_back(inst->GetInOperand(3));
    new_operands.push_back(inst->GetInOperand(2));

    inst->SetInOperands(std::move(new_operands));
    ctx->UpdateDefUse(inst);
    return true;
  };
}

// Returns a folding rule that will replace the MbcntAMD extended instruction in
// the SPV_AMD_shader_ballot extension.
//
// The instruction
//
//  %result = OpExtInst %uint %1 MbcntAMD %mask
//
// with
//
// Get SubgroupLtMask and convert the first 64-bits into a uint64_t because
// AMD's shader compiler expects a 64-bit integer mask.
//
//     %var = OpLoad %v4uint %SubgroupLtMaskKHR
// %shuffle = OpVectorShuffle %v2uint %var %var 0 1
//    %cast = OpBitcast %ulong %shuffle
//
// Perform the mask and count the bits.
//
//     %and = OpBitwiseAnd %ulong %cast %mask
//  %result = OpBitCount %uint %and
//
// Also adding the capabilities and builtins that are needed.
FoldingRule ReplaceMbcnt() {
  return [](IRContext* context, Instruction* inst,
            const std::vector<const analysis::Constant*>&) {
    analysis::TypeManager* type_mgr = context->get_type_mgr();
    analysis::DefUseManager* def_use_mgr = context->get_def_use_mgr();

    uint32_t var_id = context->GetBuiltinInputVarId(SpvBuiltInSubgroupLtMask);
    assert(var_id != 0 && "Could not get SubgroupLtMask variable.");
    context->AddCapability(SpvCapabilityGroupNonUniformBallot);
    Instruction* var_inst = def_use_mgr->GetDef(var_id);
    Instruction* var_ptr_type = def_use_mgr->GetDef(var_inst->type_id());
    Instruction* var_type =
        def_use_mgr->GetDef(var_ptr_type->GetSingleWordInOperand(1));
    assert(var_type->opcode() == SpvOpTypeVector &&
           "Variable is suppose to be a vector of 4 ints");

    // Get the type for the shuffle.
    analysis::Vector temp_type(GetUIntType(context), 2);
    const analysis::Type* shuffle_type =
        context->get_type_mgr()->GetRegisteredType(&temp_type);
    uint32_t shuffle_type_id = type_mgr->GetTypeInstruction(shuffle_type);

    uint32_t mask_id = inst->GetSingleWordInOperand(2);
    Instruction* mask_inst = def_use_mgr->GetDef(mask_id);

    // Testing with amd's shader compiler shows that a 64-bit mask is expected.
    assert(type_mgr->GetType(mask_inst->type_id())->AsInteger() != nullptr);
    assert(type_mgr->GetType(mask_inst->type_id())->AsInteger()->width() == 64);

    InstructionBuilder ir_builder(
        context, inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* load = ir_builder.AddLoad(var_type->result_id(), var_id);
    Instruction* shuffle = ir_builder.AddVectorShuffle(
        shuffle_type_id, load->result_id(), load->result_id(), {0, 1});
    Instruction* bitcast = ir_builder.AddUnaryOp(
        mask_inst->type_id(), SpvOpBitcast, shuffle->result_id());
    Instruction* t = ir_builder.AddBinaryOp(
        mask_inst->type_id(), SpvOpBitwiseAnd, bitcast->result_id(), mask_id);

    inst->SetOpcode(SpvOpBitCount);
    inst->SetInOperands({{SPV_OPERAND_TYPE_ID, {t->result_id()}}});
    context->UpdateDefUse(inst);
    return true;
  };
}

class AmdExtFoldingRules : public FoldingRules {
 public:
  explicit AmdExtFoldingRules(IRContext* ctx) : FoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {
    rules_[SpvOpGroupIAddNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformIAdd));
    rules_[SpvOpGroupFAddNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformFAdd));
    rules_[SpvOpGroupUMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformUMin));
    rules_[SpvOpGroupSMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformSMin));
    rules_[SpvOpGroupFMinNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformFMin));
    rules_[SpvOpGroupUMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformUMax));
    rules_[SpvOpGroupSMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformSMax));
    rules_[SpvOpGroupFMaxNonUniformAMD].push_back(
        ReplaceGroupNonuniformOperationOpCode(SpvOpGroupNonUniformFMax));

    uint32_t extension_id =
        context()->module()->GetExtInstImportId("SPV_AMD_shader_ballot");

    ext_rules_[{extension_id, AmdShaderBallotSwizzleInvocationsAMD}].push_back(
        NotImplementedYet());
    ext_rules_[{extension_id, AmdShaderBallotSwizzleInvocationsMaskedAMD}]
        .push_back(NotImplementedYet());
    ext_rules_[{extension_id, AmdShaderBallotWriteInvocationAMD}].push_back(
        ReplaceWriteInvocation());
    ext_rules_[{extension_id, AmdShaderBallotMbcntAMD}].push_back(
        ReplaceMbcnt());
  }
};

class AmdExtConstFoldingRules : public ConstantFoldingRules {
 public:
  AmdExtConstFoldingRules(IRContext* ctx) : ConstantFoldingRules(ctx) {}

 protected:
  virtual void AddFoldingRules() override {}
};

}  // namespace

Pass::Status AmdExtensionToKhrPass::Process() {
  bool changed = false;

  // Traverse the body of the functions to replace instructions that require
  // the extensions.
  InstructionFolder folder(
      context(),
      std::unique_ptr<AmdExtFoldingRules>(new AmdExtFoldingRules(context())),
      MakeUnique<AmdExtConstFoldingRules>(context()));
  for (Function& func : *get_module()) {
    func.ForEachInst([&changed, &folder](Instruction* inst) {
      if (folder.FoldInstruction(inst)) {
        changed = true;
      }
    });
  }

  // Now that instruction that require the extensions have been removed, we can
  // remove the extension instructions.
  std::vector<Instruction*> to_be_killed;
  for (Instruction& inst : context()->module()->extensions()) {
    if (inst.opcode() == SpvOpExtension) {
      if (!strcmp("SPV_AMD_shader_ballot",
                  reinterpret_cast<const char*>(
                      &(inst.GetInOperand(0).words[0])))) {
        to_be_killed.push_back(&inst);
      }
    }
  }

  for (Instruction& inst : context()->ext_inst_imports()) {
    if (inst.opcode() == SpvOpExtInstImport) {
      if (!strcmp("SPV_AMD_shader_ballot",
                  reinterpret_cast<const char*>(
                      &(inst.GetInOperand(0).words[0])))) {
        to_be_killed.push_back(&inst);
      }
    }
  }

  for (Instruction* inst : to_be_killed) {
    context()->KillInst(inst);
    changed = true;
  }

  // The replacements that take place use instructions that are missing before
  // SPIR-V 1.3. If we changed something, we will have to make sure the version
  // is at least SPIR-V 1.3 to make sure those instruction can be used.
  if (changed) {
    uint32_t version = get_module()->version();
    if (version < 0x00010300 /*1.3*/) {
      get_module()->set_version(0x00010300);
    }
  }
  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
