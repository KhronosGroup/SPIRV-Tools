// Copyright (c) 2026 Lee Gao
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

#include "source/opt/mali_bitwise_scalar_pass.h"

#include <cstdio>
#include <vector>

#include "source/opt/ir_builder.h"

#define LOG_(level, fmt, ...)                                                  \
    do {                                                                       \
        char buffer[256];                                                      \
        snprintf(buffer, sizeof(buffer), "MaliBitwiseScalarPass: " fmt,        \
                 ##__VA_ARGS__);                                               \
        consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer);             \
    } while (0)

#define LOGD(fmt, ...) LOG_(SPV_MSG_DEBUG, fmt, ##__VA_ARGS__)
#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ##__VA_ARGS__)

namespace spvtools {
namespace opt {

namespace {

bool IsTargetOp(spv::Op op) {
    switch (op) {
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
        return true;
    default:
        return false;
    }
}

uint32_t SourceCount(spv::Op op) { return op == spv::Op::OpNot ? 1u : 2u; }

} // namespace

bool MaliBitwiseScalarPass::IsPatternOperand(uint32_t id) {
    // TODO(leegao): replace with a dataflow analysis
    for (int depth = 0; depth < 8 && id != 0; ++depth) {
        Instruction *def = get_def_use_mgr()->GetDef(id);
        if (def == nullptr)
            return false;
        switch (def->opcode()) {
        case spv::Op::OpVectorShuffle:
        case spv::Op::OpCompositeConstruct:
        case spv::Op::OpCompositeInsert:
        case spv::Op::OpConstantComposite:
        case spv::Op::OpSpecConstantComposite:
            return true;
        case spv::Op::OpSpecConstantOp: {
            const uint32_t op = def->GetSingleWordInOperand(0);
            return op == static_cast<uint32_t>(spv::Op::OpVectorShuffle) ||
                   op == static_cast<uint32_t>(spv::Op::OpCompositeInsert);
        }
        case spv::Op::OpCopyObject:
        case spv::Op::OpBitcast:
            id = def->GetSingleWordInOperand(0);
            break;
        default:
            return false;
        }
    }
    return false;
}

bool MaliBitwiseScalarPass::Scalarize(Instruction *inst) {
    const spv::Op op = inst->opcode();
    const uint32_t ns = SourceCount(op);
    if (inst->NumInOperands() < ns)
        return false;

    analysis::Type *type = context()->get_type_mgr()->GetType(inst->type_id());
    if (type == nullptr)
        return false;
    analysis::Vector *vec = type->AsVector();
    if (vec == nullptr)
        return false;
    const analysis::Integer *component = vec->element_type()->AsInteger();
    if (component == nullptr)
        return false;
    const uint32_t lanes = vec->element_count();
    if (lanes == 0 || lanes > 16)
        return false;

    const uint32_t component_type_id =
        context()->get_type_mgr()->GetId(vec->element_type());
    if (component_type_id == 0)
        return false;

    // Every operand must be an integer vector of the result's size and
    // component width
    std::vector<uint32_t> sources;
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t src = inst->GetSingleWordInOperand(s);
        Instruction *src_inst = get_def_use_mgr()->GetDef(src);
        if (src_inst == nullptr)
            return false;
        analysis::Type *src_type =
            context()->get_type_mgr()->GetType(src_inst->type_id());
        if (src_type == nullptr || src_type->AsVector() == nullptr ||
            src_type->AsVector()->element_count() != lanes) {
            return false;
        }
        const analysis::Integer *src_component =
            src_type->AsVector()->element_type()->AsInteger();
        if (src_component == nullptr ||
            src_component->width() != component->width()) {
            return false;
        }
        sources.push_back(src);
    }

    InstructionBuilder builder(context(), inst,
                               IRContext::kAnalysisDefUse |
                                   IRContext::kAnalysisInstrToBlockMapping);

    std::vector<uint32_t> results;
    results.reserve(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        std::vector<uint32_t> lane_ids;
        for (uint32_t s = 0; s < ns; ++s) {
            Instruction *extract = builder.AddCompositeExtract(
                component_type_id, sources[s], {lane});
            if (extract == nullptr)
                return false;
            lane_ids.push_back(extract->result_id());
        }
        Instruction *scalar = nullptr;
        if (op == spv::Op::OpNot) {
            scalar = builder.AddUnaryOp(component_type_id, op, lane_ids[0]);
        } else {
            scalar = builder.AddBinaryOp(component_type_id, op, lane_ids[0],
                                         lane_ids[1]);
        }
        if (scalar == nullptr)
            return false;
        results.push_back(scalar->result_id());
    }

    Instruction::OperandList operands;
    operands.reserve(results.size());
    for (uint32_t id : results) {
        operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
    }
    inst->SetOpcode(spv::Op::OpCompositeConstruct);
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
    return true;
}

Pass::Status MaliBitwiseScalarPass::Process() {
    uint32_t rewrites = 0;
    for (Function &func : *context()->module()) {
        if (func.IsDeclaration())
            continue;
        for (BasicBlock &block : func) {
            for (Instruction &inst : block) {
                if (!IsTargetOp(inst.opcode()))
                    continue;

                const uint32_t ns = SourceCount(inst.opcode());
                if (inst.NumInOperands() < ns)
                    continue;

                bool pattern = rewrite_all_;
                for (uint32_t s = 0; s < ns && !pattern; ++s) {
                    pattern = IsPatternOperand(inst.GetSingleWordInOperand(s));
                }
                if (!pattern)
                    continue;

                if (Scalarize(&inst)) {
                    ++rewrites;
                }
            }
        }
    }

    if (rewrites != 0) {
        LOG("%u integer vector bitwise op(s) scalarised", rewrites);
    }
    return rewrites != 0 ? Status::SuccessWithChange
                         : Status::SuccessWithoutChange;
}

} // namespace opt
} // namespace spvtools
