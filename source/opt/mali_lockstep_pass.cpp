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

#include "source/opt/mali_lockstep_pass.h"

#include <cstdio>
#include <vector>

#include "source/opt/ir_builder.h"

#define LOG_(level, fmt, ...)                                                  \
    do {                                                                       \
        char buffer[256];                                                      \
        snprintf(buffer, sizeof(buffer), "MaliLockstepPass: " fmt,             \
                 ##__VA_ARGS__);                                               \
        consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer);             \
    } while (0)

#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ##__VA_ARGS__)

namespace spvtools {
namespace opt {

namespace {

bool IsExitOp(spv::Op op) {
    return op == spv::Op::OpReturn || op == spv::Op::OpReturnValue ||
           op == spv::Op::OpKill || op == spv::Op::OpUnreachable ||
           op == spv::Op::OpTerminateInvocation;
}

bool IsAtomicAccessOp(spv::Op op) {
    switch (op) {
    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicStore:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
    case spv::Op::OpAtomicFlagTestAndSet:
    case spv::Op::OpAtomicFlagClear:
        return true;
    default:
        return false;
    }
}

} // namespace

Pass::Status MaliLockstepPass::Process() {
    uint32_t entry_func = 0;
    int entry_count = 0;
    for (auto &entry : context()->module()->entry_points()) {
        if (entry.GetSingleWordInOperand(0) ==
            static_cast<uint32_t>(spv::ExecutionModel::GLCompute)) {
            entry_func = entry.GetSingleWordInOperand(1);
            ++entry_count;
        }
    }
    if (entry_count != 1)
        return Status::SuccessWithoutChange;

    uint32_t local_size_x = 0;
    for (auto &mode : context()->module()->execution_modes()) {
        if (mode.GetSingleWordInOperand(0) == entry_func &&
            mode.GetSingleWordInOperand(1) ==
                static_cast<uint32_t>(spv::ExecutionMode::LocalSize)) {
            local_size_x = mode.GetSingleWordInOperand(2);
        }
    }
    if (local_size_x < 2)
        return Status::SuccessWithoutChange;

    Function *func = context()->GetFunction(entry_func);
    if (func == nullptr)
        return Status::SuccessWithoutChange;

    auto is_workgroup_pointer = [this](uint32_t id) {
        Instruction *def = get_def_use_mgr()->GetDef(id);
        if (def == nullptr)
            return false;
        analysis::Type *type =
            context()->get_type_mgr()->GetType(def->type_id());
        analysis::Pointer *pointer =
            type != nullptr ? type->AsPointer() : nullptr;
        return pointer != nullptr &&
               pointer->storage_class() == spv::StorageClass::Workgroup;
    };

    auto is_workgroup_access = [&](const Instruction &inst) {
        switch (inst.opcode()) {
        case spv::Op::OpLoad:
        case spv::Op::OpAtomicLoad:
        case spv::Op::OpStore:
        case spv::Op::OpAtomicStore:
        case spv::Op::OpAtomicFlagClear:
            return inst.NumInOperands() >= 1 &&
                   is_workgroup_pointer(inst.GetSingleWordInOperand(0));
        case spv::Op::OpCopyMemory:
        case spv::Op::OpCopyMemorySized:
            return (inst.NumInOperands() >= 1 &&
                    is_workgroup_pointer(inst.GetSingleWordInOperand(0))) ||
                   (inst.NumInOperands() >= 2 &&
                    is_workgroup_pointer(inst.GetSingleWordInOperand(1)));
        default:
            return IsAtomicAccessOp(inst.opcode()) &&
                   inst.NumInOperands() >= 1 &&
                   is_workgroup_pointer(inst.GetSingleWordInOperand(0));
        }
    };

    // LocalInvocationId.x: a load of the built-in variable, then a composite
    // extract of lane 0.
    uint32_t local_invocation_var = 0;
    for (Instruction &inst : context()->module()->types_values()) {
        if (inst.opcode() != spv::Op::OpVariable)
            continue;
        for (Instruction *dec :
             context()->get_decoration_mgr()->GetDecorationsFor(
                 inst.result_id(), false)) {
            if (dec->opcode() == spv::Op::OpDecorate &&
                dec->GetSingleWordInOperand(1) ==
                    static_cast<uint32_t>(spv::Decoration::BuiltIn) &&
                dec->GetSingleWordInOperand(2) ==
                    static_cast<uint32_t>(spv::BuiltIn::LocalInvocationId)) {
                local_invocation_var = inst.result_id();
            }
        }
    }
    if (local_invocation_var == 0)
        return Status::SuccessWithoutChange;

    uint32_t thread_id = 0;
    for (BasicBlock &block : *func) {
        for (Instruction &inst : block) {
            if (inst.opcode() != spv::Op::OpCompositeExtract)
                continue;
            if (inst.NumInOperands() < 2)
                continue;
            if (inst.GetSingleWordInOperand(1) != 0)
                continue;
            Instruction *composite =
                get_def_use_mgr()->GetDef(inst.GetSingleWordInOperand(0));
            if (composite == nullptr ||
                composite->opcode() != spv::Op::OpLoad) {
                continue;
            }
            if (composite->NumInOperands() < 1)
                continue;
            if (composite->GetSingleWordInOperand(0) != local_invocation_var)
                continue;
            thread_id = inst.result_id();
            break;
        }
        if (thread_id != 0)
            break;
    }
    if (thread_id == 0)
        return Status::SuccessWithoutChange;

    bool changed = false;
    for (BasicBlock &header : *func) {
        Instruction *merge = header.GetLoopMergeInst();
        if (merge == nullptr)
            continue;
        const uint32_t merge_id = merge->GetSingleWordInOperand(0);
        BasicBlock *merge_block = context()->cfg()->block(merge_id);
        if (merge_block == nullptr)
            continue;

        // A block belongs to this loop when the header dominates it and the
        // merge block does not.
        DominatorAnalysis *dom = context()->GetDominatorAnalysis(func);
        auto in_loop = [&](BasicBlock *block) {
            return block != nullptr && block != merge_block &&
                   dom->Dominates(&header, block) &&
                   !dom->Dominates(merge_block, block);
        };

        // The induction phi: i = OpPhi(tid, pre; i + W, back).
        Instruction *phi = nullptr;
        Instruction *add = nullptr;
        uint32_t pre_block = 0;
        uint32_t back_block = 0;
        uint32_t w_const_id = 0;
        for (Instruction &inst : header) {
            if (inst.opcode() != spv::Op::OpPhi)
                continue;
            // A structured loop header has exactly two predecessors (the
            // preheader and the back edge)
            if (inst.NumInOperands() != 4)
                continue;
            for (uint32_t s = 0; s + 1 < inst.NumInOperands(); s += 2) {
                const uint32_t value = inst.GetSingleWordInOperand(s);
                const uint32_t block = inst.GetSingleWordInOperand(s + 1);
                if (value == thread_id) {
                    pre_block = block;
                    continue;
                }
                // 64-bit (or otherwise widened) inductions start from a
                // conversion of LocalInvocationId.x.
                Instruction *value_def = get_def_use_mgr()->GetDef(value);
                if (value_def != nullptr &&
                    (value_def->opcode() == spv::Op::OpUConvert ||
                     value_def->opcode() == spv::Op::OpSConvert ||
                     value_def->opcode() == spv::Op::OpBitcast) &&
                    value_def->NumInOperands() >= 1 &&
                    value_def->GetSingleWordInOperand(0) == thread_id) {
                    pre_block = block;
                    continue;
                }
                Instruction *def = get_def_use_mgr()->GetDef(value);
                if (def == nullptr || def->opcode() != spv::Op::OpIAdd ||
                    def->NumInOperands() != 2) {
                    continue;
                }
                const uint32_t a = def->GetSingleWordInOperand(0);
                const uint32_t b = def->GetSingleWordInOperand(1);
                uint32_t other = 0;
                if (a == inst.result_id()) {
                    other = b;
                } else if (b == inst.result_id()) {
                    other = a;
                } else {
                    // Not the loop-carried increment
                    continue;
                }
                Instruction *other_def = get_def_use_mgr()->GetDef(other);
                if (other_def == nullptr ||
                    other_def->opcode() != spv::Op::OpConstant) {
                    continue;
                }
                // 64-bit constants have two words
                const analysis::Constant *other_const =
                    context()->get_constant_mgr()->FindDeclaredConstant(other);
                if (other_const == nullptr ||
                    other_const->type()->AsInteger() == nullptr ||
                    other_const->GetZeroExtendedValue() != local_size_x) {
                    continue;
                }
                phi = &inst;
                add = def;
                back_block = block;
                w_const_id = other;
            }
            if (phi != nullptr)
                break;
        }
        if (phi == nullptr || add == nullptr || pre_block == 0 ||
            back_block == 0) {
            continue;
        }
        // Inserting the new increment next to `add` and making it read the new
        // header phi is only valid when `add` is itself in the loop.
        BasicBlock *add_block = context()->get_instr_block(add);
        if (!in_loop(add_block))
            continue;

        // The loop's body: its Workgroup accesses and its one way out.
        bool bad = false;
        bool has_workgroup_access = false;
        uint32_t exit_block = 0;
        Instruction *test = nullptr;
        // First find the block that is only the way out.
        for (BasicBlock &block : *func) {
            if (!in_loop(&block))
                continue;
            for (Instruction &inst : block) {
                if (is_workgroup_access(inst))
                    has_workgroup_access = true;
                if (IsExitOp(inst.opcode()))
                    bad = true;
                if (inst.opcode() == spv::Op::OpBranch &&
                    inst.GetSingleWordInOperand(0) == merge_id) {
                    if (&inst != block.terminator() || exit_block != 0)
                        bad = true;
                    exit_block = block.id();
                }
            }
        }
        // Then the conditional branch that leads there.
        if (exit_block != 0) {
            for (BasicBlock &block : *func) {
                if (!in_loop(&block))
                    continue;
                for (Instruction &inst : block) {
                    if (inst.opcode() == spv::Op::OpBranch &&
                        inst.GetSingleWordInOperand(0) == exit_block)
                        bad = true;
                    if (inst.opcode() != spv::Op::OpBranchConditional ||
                        inst.NumInOperands() < 3)
                        continue;
                    const uint32_t t0 = inst.GetSingleWordInOperand(1);
                    const uint32_t t1 = inst.GetSingleWordInOperand(2);
                    if (t0 == merge_id || t1 == merge_id)
                        bad = true;
                    if (t0 == exit_block || t1 == exit_block)
                        test = &inst;
                }
            }
        }
        if (bad || !has_workgroup_access || exit_block == 0 ||
            test == nullptr) {
            continue;
        }
        if (test->GetSingleWordInOperand(1) != exit_block)
            continue;

        // The test: OpULessThanEqual(n, phi) or OpUGreaterThanEqual(phi, n).
        Instruction *compare =
            get_def_use_mgr()->GetDef(test->GetSingleWordInOperand(0));
        if (compare == nullptr || compare->NumInOperands() != 2)
            continue;
        uint32_t n = 0;
        bool phi_is_first = false;
        if (compare->opcode() == spv::Op::OpULessThanEqual &&
            compare->GetSingleWordInOperand(1) == phi->result_id()) {
            n = compare->GetSingleWordInOperand(0);
            phi_is_first = false;
        } else if (compare->opcode() == spv::Op::OpUGreaterThanEqual &&
                   compare->GetSingleWordInOperand(0) == phi->result_id()) {
            n = compare->GetSingleWordInOperand(1);
            phi_is_first = true;
        } else {
            continue;
        }
        Instruction *n_def = get_def_use_mgr()->GetDef(n);
        if (n_def == nullptr)
            continue;
        if (n_def->opcode() != spv::Op::OpConstant &&
            n_def->opcode() != spv::Op::OpSpecConstant) {
            // n must be defined in a block that dominates the loop header.
            BasicBlock *n_block = context()->get_instr_block(n_def);
            if (n_block == nullptr || !dom->Dominates(n_block, &header))
                continue;
        }

        // The new phi and its increment.
        const uint32_t phi_type_id = phi->type_id();
        analysis::Type *phi_type =
            context()->get_type_mgr()->GetType(phi_type_id);
        analysis::Integer *phi_int =
            phi_type != nullptr ? phi_type->AsInteger() : nullptr;
        if (phi_int == nullptr)
            continue;
        // GetUIntConstId is always a 32-bit uint; the phi may not be.
        std::vector<uint32_t> zero_words((phi_int->width() + 31) / 32, 0);
        const analysis::Constant *zero_const =
            context()->get_constant_mgr()->GetConstant(phi_type, zero_words);
        Instruction *zero_def =
            zero_const != nullptr
                ? context()->get_constant_mgr()->GetDefiningInstruction(
                      zero_const, phi_type_id)
                : nullptr;
        if (zero_def == nullptr)
            continue;
        const uint32_t zero_id = zero_def->result_id();
        const uint32_t new_phi_id = context()->TakeNextId();
        if (new_phi_id == 0)
            return Status::Failure;

        // Insert the increment first, using the old phi as a placeholder
        // operand so the def-use manager can resolve it. The new phi is
        // inserted next, then the increment is re-pointed at it.
        InstructionBuilder add_builder(
            context(), add->NextNode(),
            IRContext::kAnalysisDefUse |
                IRContext::kAnalysisInstrToBlockMapping);
        Instruction *new_add =
            add_builder.AddIAdd(phi_type_id, phi->result_id(), w_const_id);
        if (new_add == nullptr)
            return Status::Failure;

        InstructionBuilder phi_builder(
            context(), phi->NextNode(),
            IRContext::kAnalysisDefUse |
                IRContext::kAnalysisInstrToBlockMapping);
        Instruction *new_phi = phi_builder.AddPhi(
            phi_type_id, {zero_id, pre_block, new_add->result_id(), back_block},
            new_phi_id);
        if (new_phi == nullptr)
            return Status::Failure;

        new_add->SetInOperand(0, {new_phi_id});
        context()->UpdateDefUse(new_add);

        // Keep the original comparison, which may have other users, and make a
        // fresh one for the exit branch.  Insert it before any merge
        // instruction, which must immediately precede the branch.
        Instruction *cmp_insert_before = test;
        Instruction *prev = test->PreviousNode();
        while (prev != nullptr) {
            if (prev->opcode() == spv::Op::OpSelectionMerge ||
                prev->opcode() == spv::Op::OpLoopMerge) {
                cmp_insert_before = prev;
            } else if (prev->opcode() != spv::Op::OpLine &&
                       prev->opcode() != spv::Op::OpNoLine) {
                break;
            }
            prev = prev->PreviousNode();
        }
        InstructionBuilder cmp_builder(
            context(), cmp_insert_before,
            IRContext::kAnalysisDefUse |
                IRContext::kAnalysisInstrToBlockMapping);
        Instruction *new_cmp = cmp_builder.AddBinaryOp(
            compare->type_id(), compare->opcode(),
            phi_is_first ? new_phi_id : n, phi_is_first ? n : new_phi_id);
        if (new_cmp == nullptr)
            return Status::Failure;
        test->SetInOperand(0, {new_cmp->result_id()});
        context()->UpdateDefUse(test);
        changed = true;
    }

    if (changed) {
        LOG("gave strided compute loops a workgroup-uniform trip count");
    }
    return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

} // namespace opt
} // namespace spvtools
