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

#include "source/opt/mali_pervertex_block_pass.h"

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <vector>

#include "source/opt/ir_builder.h"

#define LOG_(level, fmt, ...)                                                  \
    do {                                                                       \
        char buffer[256];                                                      \
        snprintf(buffer, sizeof(buffer), "MaliPerVertexBlockPass: " fmt,       \
                 ##__VA_ARGS__);                                               \
        consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer);             \
    } while (0)

#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ##__VA_ARGS__)

namespace spvtools {
namespace opt {

namespace {

constexpr int kMaxVars = 8;

struct LooseVar {
    uint32_t var = 0;
    uint32_t ptr_type = 0;
    uint32_t pointee = 0;
    uint32_t member_type = 0;
    uint32_t builtin = 0;
    bool invariant = false;
};

int FindVar(const std::array<LooseVar, kMaxVars> &vars, int n, uint32_t id) {
    for (int i = 0; i < n; ++i) {
        if (vars[i].var == id)
            return i;
    }
    return -1;
}

bool IsPerVertexBuiltIn(uint32_t builtin) {
    return builtin == static_cast<uint32_t>(spv::BuiltIn::Position) ||
           builtin == static_cast<uint32_t>(spv::BuiltIn::PointSize) ||
           builtin == static_cast<uint32_t>(spv::BuiltIn::ClipDistance) ||
           builtin == static_cast<uint32_t>(spv::BuiltIn::CullDistance);
}

} // namespace

Pass::Status MaliPerVertexBlockPass::Process() {
    // Only geometry and tessellation stages.
    spv::ExecutionModel model = spv::ExecutionModel::Max;
    uint32_t entry_point_count = 0;
    for (auto &entry : context()->module()->entry_points()) {
        model =
            static_cast<spv::ExecutionModel>(entry.GetSingleWordInOperand(0));
        ++entry_point_count;
    }
    if (entry_point_count != 1) {
        return Status::SuccessWithoutChange;
    }
    if (model != spv::ExecutionModel::Geometry &&
        model != spv::ExecutionModel::TessellationControl &&
        model != spv::ExecutionModel::TessellationEvaluation) {
        return Status::SuccessWithoutChange;
    }

    bool changed = false;
    const std::array<spv::StorageClass, 2> storages = {
        spv::StorageClass::Input, spv::StorageClass::Output};

    for (spv::StorageClass storage : storages) {
        const bool arrayed = storage == spv::StorageClass::Input ||
                             model == spv::ExecutionModel::TessellationControl;

        std::array<LooseVar, kMaxVars> vars{};
        int count = 0;
        bool unsupported = false;
        for (Instruction &inst : context()->module()->types_values()) {
            if (inst.opcode() != spv::Op::OpVariable)
                continue;
            if (static_cast<spv::StorageClass>(
                    inst.GetSingleWordInOperand(0)) != storage) {
                continue;
            }
            const uint32_t var_id = inst.result_id();
            std::optional<uint32_t> builtin;
            bool invariant = false;
            bool unsupported_decoration = false;
            for (Instruction *dec :
                 context()->get_decoration_mgr()->GetDecorationsFor(var_id,
                                                                    false)) {
                if (dec->opcode() != spv::Op::OpDecorate)
                    continue;
                const uint32_t decoration = dec->GetSingleWordInOperand(1);
                if (decoration ==
                    static_cast<uint32_t>(spv::Decoration::BuiltIn)) {
                    builtin = dec->GetSingleWordInOperand(2);
                } else if (decoration ==
                           static_cast<uint32_t>(spv::Decoration::Invariant)) {
                    invariant = true;
                } else {
                    unsupported_decoration = true;
                }
            }
            // Ordinary interface variables (Location, Flat, ...) are not our
            // concern; only unexpected decorations on the per-vertex built-ins
            // should stop the rewrite.
            if (!builtin || !IsPerVertexBuiltIn(*builtin))
                continue;
            if (unsupported_decoration) {
                unsupported = true;
                break;
            }
            if (count == kMaxVars) {
                unsupported = true;
                break;
            }

            analysis::Type *ptr_type =
                context()->get_type_mgr()->GetType(inst.type_id());
            analysis::Pointer *pointer =
                ptr_type != nullptr ? ptr_type->AsPointer() : nullptr;
            if (pointer == nullptr) {
                unsupported = true;
                break;
            }
            const uint32_t pointee_id =
                context()->get_type_mgr()->GetId(pointer->pointee_type());
            if (pointee_id == 0) {
                unsupported = true;
                break;
            }
            uint32_t member_type_id = pointee_id;
            if (arrayed) {
                const analysis::Array *array =
                    pointer->pointee_type()->AsArray();
                if (array == nullptr) {
                    unsupported = true;
                    break;
                }
                member_type_id =
                    context()->get_type_mgr()->GetId(array->element_type());
                if (member_type_id == 0) {
                    unsupported = true;
                    break;
                }
            }
            vars[count++] = {var_id,         inst.type_id(), pointee_id,
                             member_type_id, *builtin,       invariant};
        }
        if (unsupported || count == 0)
            continue;

        // Direct uses of the loose variables must all be rewritable.  Uses of
        // a derived pointer (an access chain) are fine because only the
        // chain's base operand changes.
        bool uses_ok = true;
        for (int i = 0; i < count && uses_ok; ++i) {
            const uint32_t var_id = vars[i].var;
            uses_ok = get_def_use_mgr()->WhileEachUser(
                var_id, [&](Instruction *user) {
                    switch (user->opcode()) {
                    case spv::Op::OpAccessChain:
                    case spv::Op::OpInBoundsAccessChain:
                        // An arrayed interface needs one index (into the
                        // array) before we insert the member index.
                        if (user->NumInOperands() < (arrayed ? 2u : 1u) ||
                            user->GetSingleWordInOperand(0) != var_id) {
                            return false;
                        }
                        return true;
                    case spv::Op::OpLoad:
                    case spv::Op::OpStore:
                        // Whole loads/stores of an arrayed interface are not
                        // handled here.
                        if (arrayed || user->NumInOperands() < 1 ||
                            user->GetSingleWordInOperand(0) != var_id) {
                            return false;
                        }
                        return true;
                    case spv::Op::OpEntryPoint:
                    case spv::Op::OpDecorate:
                    case spv::Op::OpName:
                        return true;
                    default:
                        return false;
                    }
                });
        }
        if (!uses_ok)
            continue;

        // Validate the array length before touching the module.  Every loose
        // variable of an arrayed interface shares the same array type.
        uint32_t array_length_id = 0;
        if (arrayed) {
            analysis::Type *pointee =
                context()->get_type_mgr()->GetType(vars[0].pointee);
            analysis::Array *array =
                pointee != nullptr ? pointee->AsArray() : nullptr;
            if (array == nullptr)
                continue;
            array_length_id = array->LengthId();
            if (array_length_id == 0)
                continue;
            // All members are placed in one arrayed block, so they must share
            // the same outer array length.
            bool same_length = true;
            for (int i = 1; i < count && same_length; ++i) {
                analysis::Type *other_pointee =
                    context()->get_type_mgr()->GetType(vars[i].pointee);
                const analysis::Array *other_array =
                    other_pointee != nullptr ? other_pointee->AsArray()
                                             : nullptr;
                same_length = other_array != nullptr &&
                              other_array->LengthId() == array_length_id;
            }
            if (!same_length)
                continue;
        }

        // The block struct type.
        const uint32_t block_type_id = context()->TakeNextId();
        if (block_type_id == 0)
            return Status::Failure;
        Instruction::OperandList struct_ops;
        for (int i = 0; i < count; ++i) {
            struct_ops.push_back({SPV_OPERAND_TYPE_ID, {vars[i].member_type}});
        }
        std::unique_ptr<Instruction> struct_inst(
            new Instruction(context(), spv::Op::OpTypeStruct, 0, block_type_id,
                            std::move(struct_ops)));
        context()->AddType(std::move(struct_inst));

        uint32_t var_type_id = block_type_id;
        if (arrayed) {
            const uint32_t new_array_id = context()->TakeNextId();
            if (new_array_id == 0)
                return Status::Failure;
            std::unique_ptr<Instruction> array_inst(new Instruction(
                context(), spv::Op::OpTypeArray, 0, new_array_id,
                {{SPV_OPERAND_TYPE_ID, {block_type_id}},
                 {SPV_OPERAND_TYPE_ID, {array_length_id}}}));
            context()->AddType(std::move(array_inst));
            var_type_id = new_array_id;
        }

        const uint32_t ptr_type_id =
            context()->get_type_mgr()->FindPointerToType(var_type_id, storage);
        if (ptr_type_id == 0)
            return Status::Failure;

        const uint32_t block_var_id = context()->TakeNextId();
        if (block_var_id == 0)
            return Status::Failure;
        std::unique_ptr<Instruction> var_inst(new Instruction(
            context(), spv::Op::OpVariable, ptr_type_id, block_var_id,
            {{SPV_OPERAND_TYPE_STORAGE_CLASS,
              {static_cast<uint32_t>(storage)}}}));
        context()->AddGlobalValue(std::move(var_inst));

        // Decorations: members carry the built-ins, the struct is a Block.
        for (int i = 0; i < count; ++i) {
            context()->get_decoration_mgr()->AddMemberDecoration(
                block_type_id, static_cast<uint32_t>(i),
                static_cast<uint32_t>(spv::Decoration::BuiltIn),
                vars[i].builtin);
            if (vars[i].invariant) {
                context()->get_decoration_mgr()->AddDecoration(
                    spv::Op::OpMemberDecorate,
                    {{SPV_OPERAND_TYPE_ID, {block_type_id}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                      {static_cast<uint32_t>(i)}},
                     {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                      {static_cast<uint32_t>(spv::Decoration::Invariant)}}});
            }
        }
        context()->get_decoration_mgr()->AddDecoration(
            block_type_id, static_cast<uint32_t>(spv::Decoration::Block));

        // Entry points: the old variables out, the block in (once).
        for (Instruction &entry : context()->module()->entry_points()) {
            if (entry.NumInOperands() < 3)
                continue;
            Instruction::OperandList new_ops;
            new_ops.push_back(entry.GetInOperand(0));
            new_ops.push_back(entry.GetInOperand(1));
            new_ops.push_back(entry.GetInOperand(2));
            bool had = false;
            for (uint32_t i = 3; i < entry.NumInOperands(); ++i) {
                const uint32_t id = entry.GetSingleWordInOperand(i);
                if (FindVar(vars, count, id) >= 0) {
                    had = true;
                    continue;
                }
                new_ops.push_back(entry.GetInOperand(i));
            }
            if (had) {
                new_ops.push_back({SPV_OPERAND_TYPE_ID, {block_var_id}});
            }
            entry.SetInOperands(std::move(new_ops));
        }

        for (Function &func : *context()->module()) {
            if (func.IsDeclaration())
                continue;
            for (BasicBlock &block : func) {
                for (Instruction &inst : block) {
                    if (inst.opcode() == spv::Op::OpAccessChain ||
                        inst.opcode() == spv::Op::OpInBoundsAccessChain) {
                        if (inst.NumInOperands() < 1)
                            continue;
                        const int k = FindVar(vars, count,
                                              inst.GetSingleWordInOperand(0));
                        if (k < 0)
                            continue;
                        const uint32_t member_index =
                            context()->get_constant_mgr()->GetUIntConstId(
                                static_cast<uint32_t>(k));
                        Instruction::OperandList new_ops;
                        new_ops.push_back(
                            {SPV_OPERAND_TYPE_ID, {block_var_id}});
                        uint32_t src = 1;
                        if (arrayed && src < inst.NumInOperands()) {
                            new_ops.push_back(inst.GetInOperand(src++));
                        }
                        new_ops.push_back(
                            {SPV_OPERAND_TYPE_ID, {member_index}});
                        while (src < inst.NumInOperands()) {
                            new_ops.push_back(inst.GetInOperand(src++));
                        }
                        inst.SetInOperands(std::move(new_ops));
                        context()->UpdateDefUse(&inst);
                        continue;
                    }
                    if (inst.opcode() != spv::Op::OpLoad &&
                        inst.opcode() != spv::Op::OpStore) {
                        continue;
                    }
                    if (inst.NumInOperands() < 1)
                        continue;
                    const int k =
                        FindVar(vars, count, inst.GetSingleWordInOperand(0));
                    if (k < 0)
                        continue;
                    const uint32_t member_index =
                        context()->get_constant_mgr()->GetUIntConstId(
                            static_cast<uint32_t>(k));
                    const uint32_t member_ptr_type =
                        context()->get_type_mgr()->FindPointerToType(
                            vars[k].member_type, storage);
                    if (member_ptr_type == 0)
                        continue;
                    InstructionBuilder builder(
                        context(), &inst,
                        IRContext::kAnalysisDefUse |
                            IRContext::kAnalysisInstrToBlockMapping);
                    Instruction *access = builder.AddAccessChain(
                        member_ptr_type, block_var_id, {member_index});
                    if (access == nullptr)
                        continue;
                    inst.SetInOperand(0, {access->result_id()});
                    context()->UpdateDefUse(&inst);
                }
            }
        }

        for (int i = 0; i < count; ++i) {
            context()->KillNamesAndDecorates(vars[i].var);
            Instruction *var = get_def_use_mgr()->GetDef(vars[i].var);
            if (var != nullptr)
                context()->KillInst(var);
        }
        changed = true;
    }

    if (changed) {
        LOG("folded loose per-vertex built-ins into a block");
    }
    return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

} // namespace opt
} // namespace spvtools
