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

#include "source/opt/mali_clamp_zero_pass.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "source/opt/ir_builder.h"

#define LOG_(level, fmt, ...)                                                  \
    do {                                                                       \
        char buffer[256];                                                      \
        snprintf(buffer, sizeof(buffer), "MaliClampZeroPass: " fmt,            \
                 ##__VA_ARGS__);                                               \
        consumer()(level, __FUNCTION__, {__LINE__, 0, 0}, buffer);             \
    } while (0)

#define LOG(fmt, ...) LOG_(SPV_MSG_INFO, fmt, ##__VA_ARGS__)

namespace spvtools {
namespace opt {

namespace {

// GLSL.std.450 instruction numbers.
enum {
    kFMin = 37,
    kUMin = 38,
    kSMin = 39,
    kFMax = 40,
    kUMax = 41,
    kSMax = 42,
    kFClamp = 43,
    kUClamp = 44,
    kSClamp = 45,
    kNMin = 79,
    kNMax = 80,
    kNClamp = 81,
};

bool IsClamp(uint32_t ext) {
    return ext == kFClamp || ext == kUClamp || ext == kSClamp || ext == kNClamp;
}

bool IsMinMax(uint32_t ext) {
    return (ext >= kFMin && ext <= kSMax) || ext == kNMin || ext == kNMax;
}

bool IsValueOp(spv::Op op) {
    const uint32_t v = static_cast<uint32_t>(op);
    return (v >= 109 && v <= 124) || (v >= 126 && v <= 148) ||
           op == spv::Op::OpCopyObject || op == spv::Op::OpCompositeConstruct ||
           (v >= 164 && v <= 200);
}

} // namespace

bool MaliClampZeroPass::IsGlslStd450(uint32_t set_id) {
    Instruction *set = get_def_use_mgr()->GetDef(set_id);
    if (set == nullptr || set->opcode() != spv::Op::OpExtInstImport)
        return false;
    if (set->NumInOperands() < 1)
        return false;
    const Operand &name = set->GetInOperand(0);
    if (name.words.size() < 4)
        return false;
    return std::memcmp(name.words.data(), "GLSL.std.450", 13) == 0;
}

bool MaliClampZeroPass::IsConstant(uint32_t id) {
    Instruction *def = get_def_use_mgr()->GetDef(id);
    if (def == nullptr)
        return false;
    switch (def->opcode()) {
    case spv::Op::OpConstant:
    case spv::Op::OpConstantNull:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
        return true;
    default:
        return false;
    }
}

bool MaliClampZeroPass::IsConstantZero(uint32_t id) {
    Instruction *def = get_def_use_mgr()->GetDef(id);
    if (def == nullptr)
        return false;
    switch (def->opcode()) {
    case spv::Op::OpConstant: {
        // A 64-bit constant has an operand with two literal words, so do not
        // assume every in-operand is a single word.
        if (def->NumInOperands() == 0)
            return false;
        for (uint32_t i = 0; i < def->NumInOperands(); ++i) {
            for (uint32_t word : def->GetInOperand(i).words) {
                if (word != 0)
                    return false;
            }
        }
        return true;
    }
    case spv::Op::OpConstantNull:
        return true;
    case spv::Op::OpConstantComposite: {
        if (def->NumInOperands() == 0)
            return false;
        for (uint32_t i = 0; i < def->NumInOperands(); ++i) {
            if (!IsConstantZero(def->GetSingleWordInOperand(i)))
                return false;
        }
        return true;
    }
    default:
        return false;
    }
}

bool MaliClampZeroPass::IsConstantOrSelected(uint32_t id, int depth) {
    return IsConstant(id) || IsConstantSelected(id, depth);
}

uint32_t MaliClampZeroPass::FindRootVariable(uint32_t id) {
    for (int depth = 0; depth < 16 && id != 0; ++depth) {
        Instruction *def = get_def_use_mgr()->GetDef(id);
        if (def == nullptr)
            return 0;
        switch (def->opcode()) {
        case spv::Op::OpVariable:
            return id;
        case spv::Op::OpAccessChain:
        case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpPtrAccessChain:
        case spv::Op::OpInBoundsPtrAccessChain:
        case spv::Op::OpCopyObject:
            if (def->NumInOperands() < 1)
                return 0;
            id = def->GetSingleWordInOperand(0);
            break;
        default:
            return 0;
        }
    }
    return 0;
}

bool MaliClampZeroPass::VariableHoldsConstantSelected(uint32_t var_id,
                                                      int depth) {
    if (depth > 16)
        return false;
    Instruction *var = get_def_use_mgr()->GetDef(var_id);
    if (var == nullptr || var->opcode() != spv::Op::OpVariable)
        return false;
    const uint32_t storage = var->GetSingleWordInOperand(0);
    if (storage != static_cast<uint32_t>(spv::StorageClass::Function) &&
        storage != static_cast<uint32_t>(spv::StorageClass::Private) &&
        storage != static_cast<uint32_t>(spv::StorageClass::Output)) {
        return false;
    }

    bool any = false;
    if (var->NumInOperands() >= 2) {
        if (!IsConstantOrSelected(var->GetSingleWordInOperand(1), depth + 1)) {
            return false;
        }
        any = true;
    }
    for (Function &func : *context()->module()) {
        for (BasicBlock &block : func) {
            for (Instruction &inst : block) {
                if (inst.opcode() == spv::Op::OpCopyMemory ||
                    inst.opcode() == spv::Op::OpCopyMemorySized) {
                    if ((inst.NumInOperands() >= 1 &&
                         FindRootVariable(inst.GetSingleWordInOperand(0)) ==
                             var_id) ||
                        (inst.NumInOperands() >= 2 &&
                         FindRootVariable(inst.GetSingleWordInOperand(1)) ==
                             var_id)) {
                        return false;
                    }
                    continue;
                }
                if (inst.opcode() != spv::Op::OpStore ||
                    inst.NumInOperands() < 2)
                    continue;
                if (FindRootVariable(inst.GetSingleWordInOperand(0)) != var_id)
                    continue;
                if (!IsConstantOrSelected(inst.GetSingleWordInOperand(1),
                                          depth + 1)) {
                    return false;
                }
                any = true;
            }
        }
    }
    return any;
}

bool MaliClampZeroPass::IsConstantSelected(uint32_t id, int depth) {
    if (depth > 16)
        return false;
    Instruction *def = get_def_use_mgr()->GetDef(id);
    if (def == nullptr)
        return false;
    switch (def->opcode()) {
    case spv::Op::OpSelect: {
        if (def->NumInOperands() < 3)
            return false;
        return IsConstantOrSelected(def->GetSingleWordInOperand(1),
                                    depth + 1) &&
               IsConstantOrSelected(def->GetSingleWordInOperand(2), depth + 1);
    }
    case spv::Op::OpPhi: {
        if (def->NumInOperands() < 2)
            return false;
        bool any = false;
        for (uint32_t i = 0; i + 1 < def->NumInOperands(); i += 2) {
            if (!IsConstantOrSelected(def->GetSingleWordInOperand(i),
                                      depth + 1)) {
                return false;
            }
            any = true;
        }
        return any;
    }
    case spv::Op::OpLoad: {
        if (def->NumInOperands() < 1)
            return false;
        const uint32_t var = FindRootVariable(def->GetSingleWordInOperand(0));
        return var != 0 && VariableHoldsConstantSelected(var, depth + 1);
    }
    case spv::Op::OpCompositeExtract: {
        if (def->NumInOperands() < 1)
            return false;
        return IsConstantOrSelected(def->GetSingleWordInOperand(0), depth + 1);
    }
    case spv::Op::OpVectorShuffle:
    case spv::Op::OpCompositeInsert: {
        if (def->NumInOperands() < 2)
            return false;
        return IsConstantOrSelected(def->GetSingleWordInOperand(0),
                                    depth + 1) &&
               IsConstantOrSelected(def->GetSingleWordInOperand(1), depth + 1);
    }
    case spv::Op::OpExtInst: {
        bool any = false;
        for (uint32_t i = 2; i < def->NumInOperands(); ++i) {
            if (!IsConstantOrSelected(def->GetSingleWordInOperand(i),
                                      depth + 1)) {
                return false;
            }
            any = true;
        }
        return any;
    }
    default: {
        if (!IsValueOp(def->opcode()))
            return false;
        bool any = false;
        for (uint32_t i = 0; i < def->NumInOperands(); ++i) {
            if (!IsConstantOrSelected(def->GetSingleWordInOperand(i),
                                      depth + 1)) {
                return false;
            }
            any = true;
        }
        return any;
    }
    }
}

uint32_t MaliClampZeroPass::GetScalarZero(uint32_t scalar_type_id,
                                          Function *func,
                                          Instruction *insert_before) {
    (void)insert_before;
    if (func == nullptr)
        return 0;
    const uint64_t key =
        (static_cast<uint64_t>(func->result_id()) << 32) | scalar_type_id;
    auto load_it = zero_load_by_func_type_.find(key);
    if (load_it != zero_load_by_func_type_.end())
        return load_it->second;

    analysis::Type *type = context()->get_type_mgr()->GetType(scalar_type_id);
    if (type == nullptr)
        return 0;

    uint32_t zero_const = 0;
    auto const_it = zero_const_by_type_.find(scalar_type_id);
    if (const_it != zero_const_by_type_.end()) {
        zero_const = const_it->second;
    } else {
        uint32_t width = 32;
        if (analysis::Integer *int_type = type->AsInteger()) {
            width = int_type->width();
        } else if (analysis::Float *float_type = type->AsFloat()) {
            width = float_type->width();
        }
        std::vector<uint32_t> words(width > 32 ? 2 : 1, 0);
        const analysis::Constant *constant =
            context()->get_constant_mgr()->GetConstant(type, words);
        if (constant == nullptr)
            return 0;
        Instruction *def =
            context()->get_constant_mgr()->GetDefiningInstruction(constant);
        if (def == nullptr)
            return 0;
        zero_const = def->result_id();
        zero_const_by_type_[scalar_type_id] = zero_const;
    }

    if (bitfield_insert_) {
        uint32_t width = 32;
        if (analysis::Integer *int_type = type->AsInteger())
            width = int_type->width();
        else if (analysis::Float *float_type = type->AsFloat())
            width = float_type->width();
        if (width == 32) {
            auto bfi_it = zero_bfi_by_func_type_.find(key);
            if (bfi_it != zero_bfi_by_func_type_.end())
                return bfi_it->second;

            BasicBlock &block = *func->begin();
            Instruction *at = nullptr;
            for (Instruction &inst : block) {
                if (inst.opcode() != spv::Op::OpVariable) {
                    at = &inst;
                    break;
                }
            }
            if (at == nullptr)
                at = block.terminator();
            InstructionBuilder builder(
                context(), at,
                IRContext::kAnalysisDefUse |
                    IRContext::kAnalysisInstrToBlockMapping);
            const uint32_t uint_type =
                context()->get_type_mgr()->GetUIntTypeId();
            const uint32_t u0 =
                context()->get_constant_mgr()->GetUIntConstId(0);
            if (uint_type == 0 || u0 == 0)
                return 0;
            uint32_t zero_id = 0;
            if (type->AsInteger() != nullptr) {
                const uint32_t z = context()->TakeNextId();
                if (z == 0)
                    return 0;
                builder.AddInstruction(std::unique_ptr<Instruction>(
                    new Instruction(context(), spv::Op::OpBitFieldInsert,
                                    scalar_type_id, z,
                                    {{SPV_OPERAND_TYPE_ID, {zero_const}},
                                     {SPV_OPERAND_TYPE_ID, {zero_const}},
                                     {SPV_OPERAND_TYPE_ID, {u0}},
                                     {SPV_OPERAND_TYPE_ID, {u0}}})));
                zero_id = z;
            } else {
                const uint32_t c = context()->TakeNextId();
                const uint32_t b = context()->TakeNextId();
                const uint32_t z = context()->TakeNextId();
                if (c == 0 || b == 0 || z == 0)
                    return 0;
                builder.AddInstruction(std::unique_ptr<Instruction>(
                    new Instruction(context(), spv::Op::OpBitcast, uint_type, c,
                                    {{SPV_OPERAND_TYPE_ID, {zero_const}}})));
                builder.AddInstruction(std::unique_ptr<Instruction>(
                    new Instruction(context(), spv::Op::OpBitFieldInsert,
                                    uint_type, b,
                                    {{SPV_OPERAND_TYPE_ID, {c}},
                                     {SPV_OPERAND_TYPE_ID, {u0}},
                                     {SPV_OPERAND_TYPE_ID, {u0}},
                                     {SPV_OPERAND_TYPE_ID, {u0}}})));
                builder.AddInstruction(std::unique_ptr<Instruction>(
                    new Instruction(context(), spv::Op::OpBitcast,
                                    scalar_type_id, z,
                                    {{SPV_OPERAND_TYPE_ID, {b}}})));
                zero_id = z;
            }
            zero_bfi_by_func_type_[key] = zero_id;
            return zero_id;
        }
    }

    uint32_t var = 0;
    auto var_it = zero_var_by_type_.find(scalar_type_id);
    if (var_it != zero_var_by_type_.end()) {
        var = var_it->second;
    } else {
        const uint32_t ptr_type = context()->get_type_mgr()->FindPointerToType(
            scalar_type_id, spv::StorageClass::Private);
        if (ptr_type == 0)
            return 0;
        const uint32_t var_id = context()->TakeNextId();
        if (var_id == 0)
            return 0;
        std::unique_ptr<Instruction> var_inst(new Instruction(
            context(), spv::Op::OpVariable, ptr_type, var_id,
            {{SPV_OPERAND_TYPE_STORAGE_CLASS,
              {static_cast<uint32_t>(spv::StorageClass::Private)}}}));
        context()->AddGlobalValue(std::move(var_inst));
        var = var_id;
        zero_var_by_type_[scalar_type_id] = var;

        // Store the zero at the top of every entry point.
        for (uint32_t entry_id : entry_point_functions_) {
            Function *entry = context()->GetFunction(entry_id);
            if (entry == nullptr || entry->begin() == entry->end())
                continue;
            BasicBlock &block = *entry->begin();
            Instruction *at = nullptr;
            for (Instruction &inst : block) {
                if (inst.opcode() != spv::Op::OpVariable) {
                    at = &inst;
                    break;
                }
            }
            if (at == nullptr)
                at = block.terminator();
            InstructionBuilder builder(
                context(), at,
                IRContext::kAnalysisDefUse |
                    IRContext::kAnalysisInstrToBlockMapping);
            builder.AddStore(var, zero_const);
        }
    }

    // Load it at the top of this function, after the entry point's store when
    // this is the entry point itself.
    BasicBlock &block = *func->begin();
    Instruction *at = nullptr;
    for (auto it = block.begin(); it != block.end(); ++it) {
        if (it->opcode() == spv::Op::OpStore && it->NumInOperands() >= 1 &&
            it->GetSingleWordInOperand(0) == var) {
            auto next = it;
            ++next;
            if (next != block.end())
                at = &*next;
            break;
        }
    }
    if (at == nullptr) {
        for (Instruction &inst : block) {
            if (inst.opcode() != spv::Op::OpVariable) {
                at = &inst;
                break;
            }
        }
    }
    if (at == nullptr)
        at = block.terminator();

    InstructionBuilder builder(context(), at,
                               IRContext::kAnalysisDefUse |
                                   IRContext::kAnalysisInstrToBlockMapping);
    Instruction *load = builder.AddLoad(scalar_type_id, var);
    if (load == nullptr)
        return 0;
    zero_load_by_func_type_[key] = load->result_id();
    return load->result_id();
}

uint32_t MaliClampZeroPass::GetOpaqueZero(uint32_t type_id, Function *func,
                                          Instruction *insert_before) {
    analysis::Type *type = context()->get_type_mgr()->GetType(type_id);
    if (type == nullptr)
        return 0;
    if (analysis::Vector *vec = type->AsVector()) {
        const uint32_t scalar_type_id =
            context()->get_type_mgr()->GetId(vec->element_type());
        if (scalar_type_id == 0)
            return 0;
        const uint32_t scalar =
            GetScalarZero(scalar_type_id, func, insert_before);
        if (scalar == 0)
            return 0;

        const uint64_t key =
            (static_cast<uint64_t>(func->result_id()) << 32) | type_id;
        auto it = zero_vector_by_func_type_.find(key);
        if (it != zero_vector_by_func_type_.end())
            return it->second;

        std::vector<uint32_t> ids(vec->element_count(), scalar);
        InstructionBuilder builder(context(), insert_before,
                                   IRContext::kAnalysisDefUse |
                                       IRContext::kAnalysisInstrToBlockMapping);
        Instruction *construct = builder.AddCompositeConstruct(type_id, ids);
        if (construct == nullptr)
            return 0;
        zero_vector_by_func_type_[key] = construct->result_id();
        return construct->result_id();
    }
    return GetScalarZero(type_id, func, insert_before);
}

Pass::Status MaliClampZeroPass::Process() {
    entry_point_functions_.clear();
    zero_var_by_type_.clear();
    zero_const_by_type_.clear();
    zero_load_by_func_type_.clear();
    zero_vector_by_func_type_.clear();
    zero_bfi_by_func_type_.clear();

    for (auto &entry : context()->module()->entry_points()) {
        if (entry.NumInOperands() >= 2) {
            entry_point_functions_.insert(entry.GetSingleWordInOperand(1));
        }
    }

    bool changed = false;
    for (Function &func : *context()->module()) {
        if (func.IsDeclaration())
            continue;
        for (BasicBlock &block : func) {
            for (Instruction &inst : block) {
                if (inst.opcode() != spv::Op::OpExtInst)
                    continue;
                if (inst.NumInOperands() < 4)
                    continue;
                if (!IsGlslStd450(inst.GetSingleWordInOperand(0)))
                    continue;
                const uint32_t ext = inst.GetSingleWordInOperand(1);

                if (IsClamp(ext)) {
                    if (inst.NumInOperands() < 5)
                        continue;
                    const uint32_t value = inst.GetSingleWordInOperand(2);
                    const uint32_t lo = inst.GetSingleWordInOperand(3);
                    const uint32_t hi = inst.GetSingleWordInOperand(4);
                    const bool lo_zero = IsConstantZero(lo);
                    const bool hi_zero = IsConstantZero(hi);
                    if (!lo_zero && !hi_zero)
                        continue;
                    if (!rewrite_all_ && !IsConstantSelected(value, 0))
                        continue;

                    bool rewrote = false;
                    if (lo_zero) {
                        const uint32_t type_id =
                            get_def_use_mgr()->GetDef(lo)->type_id();
                        const uint32_t zero =
                            GetOpaqueZero(type_id, &func, &inst);
                        if (zero != 0) {
                            inst.SetInOperand(3, {zero});
                            rewrote = true;
                        }
                    }
                    if (hi_zero) {
                        const uint32_t type_id =
                            get_def_use_mgr()->GetDef(hi)->type_id();
                        const uint32_t zero =
                            GetOpaqueZero(type_id, &func, &inst);
                        if (zero != 0) {
                            inst.SetInOperand(4, {zero});
                            rewrote = true;
                        }
                    }
                    if (rewrote) {
                        context()->UpdateDefUse(&inst);
                        changed = true;
                    }
                } else if (IsMinMax(ext)) {
                    if (inst.NumInOperands() < 4)
                        continue;
                    const uint32_t a = inst.GetSingleWordInOperand(2);
                    const uint32_t b = inst.GetSingleWordInOperand(3);
                    const bool a_zero = IsConstantZero(a);
                    const bool b_zero = IsConstantZero(b);
                    if (a_zero == b_zero)
                        continue;
                    const uint32_t value = a_zero ? b : a;
                    if (!rewrite_all_ && !IsConstantSelected(value, 0))
                        continue;

                    const uint32_t zero_operand = a_zero ? a : b;
                    const uint32_t type_id =
                        get_def_use_mgr()->GetDef(zero_operand)->type_id();
                    const uint32_t zero = GetOpaqueZero(type_id, &func, &inst);
                    if (zero != 0) {
                        inst.SetInOperand(a_zero ? 2 : 3, {zero});
                        context()->UpdateDefUse(&inst);
                        changed = true;
                    }
                }
            }
        }
    }

    if (changed) {
        LOG("gave min/max/clamp ops with a constant zero limit an opaque zero");
    }
    return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

} // namespace opt
} // namespace spvtools
