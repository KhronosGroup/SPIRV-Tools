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

#ifndef SOURCE_OPT_MALI_CLAMP_ZERO_PASS_H_
#define SOURCE_OPT_MALI_CLAMP_ZERO_PASS_H_

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// MaliBridge (Arm erratum 3779215; r38p1-r49p0 and r50p0-r51p0, fixed in r49p1 and
// r52p0+): Mali miscompiles a min/max/clamp whose limit is a constant zero when
// the clamped value is selected at run time from compile-time constants - the
// clamp becomes a min or a max. This pass gives such ops a zero the compiler
// cannot see as a constant: a Private variable per scalar type, stored with 0
// at the top of every entry point and loaded at the top of each function that
// needs it.
class MaliClampZeroPass : public Pass {
  public:
    explicit MaliClampZeroPass(bool rewrite_all = false,
                               bool bitfield_insert = false)
        : rewrite_all_(rewrite_all), bitfield_insert_(bitfield_insert) {}

    const char *name() const override { return "mali-clamp-zero"; }
    Status Process() override;

    IRContext::Analysis GetPreservedAnalyses() override {
        return IRContext::kAnalysisNone;
    }

  private:
    bool IsConstant(uint32_t id);
    bool IsConstantZero(uint32_t id);
    bool IsConstantSelected(uint32_t id, int depth);
    bool IsConstantOrSelected(uint32_t id, int depth);
    uint32_t FindRootVariable(uint32_t id);
    bool VariableHoldsConstantSelected(uint32_t var_id, int depth);
    bool IsGlslStd450(uint32_t set_id);

    // Returns an id holding a zero of `type_id` that the compiler cannot fold,
    // creating the Private variable / load / composite as needed.
    uint32_t GetOpaqueZero(uint32_t type_id, Function *func,
                           Instruction *insert_before);
    uint32_t GetScalarZero(uint32_t scalar_type_id, Function *func,
                           Instruction *insert_before);

    bool rewrite_all_;
    bool bitfield_insert_;

    std::unordered_map<uint32_t, uint32_t> zero_var_by_type_;
    std::unordered_map<uint32_t, uint32_t> zero_const_by_type_;
    std::unordered_map<uint64_t, uint32_t> zero_load_by_func_type_;
    std::unordered_map<uint64_t, uint32_t> zero_vector_by_func_type_;
    std::unordered_map<uint64_t, uint32_t> zero_bfi_by_func_type_;
    std::unordered_set<uint32_t> entry_point_functions_;
};

} // namespace opt
} // namespace spvtools

#endif // SOURCE_OPT_MALI_CLAMP_ZERO_PASS_H_
