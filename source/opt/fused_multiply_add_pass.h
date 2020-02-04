

#ifndef SOURCE_OPT_FUSED_MULTIPLY_ADD_H_
#define SOURCE_OPT_FUSED_MULTIPLY_ADD_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class FusedMultiplyAddPass : public Pass {
 public:
  FusedMultiplyAddPass() {}

  const char* name() const override { return "fused-multiply-add-pass"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  bool ProcessSpvOpFMul(IRContext* ctx, Instruction* instruction,
                        uint32_t inst_set_id);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FUSED_MULTIPLY_ADD_H_
