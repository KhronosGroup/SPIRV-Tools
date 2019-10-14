

#ifndef SOURCE_OPT_FUSED_MULTIPLY_ADD_H_
#define SOURCE_OPT_FUSED_MULTIPLY_ADD_H_

#include <cstdio>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/function.h"
#include "source/opt/pass.h"
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class FusedMultiplyAddPass : public Pass {
 public:
  FusedMultiplyAddPass() {
  }

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
  
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FUSED_MULTIPLY_ADD_H_
