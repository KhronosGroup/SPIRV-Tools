#ifndef SOURCE_LINT_UNINITIALIZED_ANALYSIS_H_
#define SOURCE_LINT_UNINITIALIZED_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <unordered_map>
#include <variant>
#include <vector>

#include "source/lint/variable_state.h"
#include "source/opt/basic_block.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/loop_descriptor.h"
#include "spirv/unified1/spirv.hpp11"

namespace spvtools {
namespace lint {
namespace uninitialized_variables {

using BBId = uint32_t;
// Result id of an OpVariable
using VarId = uint32_t;
// Result id of an OpType__
using TypeId = uint32_t;

struct VarStore {
  VarId var_id;
  const opt::Instruction* instruction;
  State state;
};

struct VarLoad {
  VarId var_id;
  const opt::Instruction* instruction;
  State state;
};

using VarAccess = std::variant<VarStore, VarLoad>;

struct BasicBlockAnalysis {
  std::vector<VarAccess> var_accesses;
};

struct CFLoopAnalysis;
struct CFSwitchAnalysis;
struct CFSequentialAnalysis;

struct CFCallAnalysis {
  const opt::Instruction* inst;
};

using AnyAnalysis =
    std::variant<CFLoopAnalysis, CFSwitchAnalysis, CFSequentialAnalysis,
                 CFCallAnalysis, BasicBlockAnalysis>;

struct CFSequentialAnalysis {
  std::vector<AnyAnalysis> blocks;
};

struct CFLoopAnalysis {
  CFSequentialAnalysis header_analysis;
  CFSequentialAnalysis body_analysis;
  CFSequentialAnalysis continue_analysis;
  CFSequentialAnalysis trailer_analysis;
};

struct CFSwitchAnalysis {
  CFSequentialAnalysis header;
  std::vector<CFSequentialAnalysis> branches;
};

struct Analysis {
  std::vector<BBId> incoming_;
  AnyAnalysis inner_;
};

struct BadAccess {
  VarId var_id;
  const opt::Instruction* op_load;
};

struct FunctionCall {
  FunctionCall(const FunctionCall&) = default;
  const opt::Instruction* inst;
  std::unordered_map<VarId, State> preconditions;
};

struct FunctionReturn {
  const opt::Instruction* inst;
  std::unordered_map<VarId, State> postconditions;
};

struct UnmetPrecondition {
  VarId var_id;
  const opt::Instruction* inst;
  State state_have;
  State state_need;
  // OpFunctionCall that lead to accessing the uninitialized (global) variable.
  // The function in which the actual OpLoad happened is the first entry.
  // Unused for local analysis
  std::vector<const opt::Instruction*> call_trace;
};

std::ostream& operator<<(const std::ostream&, const VarStateMap&);

struct CheckResult {
  std::vector<UnmetPrecondition> unmet_preconditions;
  VarStateMap postconditions;
  std::vector<FunctionCall> function_calls;
};

class UninitializedReadAnalysis {
 public:
  UninitializedReadAnalysis(opt::IRContext& context,
                            const opt::Function& function)
      : context_(context),
        function_(function),
        post_dom_(*context.GetPostDominatorAnalysis(&function)),
        loop_desc_(&context, &function) {}

  CheckResult RunLocal() const;
  CheckResult RunPrivate(
      const std::unordered_map<uint32_t, CheckResult>&) const;

 private:
  struct StopAtLabel {
    BBId label;
    bool inclusive;
  };
  CFSwitchAnalysis AnalyzeSwitch(const opt::BasicBlock&) const;
  CFLoopAnalysis AnalyzeLoop(const opt::BasicBlock&) const;
  CFSequentialAnalysis AnalyzeSequential(
      const opt::BasicBlock&, const std::optional<StopAtLabel> = {},
      std::optional<BBId> ignore_merge_block = {}) const;
  // Follows unconditional branches from start, until the terminator is no
  // longer an OpBranch or stop_at_label was encountered. Returns the label at
  // which the walk stopped, or nullopt if start unconditionally leads to
  // Return/Abort.
  std::optional<BBId> CollectUnconditionalSuccessors(
      const opt::BasicBlock& start, std::vector<BBId>& out_bbs,
      std::optional<BBId> stop_at_label = {}) const;
  CheckResult Check(const AnyAnalysis& analysis, const VarStateMap& preconditions,
                    spv::StorageClass,
                    const std::unordered_map<uint32_t, CheckResult>&) const;
  spv::StorageClass GetStorageClass(VarId id) const;

  opt::IRContext& context_;
  const opt::Function& function_;
  opt::PostDominatorAnalysis& post_dom_;
  opt::LoopDescriptor loop_desc_;
};

struct BadLocalAccess {
  VarId var_id;
  const opt::Instruction* load;
  State state_have;
  State state_missing;
};

struct LocalResult {
  uint32_t function_id;
  std::vector<BadLocalAccess> bad_accesses;
};

struct BadGlobalAccess {
  VarId var_id;
  const opt::Instruction* op_load;
  State state_have;
  State state_missing;
  std::vector<const opt::Instruction*> call_sequence;
};

struct GlobalResult {
  opt::Instruction* op_entrypoint;
  std::vector<BadGlobalAccess> bad_accesses;
};

struct CheckUninitializedResult {
  std::vector<LocalResult> locals;
  std::vector<GlobalResult> globals;
};

class ModuleAnalysis {
 public:
  explicit ModuleAnalysis(opt::IRContext& context) : context_(context) {}
  CheckUninitializedResult Run() const;

 private:
  opt::IRContext& context_;
};

}  // namespace uninitialized_variables
}  // namespace lint
}  // namespace spvtools

#endif  // SOURCE_LINT_UNINITIALIZED_ANALYSIS_H_
