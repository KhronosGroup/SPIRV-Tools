// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_FUNCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_FUNCTION_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddFunction : public Transformation {
 public:
  explicit TransformationAddFunction(
      const protobufs::TransformationAddFunction& message);

  // Creates a transformation to add a non live-safe function.
  explicit TransformationAddFunction(
      const std::vector<protobufs::Instruction>& instructions);

  // Creates a transformation to add a live-safe function.
  TransformationAddFunction(
      const std::vector<protobufs::Instruction>& instructions,
      uint32_t loop_limiter_variable_id, uint32_t loop_limit_constant_id,
      const std::vector<protobufs::LoopLimiterInfo>& loop_limiters,
      uint32_t kill_unreachable_return_value_id,
      const std::vector<protobufs::AccessChainClampingInfo>&
          access_chain_clampers);

  // - |message_.instruction| must correspond to a sufficiently well-formed
  //   sequence of instructions that a function can be created from them
  // - Adding the created function to the module must lead to a valid module.
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // Adds the function defined by |message_.instruction| to the module
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

  // TODO comment
  static uint32_t GetBoundForCompositeIndex(
      opt::IRContext* context, const opt::Instruction& composite_type_inst);

  // TODO comment
  static opt::Instruction* FollowCompositeIndex(
      opt::IRContext* context, const opt::Instruction& composite_type_inst,
      uint32_t index_id);

 private:
  // Attempts to create a function from the series of instructions in
  // |message_.instruction| and add it to |context|.  If
  // |message_.is_livesafe| holds then the function is made "livesafe" in the
  // process: loop limiters are provided for all loops, all array/vector/matrix
  // accesses are clamped to be in bounds, and OpKill and OpUnreachable
  // instructions are replaced with return instructions.
  //
  // Returns false if adding the function is not possible due to the messages
  // not respecting the basic structure of a function, e.g. if there is no
  // OpFunction instruction or no blocks; in this case |context| is left in an
  // indeterminate state.
  //
  // Otherwise returns true.  Whether |context| is valid after addition of the
  // function depends on the contents of |message_.instruction|.
  //
  // Intended usage:
  // - Perform a dry run of this method on a clone of a module, and use
  //   the validator to check whether the resulting module is valid.  Working
  //   on a clone means it does not matter if the function fails to be cleanly
  //   added, or leads to an invalid module.
  // - If the dry run succeeds, run the method on the real module of interest,
  //   to add the function.
  bool TryToAddFunction(opt::IRContext* context) const;

  // TODO comment
  std::vector<uint32_t> GetLoopHeaderIds() const;

  // TODO comment
  bool FunctionContainsKillOrUnreachable() const;

  // TODO comment
  bool TryToMakeFunctionLivesafe(opt::IRContext* context) const;

  // TODO comment
  bool TryToAddLoopLimiters(opt::IRContext* context,
                            opt::Function* added_function) const;

  // TODO comment
  void TurnKillOrUnreachableIntoReturn(
      opt::IRContext* context, opt::Function* added_function,
      opt::Instruction* kill_or_unreachable_inst) const;

  // TODO comment including on return type
  bool TryToClampAccessChainIndices(opt::IRContext* context,
                                    opt::Instruction* access_chain_inst) const;

  protobufs::TransformationAddFunction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_FUNCTION_H_
