// Copyright (c) 2018 Google Inc.
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

#ifndef SOURCE_REDUCE_REDUCTION_PASS_H_
#define SOURCE_REDUCE_REDUCTION_PASS_H_

#include "spirv-tools/libspirv.hpp"

#include "reduction_opportunity.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

class ReductionPass {
 public:
  explicit ReductionPass(const spv_target_env target_env)
      : target_env_(target_env),
        is_initialized_(false),
        index_(0),
        granularity_(0) {}

  virtual ~ReductionPass() = default;

  std::vector<uint32_t> ApplyReduction(const std::vector<uint32_t>& binary);

  void SetMessageConsumer(MessageConsumer consumer);

  bool ReachedMinimumGranularity() const;

  virtual std::string GetName() const = 0;

 protected:
  virtual std::vector<std::unique_ptr<ReductionOpportunity>>
  GetAvailableOpportunities(opt::IRContext* context) const = 0;

 private:
  const spv_target_env target_env_;
  MessageConsumer consumer_;
  bool is_initialized_;
  uint32_t index_;
  uint32_t granularity_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_PASS_H_
