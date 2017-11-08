// Copyright (c) 2017 Google Inc.
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

#ifndef LIBSPIRV_OPT_MERGE_RETURN_PASS_H_
#define LIBSPIRV_OPT_MERGE_RETURN_PASS_H_

#include "basic_block.h"
#include "function.h"
#include "pass.h"

#include <vector>

namespace spvtools {
namespace opt {

class MergeReturnPass : public Pass {
 public:
  const char *name() const override { return "merge-return-pass"; }
  Status Process(ir::IRContext*) override;

 private:
  std::vector<ir::BasicBlock*> collectReturnBlocks(ir::Function *function);
  bool mergeReturnBlocks(ir::Function *function,
                         const std::vector<ir::BasicBlock*> &returnBlocks);
};

} // namespace opt
} // namespace spvtools

#endif // LIBSPIRV_OPT_MERGE_RETURN_PASS_H_
