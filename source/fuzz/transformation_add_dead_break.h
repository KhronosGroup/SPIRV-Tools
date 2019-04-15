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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_

#include "transformation.h"

namespace spvtools {
namespace fuzz {

// A transformation that turns a basic block that unconditionally branches to
// its successor into a block that potentially breaks out of a structured
// control flow construct, but in such a manner that the break cannot actually
// be taken.
class TransformationAddDeadBreak : public Transformation {
 public:
  TransformationAddDeadBreak() {}

  ~TransformationAddDeadBreak() override = default;

  // TODO comment.
  bool IsApplicable(opt::IRContext* context) override;

  // TODO comment.
  void Apply(opt::IRContext* context) override;

 private:
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_DEAD_BREAK_H_
