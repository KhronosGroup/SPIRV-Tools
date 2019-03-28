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

#ifndef SOURCE_FUZZ_TRANSFORMATION_H_
#define SOURCE_FUZZ_TRANSFORMATION_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class Transformation {
 public:
  Transformation() = default;

  virtual ~Transformation() = default;

  // Determines whether the transformation can be cleanly applied to the SPIR-V
  // module, such that semantics are preserved.
  virtual bool IsApplicable(opt::IRContext* context) = 0;

  // Requires that the transformation is applicable.  Applies the
  // transformation, mutating the given SPIR-V module.
  virtual void Apply(opt::IRContext* context) = 0;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_H_
