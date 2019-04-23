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

#include "source/fuzz/protobufs/spirvfuzz.pb.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// The base class for all metamorphic transformations.
//
// Rules for transformations
// -------------------------
//
// - Immutability: a transformation must be immutable.
// - Ability to copy and serialize: to ensure that a copy of a transformation,
//     possibly saved out to disk and read back again, is indistinguishable
//     from the original transformation, thus a transformation must depend
//     only on well-defined pieces of state, such as instruction ids.  It must
//     not rely on state such as pointers to instructions and blocks.
// - Determinism: the effect of a transformation on a module be a deterministic
//     function of the module and the transformation.  Any randomization should
//     be applied before creating the transformation, not during its
//     application.
// - Well-defined and precondition: the 'IsApplicable' method should only
//     return true if the transformation can be cleanly applied to the given
//     module, to mutate it into a valid and semantically-equivalent module, as
//     long as the module is initially valid.
// - Ability to test precondition on any valid module: 'IsApplicable' should be
//     designed so that it is safe to ask whether a transformation is
//     applicable to an arbitrary valid module.  For example, if a
//     transformation involves a block id, 'IsApplicable' should check whether
//     the module indeed has a block with that id, and return false if not.  It
//     must not assume that there is such a block.
// - Documented precondition: while the implementation of 'IsApplicable' should
//     should codify the precondition, the method should be commented in the
//     header file for a transformation with a precise English description of
//     the precondition.
// - Documented effect: while the implementation of 'Apply' should codify the
//     effect of the transformation, the method should be commented in the
//     header file for a transformation with a precise English description of
//     the effect.
class Transformation {
 public:
  Transformation() = default;

  virtual ~Transformation() = default;

  // A precondition that determines whether the transformation can be cleanly
  // applied to the SPIR-V module, such that semantics are preserved.
  // Subclasses must document the precondition in their header file using
  // precise English.
  virtual bool IsApplicable(opt::IRContext* context) = 0;

  // Requires that the transformation is applicable.  Applies the
  // transformation, mutating the given SPIR-V module.
  virtual void Apply(opt::IRContext* context) = 0;

  // Obtain a protobuf message corresponding to the transformation.
  virtual protobufs::Transformation ToMessage() = 0;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_H_
