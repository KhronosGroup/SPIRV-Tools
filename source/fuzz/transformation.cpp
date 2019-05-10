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

#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_add_constant_boolean.h"
#include "source/fuzz/transformation_add_constant_scalar.h"
#include "source/fuzz/transformation_add_dead_break.h"
#include "source/fuzz/transformation_add_type_boolean.h"
#include "source/fuzz/transformation_add_type_float.h"
#include "source/fuzz/transformation_add_type_int.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_move_block_down.h"
#include "source/fuzz/transformation_replace_boolean_constant_with_constant_binary.h"
#include "source/fuzz/transformation_replace_constant_with_uniform.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {
namespace transformation {

using protobufs::Transformation;

bool IsApplicable(const Transformation& message,
                  spvtools::opt::IRContext* context,
                  const spvtools::fuzz::FactManager& fact_manager) {
  switch (message.transformation_case()) {
    case Transformation::TransformationCase::kAddConstantBoolean:
      return transformation::IsApplicable(message.add_constant_boolean(),
                                          context, fact_manager);
    case Transformation::TransformationCase::kAddConstantScalar:
      return transformation::IsApplicable(message.add_constant_scalar(),
                                          context, fact_manager);
    case Transformation::TransformationCase::kAddDeadBreak:
      return transformation::IsApplicable(message.add_dead_break(), context,
                                          fact_manager);
    case Transformation::TransformationCase::kAddTypeBoolean:
      return transformation::IsApplicable(message.add_type_boolean(), context,
                                          fact_manager);
    case Transformation::TransformationCase::kAddTypeFloat:
      return transformation::IsApplicable(message.add_type_float(), context,
                                          fact_manager);
    case Transformation::TransformationCase::kAddTypeInt:
      return transformation::IsApplicable(message.add_type_int(), context,
                                          fact_manager);
    case Transformation::TransformationCase::kAddTypePointer:
      return transformation::IsApplicable(message.add_type_pointer(), context,
                                          fact_manager);
    case Transformation::TransformationCase::kMoveBlockDown:
      return transformation::IsApplicable(message.move_block_down(), context,
                                          fact_manager);
    case Transformation::TransformationCase::
        kReplaceBooleanConstantWithConstantBinary:
      return transformation::IsApplicable(
          message.replace_boolean_constant_with_constant_binary(), context,
          fact_manager);
    case Transformation::TransformationCase::kReplaceConstantWithUniform:
      return transformation::IsApplicable(
          message.replace_constant_with_uniform(), context, fact_manager);
    case Transformation::TransformationCase::kSplitBlock:
      return transformation::IsApplicable(message.split_block(), context,
                                          fact_manager);
    case Transformation::TRANSFORMATION_NOT_SET:
      assert(false);
      return false;
  }
  assert(false);
  return false;
}

void Apply(const Transformation& message, spvtools::opt::IRContext* context,
           spvtools::fuzz::FactManager* fact_manager) {
  switch (message.transformation_case()) {
    case Transformation::TransformationCase::kAddConstantBoolean:
      transformation::Apply(message.add_constant_boolean(), context,
                            fact_manager);
      break;
    case Transformation::TransformationCase::kAddConstantScalar:
      transformation::Apply(message.add_constant_scalar(), context,
                            fact_manager);
      break;
    case Transformation::TransformationCase::kAddDeadBreak:
      transformation::Apply(message.add_dead_break(), context, fact_manager);
      break;
    case Transformation::TransformationCase::kAddTypeBoolean:
      transformation::Apply(message.add_type_boolean(), context, fact_manager);
      break;
    case Transformation::TransformationCase::kAddTypeFloat:
      transformation::Apply(message.add_type_float(), context, fact_manager);
      break;
    case Transformation::TransformationCase::kAddTypeInt:
      transformation::Apply(message.add_type_int(), context, fact_manager);
      break;
    case Transformation::TransformationCase::kAddTypePointer:
      transformation::Apply(message.add_type_pointer(), context, fact_manager);
      break;
    case Transformation::TransformationCase::kMoveBlockDown:
      transformation::Apply(message.move_block_down(), context, fact_manager);
      break;
    case Transformation::TransformationCase::
        kReplaceBooleanConstantWithConstantBinary:
      transformation::Apply(
          message.replace_boolean_constant_with_constant_binary(), context,
          fact_manager);
      break;
    case Transformation::TransformationCase::kReplaceConstantWithUniform:
      transformation::Apply(message.replace_constant_with_uniform(), context,
                            fact_manager);
      break;
    case Transformation::TransformationCase::kSplitBlock:
      transformation::Apply(message.split_block(), context, fact_manager);
      break;
    case Transformation::TRANSFORMATION_NOT_SET:
      assert(false);
  }
}

}  // namespace transformation
}  // namespace fuzz
}  // namespace spvtools
