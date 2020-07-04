// Copyright (c) 2020 Stefano Milizia
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS_H
#define SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS_H

#include "source/fuzz/transformation.h"

namespace spvtools {
namespace fuzz {

class TransformationRecordSynonymousConstants : public Transformation {
 public:
  explicit TransformationRecordSynonymousConstants(
      const protobufs::TransformationRecordSynonymousConstants& message);

  TransformationRecordSynonymousConstants(uint32_t constant_id,
                                          uint32_t synonym_id);

  // - |message_.constant_id| and |message_.synonym_id| are ids of constants
  // - |message_.constant_id| and |message_.synonym_id| are equivalent, i.e.
  //   both of them represent zero-like scalar values of the same type (for
  //   example OpConstant of type int and value 0 and OpConstantNull of type
  //   int), but they are the same.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationRecordSynonymousConstants message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_RECORD_SYNONYMOUS_CONSTANTS
