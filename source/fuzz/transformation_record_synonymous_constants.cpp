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

#include "transformation_record_synonymous_constants.h"

namespace spvtools {
namespace fuzz {
TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(
        const protobufs::TransformationRecordSynonymousConstants& message)
    : message_(message) {}

// TODO
TransformationRecordSynonymousConstants::
    TransformationRecordSynonymousConstants(uint32_t /* constant_id */,
                                            uint32_t /* synonym_id */) {}

// TODO
bool TransformationRecordSynonymousConstants::IsApplicable(
    opt::IRContext* /* ir_context */,
    const TransformationContext& /* transformation_context */) const {
  return false;
}

// TODO
void TransformationRecordSynonymousConstants::Apply(
    opt::IRContext* /* ir_context */,
    TransformationContext* /* transformation_context */) const {}

// TODO
protobufs::Transformation TransformationRecordSynonymousConstants::ToMessage()
    const {
  return protobufs::Transformation();
}

}  // namespace fuzz
}  // namespace spvtools