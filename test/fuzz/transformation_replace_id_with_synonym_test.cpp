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

#include "source/fuzz/transformation_replace_id_with_synonym.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(TransformationReplaceIdWithSynonymTest, IllegalTransformations) {
  // Synonym does not dominate use.
  // Id in use is not synonymous
  assert(0);
}

TEST(TransformationReplaceIdWithSynonymTest, LegalTransformations) {
  // Synonym of global constant
  // Synonym of global variable
  // Synonym of local expr
  // Synonym of local variable
  assert(0);
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
