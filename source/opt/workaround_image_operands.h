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

#ifndef LIBSPIRV_OPT_STRENGTHEN_IMGAE_OPERANDS_H_
#define LIBSPIRV_OPT_STRENGTHEN_IMGAE_OPERANDS_H_

#include "pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class WorkaroundImageOperands : public Pass {
 public:
  const char* name() const override { return "workaround-image-operands"; }
  Status Process() override;

 private:
  // Returns true if the code changed.
  bool FixupOpTypeImage();
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_STRENGTHEN_IMGAE_OPERANDS_H_
