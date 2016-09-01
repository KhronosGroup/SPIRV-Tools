// Copyright (c) 2016 Google Inc.
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

#ifndef LIBSPIRV_OPT_PASS_H_
#define LIBSPIRV_OPT_PASS_H_

#include <memory>

#include "module.h"

namespace spvtools {
namespace opt {

// Abstract class of a pass. All passes should implement this abstract class
// and all analysis and transformation is done via the Process() method.
class Pass {
 public:
  // Returns a descriptive name for this pass.
  virtual const char* name() const = 0;
  // Processes the given |module| and returns true if the given |module| is
  // modified for optimization.
  virtual bool Process(ir::Module* module) = 0;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_H_
