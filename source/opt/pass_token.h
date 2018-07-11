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

#ifndef SOURCE_OPT_PASS_TOKEN_H_
#define SOURCE_OPT_PASS_TOKEN_H_

#include <memory>

namespace spvtools {
namespace opt {

class Pass;

class PassToken {
 public:
  virtual ~PassToken() = default;

  // Returns a descriptive name for this pass.
  //
  // NOTE: When deriving a new pass class, make sure you make the name
  // compatible with the corresponding spirv-opt command-line flag. For example,
  // if you add the flag --my-pass to spirv-opt, make this function return
  // "my-pass" (no leading hyphens).
  virtual const char* name() const = 0;

  // Returns a new pass to be executed.
  virtual std::unique_ptr<Pass> CreatePass() const = 0;

 protected:
  PassToken() = default;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PASS_TOKEN_H_
