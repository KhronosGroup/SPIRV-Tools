// Copyright (c) 2017 Google Inc.
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

#ifndef LIBSPIRV_OPT_FEATURE_MANAGER_H_
#define LIBSPIRV_OPT_FEATURE_MANAGER_H_

#include "extensions.h"
#include "module.h"

namespace spvtools {
namespace opt {

// Tracks features enabled by a module. The IRContext has a FeatureManager.
class FeatureManager {
 public:
  FeatureManager() = default;

  // Returns true if |ext| is an enabled extension in the module.
  bool HasExtension(libspirv::Extension ext) const {
    return extensions_.Contains(ext);
  }

  // Analyzes |module| and records enabled extensions.
  void Analyze(ir::Module* module);

 private:
  // The enabled extensions.
  libspirv::ExtensionSet extensions_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_FEATURE_MANAGER_H_
