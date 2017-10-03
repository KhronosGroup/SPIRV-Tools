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

#include "markv_model_factory.h"
#include "markv_model_shader_default.h"

namespace spvtools {

std::unique_ptr<MarkvModel> CreateMarkvModel(MarkvModelType type) {
  std::unique_ptr<MarkvModel> model;
  switch (type) {
    case kMarkvModelShaderDefault: {
      model.reset(new MarkvModelShaderDefault());
      break;
    }
  }

  model->SetModelType(static_cast<uint32_t>(type));

  return model;
}

}  // namespace spvtools
