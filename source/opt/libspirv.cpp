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

#include "libspirv.hpp"

#include "ir_loader.h"
#include "make_unique.h"
#include "message.h"
#include "table.h"

namespace spvtools {

// Structs for holding the data members for SpvTools.
struct SpvTools::Impl {
  explicit Impl(spv_target_env env) : context(spvContextCreate(env)) {
    // The default consumer in spv_context_t is a null consumer, which provides
    // equivalent functionality (from the user's perspective) as a real consumer
    // does nothing.
  }
  ~Impl() { spvContextDestroy(context); }

  spv_context context;  // C interface context object.
};

SpvTools::SpvTools(spv_target_env env) : impl_(new Impl(env)) {}

SpvTools::~SpvTools() {}

void SpvTools::SetMessageConsumer(MessageConsumer consumer) {
  SetContextMessageConsumer(impl_->context, std::move(consumer));
}

bool SpvTools::Assemble(const std::string& text,
                        std::vector<uint32_t>* binary) const {
  spv_binary spvbinary = nullptr;
  spv_result_t status = spvTextToBinary(impl_->context, text.data(),
                                        text.size(), &spvbinary, nullptr);
  if (status == SPV_SUCCESS) {
    binary->assign(spvbinary->code, spvbinary->code + spvbinary->wordCount);
  }
  spvBinaryDestroy(spvbinary);
  return status == SPV_SUCCESS;
}

bool SpvTools::Disassemble(const std::vector<uint32_t>& binary,
                           std::string* text, uint32_t options) const {
  spv_text spvtext = nullptr;
  spv_result_t status = spvBinaryToText(
      impl_->context, binary.data(), binary.size(), options, &spvtext, nullptr);
  if (status == SPV_SUCCESS) {
    text->assign(spvtext->str, spvtext->str + spvtext->length);
  }
  spvTextDestroy(spvtext);
  return status == SPV_SUCCESS;
}

bool SpvTools::Validate(const std::vector<uint32_t>& binary) const {
  spv_const_binary_t b = {binary.data(), binary.size()};
  return spvValidate(impl_->context, &b, nullptr) == SPV_SUCCESS;
}

}  // namespace spvtools
