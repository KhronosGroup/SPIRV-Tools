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

#include "build_module.h"

#include "ir_loader.h"
#include "make_unique.h"
#include "table.h"

namespace spvtools {

namespace {

// Sets the module header for IrLoader. Meets the interface requirement of
// spvBinaryParse().
spv_result_t SetSpvHeader(void* builder, spv_endianness_t, uint32_t magic,
                          uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t reserved) {
  reinterpret_cast<ir::IrLoader*>(builder)
      ->SetModuleHeader(magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
};

// Processes a parsed instruction for IrLoader. Meets the interface requirement
// of spvBinaryParse().
spv_result_t SetSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  if (reinterpret_cast<ir::IrLoader*>(builder)->AddInstruction(inst)) {
    return SPV_SUCCESS;
  }
  return SPV_ERROR_INVALID_BINARY;
};

}  // annoymous namespace

std::unique_ptr<ir::Module> BuildModule(spv_target_env env,
                                        MessageConsumer consumer,
                                        const uint32_t* binary,
                                        const size_t size) {
  auto context = spvContextCreate(env);
  SetContextMessageConsumer(context, consumer);

  auto module = MakeUnique<ir::Module>();
  ir::IrLoader loader(context->consumer, module.get());

  spv_result_t status = spvBinaryParse(context, &loader, binary, size,
                                       SetSpvHeader, SetSpvInst, nullptr);
  loader.EndModule();

  spvContextDestroy(context);

  return status == SPV_SUCCESS ? std::move(module) : nullptr;
}

std::unique_ptr<ir::Module> BuildModule(spv_target_env env,
                                        MessageConsumer consumer,
                                        const std::string& text,
                                        uint32_t assemble_options) {
  SpirvTools t(env);
  t.SetMessageConsumer(consumer);
  std::vector<uint32_t> binary;
  if (!t.Assemble(text, &binary, assemble_options)) return nullptr;
  return BuildModule(env, consumer, binary.data(), binary.size());
}

}  // namespace spvtools
