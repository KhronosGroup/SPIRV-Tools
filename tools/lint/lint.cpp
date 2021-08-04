// Copyright (c) 2021 Google LLC.
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

#include <iostream>

#include "spirv-tools/linter.hpp"
#include "tools/util/cli_consumer.h"

const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_5;

int main(int argc, const char** argv) {
  (void)argc;
  (void)argv;

  spv_target_env target_env = kDefaultEnvironment;

  spvtools::Linter linter(target_env);
  linter.SetMessageConsumer(spvtools::utils::CLIMessageConsumer);

  bool ok = linter.Run(nullptr, 0);

  return ok ? 0 : 1;
}
