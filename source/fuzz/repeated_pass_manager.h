// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_REPEATED_PASS_MANAGER_
#define SOURCE_FUZZ_REPEATED_PASS_MANAGER_

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_pass.h"
#include "source/fuzz/repeated_pass_instances.h"

namespace spvtools {
namespace fuzz {

// An interface to encapsulate the manner in which the sequence of repeated
// passes that are applied during fuzzing is chosen.  An implementation of this
// interface could, for example, keep track of the history of passes that have
// been run and bias the selection of future passes according to this history.
class RepeatedPassManager {
 public:
  RepeatedPassManager(FuzzerContext* fuzzer_context,
                      RepeatedPassInstances* pass_instances);

  virtual ~RepeatedPassManager();

  // Returns the fuzzer pass instance that should be run next.
  virtual FuzzerPass* ChoosePass() = 0;

 protected:
  FuzzerContext* GetFuzzerContext() { return fuzzer_context_; }

  RepeatedPassInstances* GetPassInstances() { return pass_instances_; }

 private:
  // Provided in order to allow the pass manager to make random decisions.
  FuzzerContext* fuzzer_context_;

  // The repeated fuzzer passes that are enabled.
  RepeatedPassInstances* pass_instances_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_REPEATED_PASS_MANAGER_
