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

#ifndef SOURCE_REDUCE_REDUCER_H_
#define SOURCE_REDUCE_REDUCER_H_

#include <functional>
#include <string>

#include "spirv-tools/libspirv.hpp"

#include "reduction_pass.h"

namespace spvtools {
namespace reduce {

class Reducer {
 public:
  using ErrorOrBool = std::pair<std::string, bool>;
  using InterestingFunction = std::function<bool(const std::vector<uint32_t>&)>;

  // Constructs an instance with the given target |env|, which is used to decode
  // the binary to be reduced later.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  explicit Reducer(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  Reducer(const Reducer&) = delete;
  Reducer(Reducer&&) = delete;
  Reducer& operator=(const Reducer&) = delete;
  Reducer& operator=(Reducer&&) = delete;

  // Destructs this instance.
  ~Reducer();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Sets the function that will be used to decide whether a reduced binary
  // turned out to be interesting.
  void SetInterestingFunction(InterestingFunction interesting_function);

  // Adds a reduction pass to the sequence of passes that will be iterated
  // over.
  void AddReductionPass(std::unique_ptr<ReductionPass>&& reduction_pass);

  // Reduces the given SPIR-V module |binary_out|.
  // Returns true on successful reduction.  Returns false if errors
  // occur when processing |binary_in|.
  // The reduced binary ends up in |binary_out|.
  bool Run(std::vector<uint32_t>&& binary_in, std::vector<uint32_t>& binary_out,
           spv_const_reducer_options options) const;

 private:
  struct Impl;                  // Opaque struct for holding internal data.
  std::unique_ptr<Impl> impl_;  // Unique pointer to internal data.
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCER_H_
