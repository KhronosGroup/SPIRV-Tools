// Copyright (c) 2017 Pierre Moreau
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

#ifndef SPIRV_TOOLS_LINKER_HPP_
#define SPIRV_TOOLS_LINKER_HPP_

#include <cstdint>

#include <memory>
#include <vector>

#include "libspirv.hpp"

namespace spvtools {

class LinkerOptions {
public:
  LinkerOptions() : createLibrary_(false) {}
  void SetCreateLibrary(bool create_library) {
    createLibrary_ = create_library;
  }
  bool GetCreateLibrary() const { return createLibrary_; }

private:
  bool createLibrary_;
};

class Linker {
 public:
  // Constructs an instance targeting the given environment |env|.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  explicit Linker(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  Linker(const Linker&) = delete;
  Linker(Linker&&) = delete;
  Linker& operator=(const Linker&) = delete;
  Linker& operator=(Linker&&) = delete;

  // Destructs this instance.
  ~Linker();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  spv_result_t Link(const std::vector<std::vector<uint32_t>>& binaries,
                    std::vector<uint32_t>& linked_binary,
                    const LinkerOptions& options) const;
  spv_result_t Link(const uint32_t* const* binaries, const size_t* binary_sizes,
                    size_t num_binaries, std::vector<uint32_t>& linked_binary,
                    const LinkerOptions& options) const;

 private:
  struct Impl;  // Opaque struct for holding the data fields used by this class.
  std::unique_ptr<Impl> impl_;  // Unique pointer to implementation data.
};

} // namespace spvtools

#endif // SPIRV_TOOLS_LINKER_HPP_
