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
  LinkerOptions()
      : createLibrary_(false),
        verifyIds_(false) {}

  // Returns whether a library or an executable should be produced by the
  // linking phase.
  //
  // All exported symbols are kept when creating a library, whereas they will
  // be removed when creating an executable.
  // The returned value will be true if creating a library, and false if
  // creating an executable.
  bool GetCreateLibrary() const { return createLibrary_; }

  // Sets whether a library or an executable should be produced.
  void SetCreateLibrary(bool create_library) {
    createLibrary_ = create_library;
  }

  // Returns whether to verify the uniqueness of the unique ids in the merged
  // context.
  bool GetVerifyIds() const { return verifyIds_; }

  // Sets whether to verify the uniqueness of the unique ids in the merged
  // context.
  void SetVerifyIds(bool verifyIds) {
    verifyIds_ = verifyIds;
  }

 private:
  bool createLibrary_;
  bool verifyIds_;
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

  // Links one or more SPIR-V modules into a new SPIR-V module. That is,
  // combine several SPIR-V modules into one, resolving link dependencies
  // between them.
  //
  // At least one binary has to be provided in |binaries|. Those binaries do
  // not have to be valid, but they should be at least parseable.
  // The functions can fail due to the following:
  // * No input modules were given;
  // * One or more of those modules were not parseable;
  // * The input modules used different addressing or memory models;
  // * The ID or global variable number limit were exceeded;
  // * Some entry points were defined multiple times;
  // * Some imported symbols did not have an exported counterpart;
  // * Possibly other reasons.
  spv_result_t Link(const std::vector<std::vector<uint32_t>>& binaries,
                    std::vector<uint32_t>& linked_binary,
                    const LinkerOptions& options = LinkerOptions()) const;
  spv_result_t Link(const uint32_t* const* binaries, const size_t* binary_sizes,
                    size_t num_binaries, std::vector<uint32_t>& linked_binary,
                    const LinkerOptions& options = LinkerOptions()) const;

 private:
  struct Impl;  // Opaque struct for holding the data fields used by this class.
  std::unique_ptr<Impl> impl_;  // Unique pointer to implementation data.
};

}  // namespace spvtools

#endif  // SPIRV_TOOLS_LINKER_HPP_
