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

#ifndef LIBSPIRV_OPT_PASS_MANAGER_H_
#define LIBSPIRV_OPT_PASS_MANAGER_H_

#include <memory>
#include <ostream>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/opt/pass_token.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {

// The pass manager, responsible for tracking and running passes.
// Clients should first call AddPass() to add passes and then call Run()
// to run on a module. Passes are executed in the exact order of addition.
class PassManager {
 public:
  // Constructs a pass manager.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  PassManager()
      : consumer_(nullptr),
        print_all_stream_(nullptr),
        time_report_stream_(nullptr) {}

  // Sets the message consumer to the given |consumer|.
  void SetMessageConsumer(MessageConsumer c) { consumer_ = std::move(c); }

  // Adds an externally constructed pass.
  void AddPassToken(std::unique_ptr<PassToken> pass);
  // Uses the argument |args| to construct a pass instance of type |T|, and adds
  // the pass instance to this pass manger. The pass added will use this pass
  // manager's message consumer.
  template <typename T, typename... Args>
  void AddPassToken(Args&&... args);

  // Returns the number of passes added.
  uint32_t NumPasses() const;
  // Returns a pointer to the |index|th pass added.
  inline PassToken* GetPassToken(uint32_t index) const;

  // Returns the message consumer.
  inline const MessageConsumer& consumer() const;

  // Runs all passes on the given |module|. Returns Status::Failure if errors
  // occur when processing using one of the registered passes. All passes
  // registered after the error-reporting pass will be skipped. Returns the
  // corresponding Status::Success if processing is successful to indicate
  // whether changes are made to the module.
  Pass::Status Run(opt::IRContext* context);

  // Sets the option to print the disassembly before each pass and after the
  // last pass.   Output is written to |out| if that is not null.  No output
  // is generated if |out| is null.
  PassManager& SetPrintAll(std::ostream* out) {
    print_all_stream_ = out;
    return *this;
  }

  // Sets the option to print the resource utilization of each pass. Output is
  // written to |out| if that is not null. No output is generated if |out| is
  // null.
  PassManager& SetTimeReport(std::ostream* out) {
    time_report_stream_ = out;
    return *this;
  }

 private:
  // Consumer for messages.
  MessageConsumer consumer_;
  // A vector of passes. Order matters.
  std::vector<std::unique_ptr<PassToken>> pass_tokens_;
  // The output stream to write disassembly to before each pass, and after
  // the last pass.  If this is null, no output is generated.
  std::ostream* print_all_stream_;
  // The output stream to write the resource utilization of each pass. If this
  // is null, no output is generated.
  std::ostream* time_report_stream_;
};

inline void PassManager::AddPassToken(std::unique_ptr<PassToken> pass_token) {
  pass_tokens_.push_back(std::move(pass_token));
}

template <typename T, typename... Args>
inline void PassManager::AddPassToken(Args&&... args) {
  pass_tokens_.emplace_back(new T(std::forward<Args>(args)...));
}

inline uint32_t PassManager::NumPasses() const {
  return static_cast<uint32_t>(pass_tokens_.size());
}

inline PassToken* PassManager::GetPassToken(uint32_t index) const {
  SPIRV_ASSERT(consumer_, index < pass_tokens_.size(), "index out of bound");
  return pass_tokens_[index].get();
}

inline const MessageConsumer& PassManager::consumer() const {
  return consumer_;
}

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_MANAGER_H_
