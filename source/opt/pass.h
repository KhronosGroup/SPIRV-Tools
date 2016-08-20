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

#ifndef LIBSPIRV_OPT_PASS_H_
#define LIBSPIRV_OPT_PASS_H_

#include <memory>

#include "message.h"
#include "module.h"

namespace spvtools {
namespace opt {

// Abstract class of a pass. All passes should implement this abstract class
// and all analysis and transformation is done via the Process() method.
class Pass {
 public:
  // Constructs a new pass with the given message consumer, which will be
  // invoked every time there is a message to be communicated to the outside.
  //
  // This pass just keeps a reference to the message consumer; so the message
  // consumer should outlive this pass.
  explicit Pass(const MessageConsumer& c) : consumer_(c) {}

  // Returns a descriptive name for this pass.
  virtual const char* name() const = 0;
  // Returns the reference to the message consumer for this pass.
  const MessageConsumer& consumer() const { return consumer_; }

  // Processes the given |module| and returns true if the given |module| is
  // modified for optimization.
  virtual bool Process(ir::Module* module) = 0;

 private:
  const MessageConsumer& consumer_;  // Message consumer.
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_PASS_H_
