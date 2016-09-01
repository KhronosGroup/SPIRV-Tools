// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef LIBSPIRV_DIAGNOSTIC_H_
#define LIBSPIRV_DIAGNOSTIC_H_

#include <iostream>
#include <sstream>
#include <utility>

#include "spirv-tools/libspirv.h"

namespace libspirv {

// A DiagnosticStream remembers the current position of the input and an error
// code, and captures diagnostic messages via the left-shift operator.
// If the error code is not SPV_FAILED_MATCH, then captured messages are
// emitted during the destructor.
class DiagnosticStream {
 public:
  DiagnosticStream(spv_position_t position, spv_diagnostic* pDiagnostic,
                   spv_result_t error)
      : position_(position), pDiagnostic_(pDiagnostic), error_(error) {}

  DiagnosticStream(DiagnosticStream&& other)
      : stream_(other.stream_.str()),
        position_(other.position_),
        pDiagnostic_(other.pDiagnostic_),
        error_(other.error_) {
    // The other object's destructor will emit the text in its stream_
    // member if its pDiagnostic_ member is non-null.  Prevent that,
    // since emitting that text is now the responsibility of *this.
    other.pDiagnostic_ = nullptr;
  }

  ~DiagnosticStream();

  // Adds the given value to the diagnostic message to be written.
  template <typename T>
  DiagnosticStream& operator<<(const T& val) {
    stream_ << val;
    return *this;
  }

  // Conversion operator to spv_result, returning the error code.
  operator spv_result_t() { return error_; }

 private:
  std::stringstream stream_;
  spv_position_t position_;
  spv_diagnostic* pDiagnostic_;
  spv_result_t error_;
};

std::string spvResultToString(spv_result_t res);

}  // namespace libspirv

#endif  // LIBSPIRV_DIAGNOSTIC_H_
