// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

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
