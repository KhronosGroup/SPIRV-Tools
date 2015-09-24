// Copyright (c) 2015 The Khronos Group Inc.
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

#ifndef _LIBSPIRV_UTIL_DIAGNOSTIC_H_
#define _LIBSPIRV_UTIL_DIAGNOSTIC_H_

#include <libspirv/libspirv.h>

#include <iostream>
#include <sstream>

class diagnostic_helper {
 public:
  diagnostic_helper(spv_position_t &position, spv_diagnostic *pDiagnostic)
      : position(&position), pDiagnostic(pDiagnostic) {}

  diagnostic_helper(spv_position position, spv_diagnostic *pDiagnostic)
      : position(position), pDiagnostic(pDiagnostic) {}

  ~diagnostic_helper() {
    *pDiagnostic = spvDiagnosticCreate(position, stream.str().c_str());
  }

  std::stringstream stream;

 private:
  spv_position position;
  spv_diagnostic *pDiagnostic;
};

// On destruction of the diagnostic stream, a diagnostic message will be
// written to pDiagnostic containing all of the data written to the stream.
// TODO(awoloszyn): This is very similar to diagnostic_helper, and hides
//                  the data more easily. Replace diagnostic_helper elsewhere
//                  eventually.
class DiagnosticStream {
 public:
  DiagnosticStream(spv_position position, spv_diagnostic *pDiagnostic)
      : position_(position), pDiagnostic_(pDiagnostic) {}

  DiagnosticStream(DiagnosticStream &&other) : position_(other.position_) {
    stream_.str(other.stream_.str());
    other.stream_.str("");
    pDiagnostic_ = other.pDiagnostic_;
    other.pDiagnostic_ = nullptr;
  }

  ~DiagnosticStream();

  // Adds the given value to the diagnostic message to be written.
  template <typename T>
  DiagnosticStream &operator<<(const T &val) {
    stream_ << val;
    return *this;
  }

 private:
  std::stringstream stream_;
  spv_position position_;
  spv_diagnostic *pDiagnostic_;
};

#define DIAGNOSTIC                                 \
  diagnostic_helper helper(position, pDiagnostic); \
  helper.stream

#endif
