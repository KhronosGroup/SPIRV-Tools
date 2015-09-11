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

#include <libspirv/libspirv.h>
#include "diagnostic.h"

#include <assert.h>
#include <string.h>

#include <iostream>

// Diagnostic API

spv_diagnostic spvDiagnosticCreate(const spv_position position,
                                   const char *message) {
  spv_diagnostic diagnostic = new spv_diagnostic_t;
  if (!diagnostic) return nullptr;
  size_t length = strlen(message) + 1;
  diagnostic->error = new char[length];
  if (!diagnostic->error) {
    delete diagnostic;
    return nullptr;
  }
  diagnostic->position = *position;
  diagnostic->isTextSource = false;
  memset(diagnostic->error, 0, length);
  strncpy(diagnostic->error, message, length);
  return diagnostic;
}

void spvDiagnosticDestroy(spv_diagnostic diagnostic) {
  if (!diagnostic) return;
  if (diagnostic->error) {
    delete[] diagnostic->error;
  }
  delete diagnostic;
}

spv_result_t spvDiagnosticPrint(const spv_diagnostic diagnostic) {
  if (!diagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  if (diagnostic->isTextSource) {
    // NOTE: This is a text position
    // NOTE: add 1 to the line as editors start at line 1, we are counting new
    // line characters to start at line 0
    std::cerr << "error: " << diagnostic->position.line + 1 << ": "
              << diagnostic->position.column + 1 << ": " << diagnostic->error
              << "\n";
    return SPV_SUCCESS;
  } else {
    // NOTE: Assume this is a binary position
    std::cerr << "error: " << diagnostic->position.index << ": "
              << diagnostic->error << "\n";
    return SPV_SUCCESS;
  }

  return SPV_ERROR_INVALID_VALUE;
}
