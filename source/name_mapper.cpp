// Copyright (c) 2016 Google Inc.
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

#include "name_mapper.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"

namespace {

// Converts a uint32_t to its string decimal representation.
std::string to_string(uint32_t id) {
  // Use stringstream, since some versions of Android compilers lack
  // std::to_string.
  std::stringstream os;
  os << id;
  return os.str();
}

}  // anonymous namespace

namespace libspirv {

NameMapper GetTrivialNameMapper() { return to_string; }

FriendlyNameMapper::FriendlyNameMapper(const spv_const_context context,
                                       const uint32_t* code,
                                       const size_t wordCount) {
  spv_diagnostic diag = nullptr;
  // We don't care if the parse fails.
  spvBinaryParse(context, this, code, wordCount, nullptr,
                 ParseInstructionForwarder, &diag);
  spvDiagnosticDestroy(diag);
}

std::string FriendlyNameMapper::NameForId(uint32_t id) {
  std::string result;
  auto iter = name_for_id_.find(id);
  if (iter == name_for_id_.end()) {
    // It must have been an invalid module, so just return a trivial mapping.
    // We don't care about uniqueness.
    result = to_string(id);
  } else {
    result = (*iter).second;
  }
  return result;
}

void FriendlyNameMapper::SaveName(uint32_t id,
                                  const std::string& suggested_name) {
  std::string name = suggested_name;
  auto inserted = used_names_.insert(name);
  if (!inserted.second) {
    const std::string base_name = suggested_name + "_";
    for (uint32_t index = 0; !inserted.second; index++) {
      name = base_name + to_string(index);
      inserted = used_names_.insert(name);
    }
  }
  name_for_id_[id] = name;
}

spv_result_t FriendlyNameMapper::ParseInstruction(
    const spv_parsed_instruction_t& inst) {
  if (inst.result_id) {
    switch (inst.opcode) {
      case SpvOpTypeVoid:
        SaveName(inst.result_id, "void");
        break;
      default:
        break;
    }
  }
  return SPV_SUCCESS;
}

}  // namespace libspirv
