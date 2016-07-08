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

#include <algorithm>
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

std::string FriendlyNameMapper::Sanitize(const std::string& suggested_name) {
  // Just replace invalid characters by '_'.
  std::string result;
  std::string valid =
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "_0123456789";
  std::transform(suggested_name.begin(), suggested_name.end(),
                 std::back_inserter(result), [&valid](const char c) {
                   char newchar = (std::string::npos == valid.find(c)) ? '_' : c;
                   return newchar;
                 });
  return result;
}

void FriendlyNameMapper::SaveName(uint32_t id,
                                  const std::string& suggested_name) {
  auto found_in_mapping = name_for_id_.find(id);
  if (found_in_mapping != name_for_id_.end())
    return;

  const std::string sanitized_suggested_name = Sanitize(suggested_name);
  std::string name = sanitized_suggested_name;
  auto inserted = used_names_.insert(name);
  if (!inserted.second) {
    const std::string base_name = sanitized_suggested_name + "_";
    for (uint32_t index = 0; !inserted.second; index++) {
      name = base_name + to_string(index);
      inserted = used_names_.insert(name);
    }
  }
  name_for_id_[id] = name;
}

spv_result_t FriendlyNameMapper::ParseInstruction(
    const spv_parsed_instruction_t& inst) {
  const auto result_id = inst.result_id;
  switch (inst.opcode) {
    case SpvOpName:
      SaveName(inst.words[1], reinterpret_cast<const char*>(inst.words + 2));
      break;
    case SpvOpTypeVoid:
      SaveName(result_id, "void");
      break;
    case SpvOpTypeBool:
      SaveName(result_id, "bool");
      break;
    case SpvOpTypeInt: {
      std::string signedness;
      std::string root;
      const auto bit_width = inst.words[2];
      switch (bit_width) {
        case 8:
          root = "char";
          break;
        case 16:
          root = "short";
          break;
        case 32:
          root = "int";
          break;
        case 64:
          root = "long";
          break;
        default:
          root = to_string(bit_width);
          signedness = "i";
          break;
      }
      if (0 == inst.words[3]) signedness = "u";
      SaveName(result_id, signedness + root);
    } break;
    case SpvOpTypeFloat: {
      const auto bit_width = inst.words[2];
      switch (bit_width) {
        case 16:
          SaveName(result_id, "half");
          break;
        case 32:
          SaveName(result_id, "float");
          break;
        case 64:
          SaveName(result_id, "double");
          break;
        default:
          SaveName(result_id, std::string("fp") + to_string(bit_width));
          break;
      }
    } break;
    case SpvOpTypeVector:
      SaveName(result_id, std::string("v") + to_string(inst.words[3]) +
                              NameForId(inst.words[2]));
      break;
    case SpvOpTypeMatrix:
      SaveName(result_id, std::string("mat") + to_string(inst.words[3]) +
                              NameForId(inst.words[2]));
      break;
    default:
      // If this instruction otherwise defines an Id, then save a mapping for
      // it.  This is needed to ensure uniqueness in there is an OpName with
      // string something like "1" that might collide with this result_id.
      // We should only do this if a name hasn't already been registered by some
      // previous forward reference.
      if (result_id && name_for_id_.find(result_id) == name_for_id_.end())
          SaveName(result_id, to_string(result_id));
      break;
  }
  return SPV_SUCCESS;
}

}  // namespace libspirv
