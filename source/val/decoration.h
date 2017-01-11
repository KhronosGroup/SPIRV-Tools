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

#ifndef LIBSPIRV_VAL_DECORATION_H_
#define LIBSPIRV_VAL_DECORATION_H_

#include <unordered_map>
#include <vector>

namespace libspirv {

class Decoration {
 public:
  Decoration(SpvDecoration t, std::vector<uint32_t>& parameters,
             uint32_t member_index = -1)
      : dec_type_(t), params_(parameters), struct_member_index_(member_index) {}

  void set_struct_member_index(uint32_t index) { struct_member_index_ = index; }
  int struct_member_index() { return struct_member_index_; }
  SpvDecoration dec_type() { return dec_type_; }
  std::vector<uint32_t>& params() { return params_; }

 private:
  SpvDecoration dec_type_;
  std::vector<uint32_t> params_;

  // If the decoration applies to a member of a structure type, then the index
  // of the member is stored here.
  int struct_member_index_;
};

}  // namespace libspirv

#endif  /// LIBSPIRV_VAL_DECORATION_H_

