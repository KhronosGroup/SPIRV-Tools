// Copyright (c) 2017 Google Inc.
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

// An object of this class represents a specific decoration including its
// parameters (if any). Decorations are used by OpDecorate and OpMemberDecorate,
// and they describe certain properties that can be assigned to one or several
// <id>s.
//
// There are two types of Decorations:
//
// 1- Decorations that describe an attribute of a specific type <id>. For
// example: "ArrayStride" is a decoration that applies to an array type to
// specify the stride, in bytes, of the array’s elements. It must not be applied
// to anything other than an array type. Example usage:
//
//         OpDecorate  %1            ArrayStride                        4
//        Target<id>---^  Decoration Type ---^    Decoration Parameter--^
//
// 2- Decorations that describe an attribute of a member of a structure. For
// example: "Offset" decoration gives the byte offset of the member relative to
// the beginning of the structure. Example usage:
//
// OpMemberDecorate %struct                 2               Offset           2
//      Target<id>--^  struct member index--^  DecorationType--^ Parameters--^
//
// A decoration, therefore, has potentially one, two, or three components:
// 1- Decoration Type.
// 2- Decoration Parameters.
// 3- Structure member index
//
// The Decoration object will store these if present. If there are no
// parameters, the object will have an empty vector. If the decoration does not
// apply to a structure member, struct_member_index_ will be 'kInvalidMember' by
// default.
//
// Example 1: Decoration for an object<id> with no parameters:
// OpDecorate %obj Flat
//            dec_type_ = SpvDecorationFlat
//              params_ = empty vector
// struct_member_index_ = kInvalidMember
//
// Example 2: Decoration for an object<id> with two parameters:
// OpDecorate %obj LinkageAttributes "link" Import
//            dec_type_ = SpvDecorationLinkageAttributes
//              params_ = vector { link, Import }
// struct_member_index_ = kInvalidMember
//
// Example 3: Decoration for a member of a structure with one parameter:
// OpMemberDecorate %struct 2 Offset 2
//            dec_type_ = SpvDecorationOffset
//              params_ = vector { 2 }
// struct_member_index_ = 2
//
// Note that the Decoration object does not store the target <id>. It is
// possible for the same decoration to be applied to several <id>s (and they
// might be assigned using separate spir-v instructions, possibly using an
// assignment through GroupDecorate). It is important, however, that each <id>
// knows the decorations that applies to it. Therefore, the ValidationState_t
// class maps an <id> to a vector of Decoration objects that apply to the <id>.
//
class Decoration {
 public:
  enum { kInvalidMember = -1 };
  Decoration(SpvDecoration t,
             const std::vector<uint32_t>& parameters = std::vector<uint32_t>(),
             uint32_t member_index = kInvalidMember)
      : dec_type_(t), params_(parameters), struct_member_index_(member_index) {}

  void set_struct_member_index(uint32_t index) { struct_member_index_ = index; }
  int struct_member_index() { return struct_member_index_; }
  SpvDecoration dec_type() { return dec_type_; }
  std::vector<uint32_t>& params() { return params_; }

  inline bool operator==(const Decoration& rhs) const {
    return (dec_type_ == rhs.dec_type_ && params_ == rhs.params_ &&
            struct_member_index_ == rhs.struct_member_index_);
  }

 private:
  SpvDecoration dec_type_;
  std::vector<uint32_t> params_;

  // If the decoration applies to a member of a structure type, then the index
  // of the member is stored here.
  int struct_member_index_;
};

}  // namespace libspirv

#endif  /// LIBSPIRV_VAL_DECORATION_H_

