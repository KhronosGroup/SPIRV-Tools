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

#ifndef LIBSPIRV_VAL_ID_H_
#define LIBSPIRV_VAL_ID_H_

#include <cstdint>

#include <functional>
#include <set>

#include "spirv-tools/libspirv.h"
#include "val/Function.h"

namespace libspirv {

class BasicBlock;

/// Represents a definition of any ID
///
/// This class represents the a definition of an Id in a module. This object can
/// be formed as a complete, or incomplete Id. A complete Id allows you to
/// reference all of the properties of this class and forms a fully defined
/// object in the module. The incomplete Id is defined only by its integer value
/// in a module and can only be used to search in a data structure.
class Id {
 public:
  /// This constructor creates an incomplete Id. This constructor can be used to
  /// create Id that are used to find other Ids
  explicit Id(const uint32_t result_id = 0);

  /// This constructor creates a complete Id.
  explicit Id(const spv_parsed_instruction_t* inst,
              Function* function = nullptr, BasicBlock* block = nullptr);

  /// Registers a use of the Id
  void RegisterUse(const BasicBlock* block = nullptr);

  /// returns the id of the Id
  operator uint32_t() const { return id_; }

  uint32_t id() const { return id_; }
  uint32_t type_id() const { return type_id_; }
  SpvOp opcode() const { return opcode_; }

  /// returns the Function where the id was defined. nullptr if it was defined
  /// outside of a Function
  const Function* defining_function() const { return defining_function_; }

  /// returns the BasicBlock where the id was defined. nullptr if it was defined
  /// outside of a BasicBlock
  const BasicBlock* defining_block() const { return defining_block_; }

  /// Returns the set of blocks where this Id was used
  const std::set<const BasicBlock*>& uses() const { return uses_; }

  /// The words used to define the Id
  const std::vector<uint32_t>& words() const { return words_; }

 private:
  /// The integer that identifies the Id
  uint32_t id_;

  /// The type of the Id
  uint32_t type_id_;

  /// The opcode used to define the Id
  SpvOp opcode_;

  /// The function in which the Id was defined
  Function* defining_function_;

  /// The block in which the Id was defined
  BasicBlock* defining_block_;

  /// The blocks in which the Id was used
  std::set<const BasicBlock*> uses_;

  /// The words of the instuction that defined the Id
  std::vector<uint32_t> words_;

#define OPERATOR(OP)                                     \
  friend bool operator OP(const Id& lhs, const Id& rhs); \
  friend bool operator OP(const Id& lhs, uint32_t rhs)
  OPERATOR(<);
  OPERATOR(==);
#undef OPERATOR
};

#define OPERATOR(OP)                              \
  bool operator OP(const Id& lhs, const Id& rhs); \
  bool operator OP(const Id& lhs, uint32_t rhs)

OPERATOR(<);
OPERATOR(==);
#undef OPERATOR

}  // namespace libspirv

// custom specialization of std::hash for Id
namespace std {
template <>
struct hash<libspirv::Id> {
  typedef libspirv::Id argument_type;
  typedef std::size_t result_type;
  result_type operator()(const argument_type& id) const {
    return hash<uint32_t>()(id);
  }
};
}  /// namespace std

#endif  // LIBSPIRV_VAL_ID_H_
