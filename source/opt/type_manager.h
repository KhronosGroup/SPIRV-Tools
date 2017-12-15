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

#ifndef LIBSPIRV_OPT_TYPE_MANAGER_H_
#define LIBSPIRV_OPT_TYPE_MANAGER_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "module.h"
#include "spirv-tools/libspirv.hpp"
#include "types.h"

namespace spvtools {
namespace ir {
class IRContext;
}  // namespace ir
namespace opt {
namespace analysis {

// Hashing functor.
//
// All type pointers must be non-null.
struct HashTypePointer {
  size_t operator()(const Type* type) const {
    assert(type);
    return type->HashValue();
  }
};

// Equality functor.
//
// Checks if two types pointers are the same type.
//
// All type pointers must be non-null.
struct CompareTypePointers {
  bool operator()(const Type* lhs, const Type* rhs) const {
    assert(lhs && rhs);
    return lhs->IsSame(rhs);
  }
};

// A class for managing the SPIR-V type hierarchy.
class TypeManager {
 public:
  using IdToTypeMap = std::unordered_map<uint32_t, std::unique_ptr<Type>>;

  // Constructs a type manager from the given |module|. All internal messages
  // will be communicated to the outside via the given message |consumer|.
  // This instance only keeps a reference to the |consumer|, so the |consumer|
  // should outlive this instance.
  TypeManager(const MessageConsumer& consumer, spvtools::ir::IRContext* c);

  TypeManager(const TypeManager&) = delete;
  TypeManager(TypeManager&&) = delete;
  TypeManager& operator=(const TypeManager&) = delete;
  TypeManager& operator=(TypeManager&&) = delete;

  // Returns the type for the given type |id|. Returns nullptr if the given |id|
  // does not define a type.
  Type* GetType(uint32_t id) const;
  // Returns the id for the given |type|. Returns 0 if can not find the given
  // |type|.
  uint32_t GetId(const Type* type) const;
  // Returns the number of types hold in this manager.
  size_t NumTypes() const { return id_to_type_.size(); }
  // Iterators for all types contained in this manager.
  IdToTypeMap::const_iterator begin() const { return id_to_type_.cbegin(); }
  IdToTypeMap::const_iterator end() const { return id_to_type_.cend(); }

  // Returns the forward pointer type at the given |index|.
  ForwardPointer* GetForwardPointer(uint32_t index) const;
  // Returns the number of forward pointer types hold in this manager.
  size_t NumForwardPointers() const { return forward_pointers_.size(); }

  // Returns a pair of the type and pointer to the type in |sc|.
  //
  // |id| must be a registered type.
  std::pair<Type*, std::unique_ptr<Pointer>> GetTypeAndPointerType(
      uint32_t id, SpvStorageClass sc) const;

  // Returns an id for a declaration representing |type|.
  //
  // If |type| is registered, then the registered id is returned. Otherwise,
  // this function recursively adds type and annotation instructions as
  // necessary to fully define |type|.
  uint32_t GetTypeInstruction(const Type* type);

  // Registers |id| to |type|.
  //
  // If GetId(|type|) already returns a non-zero id, the return value will be
  // unchanged.
  void RegisterType(uint32_t id, const Type& type);

  // Removes knowledge of |id| from the manager.
  //
  // If |id| is an ambiguous type the multiple ids may be registered to |id|'s
  // type (e.g. %struct1 and %struct1 might hash to the same type). In that
  // case, calling GetId() with |id|'s type will return another suitable id
  // defining that type.
  void RemoveId(uint32_t id);

 private:
  using TypeToIdMap = std::unordered_map<const Type*, uint32_t, HashTypePointer,
                                         CompareTypePointers>;
  using ForwardPointerVector = std::vector<std::unique_ptr<ForwardPointer>>;

  // Analyzes the types and decorations on types in the given |module|.
  void AnalyzeTypes(const spvtools::ir::Module& module);

  spvtools::ir::IRContext* context() { return context_; }

  // Attachs the decorations on |type| to |id|.
  void AttachDecorations(uint32_t id, const Type* type);

  // Create the annotation instruction.
  //
  // If |element| is zero, an OpDecorate is created, other an OpMemberDecorate
  // is created. The annotation is registered with the DefUseManager and the
  // DecorationManager.
  void CreateDecoration(uint32_t id, const std::vector<uint32_t>& decoration,
                        uint32_t element = 0);

  // Creates and returns a type from the given SPIR-V |inst|. Returns nullptr if
  // the given instruction is not for defining a type.
  Type* RecordIfTypeDefinition(const spvtools::ir::Instruction& inst);
  // Attaches the decoration encoded in |inst| to a type. Does nothing if the
  // given instruction is not a decoration instruction or not decorating a type.
  void AttachIfTypeDecoration(const spvtools::ir::Instruction& inst);

  const MessageConsumer& consumer_;  // Message consumer.
  spvtools::ir::IRContext* context_;
  IdToTypeMap id_to_type_;  // Mapping from ids to their type representations.
  TypeToIdMap type_to_id_;  // Mapping from types to their defining ids.
  ForwardPointerVector forward_pointers_;  // All forward pointer declarations.
  // All unresolved forward pointer declarations.
  // Refers the contents in the above vector.
  std::unordered_set<ForwardPointer*> unresolved_forward_pointers_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_TYPE_MANAGER_H_
