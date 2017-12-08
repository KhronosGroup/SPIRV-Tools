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

#ifndef LIBSPIRV_OPT_CONSTANTS_H_
#define LIBSPIRV_OPT_CONSTANTS_H_

#include <memory>
#include <utility>
#include <vector>

#include "make_unique.h"
#include "module.h"
#include "type_manager.h"
#include "types.h"

namespace spvtools {
namespace opt {
namespace analysis {

// Class hierarchy to represent the normal constants defined through
// OpConstantTrue, OpConstantFalse, OpConstant, OpConstantNull and
// OpConstantComposite instructions.
// TODO(qining): Add class for constants defined with OpConstantSampler.
class Constant;
class ScalarConstant;
class IntConstant;
class FloatConstant;
class BoolConstant;
class CompositeConstant;
class StructConstant;
class VectorConstant;
class ArrayConstant;
class NullConstant;

// Abstract class for a SPIR-V constant. It has a bunch of As<subclass> methods,
// which is used as a way to probe the actual <subclass>
class Constant {
 public:
  Constant() = delete;
  virtual ~Constant() {}

  // Make a deep copy of this constant.
  virtual std::unique_ptr<Constant> Copy() const = 0;

  // reflections
  virtual ScalarConstant* AsScalarConstant() { return nullptr; }
  virtual IntConstant* AsIntConstant() { return nullptr; }
  virtual FloatConstant* AsFloatConstant() { return nullptr; }
  virtual BoolConstant* AsBoolConstant() { return nullptr; }
  virtual CompositeConstant* AsCompositeConstant() { return nullptr; }
  virtual StructConstant* AsStructConstant() { return nullptr; }
  virtual VectorConstant* AsVectorConstant() { return nullptr; }
  virtual ArrayConstant* AsArrayConstant() { return nullptr; }
  virtual NullConstant* AsNullConstant() { return nullptr; }

  virtual const ScalarConstant* AsScalarConstant() const { return nullptr; }
  virtual const IntConstant* AsIntConstant() const { return nullptr; }
  virtual const FloatConstant* AsFloatConstant() const { return nullptr; }
  virtual const BoolConstant* AsBoolConstant() const { return nullptr; }
  virtual const CompositeConstant* AsCompositeConstant() const {
    return nullptr;
  }
  virtual const StructConstant* AsStructConstant() const { return nullptr; }
  virtual const VectorConstant* AsVectorConstant() const { return nullptr; }
  virtual const ArrayConstant* AsArrayConstant() const { return nullptr; }
  virtual const NullConstant* AsNullConstant() const { return nullptr; }

  const analysis::Type* type() const { return type_; }

 protected:
  Constant(const analysis::Type* ty) : type_(ty) {}

  // The type of this constant.
  const analysis::Type* type_;
};

// Abstract class for scalar type constants.
class ScalarConstant : public Constant {
 public:
  ScalarConstant() = delete;
  ScalarConstant* AsScalarConstant() override { return this; }
  const ScalarConstant* AsScalarConstant() const override { return this; }

  // Returns a const reference of the value of this constant in 32-bit words.
  virtual const std::vector<uint32_t>& words() const { return words_; }

 protected:
  ScalarConstant(const analysis::Type* ty, const std::vector<uint32_t>& w)
      : Constant(ty), words_(w) {}
  ScalarConstant(const analysis::Type* ty, std::vector<uint32_t>&& w)
      : Constant(ty), words_(std::move(w)) {}
  std::vector<uint32_t> words_;
};

// Integer type constant.
class IntConstant : public ScalarConstant {
 public:
  IntConstant(const analysis::Integer* ty, const std::vector<uint32_t>& w)
      : ScalarConstant(ty, w) {}
  IntConstant(const analysis::Integer* ty, std::vector<uint32_t>&& w)
      : ScalarConstant(ty, std::move(w)) {}

  IntConstant* AsIntConstant() override { return this; }
  const IntConstant* AsIntConstant() const override { return this; }

  // Make a copy of this IntConstant instance.
  std::unique_ptr<IntConstant> CopyIntConstant() const {
    return MakeUnique<IntConstant>(type_->AsInteger(), words_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyIntConstant().release());
  }
};

// Float type constant.
class FloatConstant : public ScalarConstant {
 public:
  FloatConstant(const analysis::Float* ty, const std::vector<uint32_t>& w)
      : ScalarConstant(ty, w) {}
  FloatConstant(const analysis::Float* ty, std::vector<uint32_t>&& w)
      : ScalarConstant(ty, std::move(w)) {}

  FloatConstant* AsFloatConstant() override { return this; }
  const FloatConstant* AsFloatConstant() const override { return this; }

  // Make a copy of this FloatConstant instance.
  std::unique_ptr<FloatConstant> CopyFloatConstant() const {
    return MakeUnique<FloatConstant>(type_->AsFloat(), words_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyFloatConstant().release());
  }
};

// Bool type constant.
class BoolConstant : public ScalarConstant {
 public:
  BoolConstant(const analysis::Bool* ty, bool v)
      : ScalarConstant(ty, {static_cast<uint32_t>(v)}), value_(v) {}

  BoolConstant* AsBoolConstant() override { return this; }
  const BoolConstant* AsBoolConstant() const override { return this; }

  // Make a copy of this BoolConstant instance.
  std::unique_ptr<BoolConstant> CopyBoolConstant() const {
    return MakeUnique<BoolConstant>(type_->AsBool(), value_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyBoolConstant().release());
  }

  bool value() const { return value_; }

 private:
  bool value_;
};

// Abstract class for composite constants.
class CompositeConstant : public Constant {
 public:
  CompositeConstant() = delete;
  CompositeConstant* AsCompositeConstant() override { return this; }
  const CompositeConstant* AsCompositeConstant() const override { return this; }

  // Returns a const reference of the components holded in this composite
  // constant.
  virtual const std::vector<const Constant*>& GetComponents() const {
    return components_;
  }

 protected:
  CompositeConstant(const analysis::Type* ty) : Constant(ty), components_() {}
  CompositeConstant(const analysis::Type* ty,
                    const std::vector<const Constant*>& components)
      : Constant(ty), components_(components) {}
  CompositeConstant(const analysis::Type* ty,
                    std::vector<const Constant*>&& components)
      : Constant(ty), components_(std::move(components)) {}
  std::vector<const Constant*> components_;
};

// Struct type constant.
class StructConstant : public CompositeConstant {
 public:
  StructConstant(const analysis::Struct* ty) : CompositeConstant(ty) {}
  StructConstant(const analysis::Struct* ty,
                 const std::vector<const Constant*>& components)
      : CompositeConstant(ty, components) {}
  StructConstant(const analysis::Struct* ty,
                 std::vector<const Constant*>&& components)
      : CompositeConstant(ty, std::move(components)) {}

  StructConstant* AsStructConstant() override { return this; }
  const StructConstant* AsStructConstant() const override { return this; }

  // Make a copy of this StructConstant instance.
  std::unique_ptr<StructConstant> CopyStructConstant() const {
    return MakeUnique<StructConstant>(type_->AsStruct(), components_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyStructConstant().release());
  }
};

// Vector type constant.
class VectorConstant : public CompositeConstant {
 public:
  VectorConstant(const analysis::Vector* ty)
      : CompositeConstant(ty), component_type_(ty->element_type()) {}
  VectorConstant(const analysis::Vector* ty,
                 const std::vector<const Constant*>& components)
      : CompositeConstant(ty, components),
        component_type_(ty->element_type()) {}
  VectorConstant(const analysis::Vector* ty,
                 std::vector<const Constant*>&& components)
      : CompositeConstant(ty, std::move(components)),
        component_type_(ty->element_type()) {}

  VectorConstant* AsVectorConstant() override { return this; }
  const VectorConstant* AsVectorConstant() const override { return this; }

  // Make a copy of this VectorConstant instance.
  std::unique_ptr<VectorConstant> CopyVectorConstant() const {
    auto another = MakeUnique<VectorConstant>(type_->AsVector());
    another->components_.insert(another->components_.end(), components_.begin(),
                                components_.end());
    return another;
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyVectorConstant().release());
  }

  const analysis::Type* component_type() { return component_type_; }

 private:
  const analysis::Type* component_type_;
};

// Array type constant.
class ArrayConstant : public CompositeConstant {
 public:
  ArrayConstant(const analysis::Array* ty) : CompositeConstant(ty) {}
  ArrayConstant(const analysis::Array* ty,
                const std::vector<const Constant*>& components)
      : CompositeConstant(ty, components) {}
  ArrayConstant(const analysis::Array* ty,
                std::vector<const Constant*>&& components)
      : CompositeConstant(ty, std::move(components)) {}

  ArrayConstant* AsArrayConstant() override { return this; }
  const ArrayConstant* AsArrayConstant() const override { return this; }

  // Make a copy of this ArrayConstant instance.
  std::unique_ptr<ArrayConstant> CopyArrayConstant() const {
    return MakeUnique<ArrayConstant>(type_->AsArray(), components_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyArrayConstant().release());
  }
};

// Null type constant.
class NullConstant : public Constant {
 public:
  NullConstant(const analysis::Type* ty) : Constant(ty) {}
  NullConstant* AsNullConstant() override { return this; }
  const NullConstant* AsNullConstant() const override { return this; }

  // Make a copy of this NullConstant instance.
  std::unique_ptr<NullConstant> CopyNullConstant() const {
    return MakeUnique<NullConstant>(type_);
  }
  std::unique_ptr<Constant> Copy() const override {
    return std::unique_ptr<Constant>(CopyNullConstant().release());
  }
};

class IRContext;

// This class represents a pool of constants.
class ConstantManager {
 public:
  ConstantManager(ir::IRContext* ctx) : ctx_(ctx) {}

  ir::IRContext* context() const { return ctx_; }

  // Creates a Constant instance with the given type and a vector of constant
  // defining words. Returns an unique pointer to the created Constant instance
  // if the Constant instance can be created successfully. To create scalar
  // type constants, the vector should contain the constant value in 32 bit
  // words and the given type must be of type Bool, Integer or Float. To create
  // composite type constants, the vector should contain the component ids, and
  // those component ids should have been recorded before as Normal Constants.
  // And the given type must be of type Struct, Vector or Array. When creating
  // VectorType Constant instance, the components must be scalars of the same
  // type, either Bool, Integer or Float. If any of the rules above failed, the
  // creation will fail and nullptr will be returned. If the vector is empty,
  // a NullConstant instance will be created with the given type.
  std::unique_ptr<Constant> CreateConstant(
      const Type* type,
      const std::vector<uint32_t>& literal_words_or_ids) const;

  // Creates a Constant instance to hold the constant value of the given
  // instruction. If the given instruction defines a normal constants whose
  // value is already known in the module, returns the unique pointer to the
  // created Constant instance. Otherwise does not create anything and returns a
  // nullptr.
  std::unique_ptr<Constant> CreateConstantFromInst(ir::Instruction* inst) const;

  // Creates a constant defining instruction for the given Constant instance
  // and inserts the instruction at the position specified by the given
  // instruction iterator. Returns a pointer to the created instruction if
  // succeeded, otherwise returns a null pointer. The instruction iterator
  // points to the same instruction before and after the insertion. This is the
  // only method that actually manages id creation/assignment and instruction
  // creation/insertion for a new Constant instance.
  ir::Instruction* BuildInstructionAndAddToModule(
      std::unique_ptr<analysis::Constant> c, ir::Module::inst_iterator* pos);

  // Creates an instruction with the given result id to declare a constant
  // represented by the given Constant instance. Returns an unique pointer to
  // the created instruction if the instruction can be created successfully.
  // Otherwise, returns a null pointer.
  std::unique_ptr<ir::Instruction> CreateInstruction(
      uint32_t result_id, analysis::Constant* c) const;

  // Creates an OpConstantComposite instruction with the given result id and
  // the CompositeConst instance which represents a composite constant. Returns
  // an unique pointer to the created instruction if succeeded. Otherwise
  // returns a null pointer.
  std::unique_ptr<ir::Instruction> CreateCompositeInstruction(
      uint32_t result_id, analysis::CompositeConstant* cc) const;

  // A helper function to get the result type of the given instruction. Returns
  // nullptr if the instruction does not have a type id (type id is 0).
  analysis::Type* GetType(const ir::Instruction* inst) const;

  // A helper function to get the collected normal constant with the given id.
  // Returns the pointer to the Constant instance in case it is found.
  // Otherwise, returns null pointer.
  analysis::Constant* FindRecordedConstant(uint32_t id) const;

  // A helper function to get the id of a collected constant with the pointer
  // to the Constant instance. Returns 0 in case the constant is not found.
  uint32_t FindRecordedConstant(const analysis::Constant* c) const;

  // A helper function to get a vector of Constant instances with the specified
  // ids. If can not find the Constant instance for any one of the ids, returns
  // an empty vector.
  std::vector<const analysis::Constant*> GetConstantsFromIds(
      const std::vector<uint32_t>& ids) const;

  // Records a new mapping between |inst| and |const_value|.
  // This updates the two mappings |id_to_const_val_| and |const_val_to_id_|.
  void MapConstantToInst(std::unique_ptr<analysis::Constant> const_value,
                         ir::Instruction* inst) {
    const_val_to_id_[const_value.get()] = inst->result_id();
    id_to_const_val_[inst->result_id()] = std::move(const_value);
  }

 private:
  // IR context that owns this constant manager.
  ir::IRContext* ctx_;

  // A mapping from the result ids of Normal Constants to their
  // analysis::Constant instances. All Normal Constants in the module, either
  // existing ones before optimization or the newly generated ones, should have
  // their Constant instance stored and their result id registered in this map.
  std::unordered_map<uint32_t, std::unique_ptr<analysis::Constant>>
      id_to_const_val_;

  // A mapping from the analsis::Constant instance of Normal Contants to their
  // result id in the module. This is a mirror map of id_to_const_val_. All
  // Normal Constants that defining instructions in the module should have
  // their analysis::Constant and their result id registered here.
  std::unordered_map<const analysis::Constant*, uint32_t> const_val_to_id_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_CONSTANTS_H_
