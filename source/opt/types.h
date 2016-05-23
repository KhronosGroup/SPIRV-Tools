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

#ifndef LIBSPIRV_OPT_TYPES_H_
#define LIBSPIRV_OPT_TYPES_H_

#include <string>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/spirv.h"

namespace spvtools {
namespace opt {
namespace type {

class Void;
class Bool;
class Integer;
class Float;
class Vector;
class Matrix;
class Array;
class RuntimeArray;
class Struct;
class Pointer;
class Function;

class Type {
 public:
  virtual ~Type() {}

  void AddDecoration(std::vector<uint32_t>&& d) {
    decorations_.push_back(std::move(d));
  }
  bool HasSameDecorations(const Type* that) const;
  virtual bool IsSame(Type* that) const = 0;
  virtual std::string str() const = 0;

  virtual Void* AsVoid() { return nullptr; }
  virtual Bool* AsBool() { return nullptr; }
  virtual Integer* AsInteger() { return nullptr; }
  virtual Float* AsFloat() { return nullptr; }
  virtual Vector* AsVector() { return nullptr; }
  virtual Matrix* AsMatrix() { return nullptr; }
  virtual Array* AsArray() { return nullptr; }
  virtual RuntimeArray* AsRuntimeArray() { return nullptr; }
  virtual Struct* AsStruct() { return nullptr; }
  virtual Pointer* AsPointer() { return nullptr; }
  virtual Function* AsFunction() { return nullptr; }

 protected:
  // Decorations attached to this type. Each decoration is encoded as a vector
  // of uint32_t numbers. The first uint32_t number is the decoration value,
  // while the rest are the parameters to the decoration (if exists).
  std::vector<std::vector<uint32_t>> decorations_;
};

class Void : public Type {
 public:
  bool IsSame(Type* that) const override {
    return that->AsVoid() && HasSameDecorations(that);
  }
  std::string str() const override { return "void"; }

  Void* AsVoid() override { return this; }
};

class Bool : public Type {
 public:
  bool IsSame(Type* that) const override {
    return that->AsBool() && HasSameDecorations(that);
  }
  std::string str() const override { return "bool"; }

  Bool* AsBool() override { return this; }
};

class Integer : public Type {
 public:
  Integer(uint32_t width, bool is_signed) : width_(width), signed_(is_signed) {}
  Integer(const Integer& that) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Integer* AsInteger() override { return this; }

 private:
  uint32_t width_;
  bool signed_;
};

class Float : public Type {
 public:
  Float(uint32_t width) : width_(width) {}
  Float(const Float&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Float* AsFloat() override { return this; }

 private:
  uint32_t width_;
};

class Vector : public Type {
 public:
  Vector(Type* element_type, uint32_t count);
  Vector(const Vector&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Vector* AsVector() override { return this; }

 private:
  Type* element_type_;
  uint32_t count_;
};

class Matrix : public Type {
 public:
  Matrix(Type* element_type, uint32_t count);
  Matrix(const Matrix&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Matrix* AsMatrix() override { return this; }

 private:
  Type* element_type_;
  uint32_t count_;
};

class Array : public Type {
 public:
  Array(Type* element_type, uint32_t length_id);
  Array(const Array&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Array* AsArray() override { return this; }

 private:
  Type* element_type_;
  uint32_t length_id_;
};

class RuntimeArray : public Type {
 public:
  RuntimeArray(Type* element_type);
  RuntimeArray(const RuntimeArray&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  RuntimeArray* AsRuntimeArray() override { return this; }

 private:
  Type* element_type_;
};

class Struct : public Type {
 public:
  Struct(const std::vector<Type*>& element_types);
  Struct(const Struct&) = default;

  void AddMemeberDecoration(uint32_t index, std::vector<uint32_t>&& decoration);

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Struct* AsStruct() override { return this; }

 private:
  std::vector<Type*> element_types_;
};

class Pointer : public Type {
 public:
  Pointer(Type* pointee_type, SpvStorageClass storage_class);
  Pointer(const Pointer&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Pointer* AsPointer() override { return this; }

 private:
  Type* pointee_type_;
  SpvStorageClass storage_class_;
};

class Function : public Type {
 public:
  Function(Type* return_type, const std::vector<Type*>& param_types);
  Function(const Function&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Function* AsFunction() override { return this; }

 private:
  Type* return_type_;
  std::vector<Type*> param_types_;
};

}  // namespace type
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_TYPES_H_
