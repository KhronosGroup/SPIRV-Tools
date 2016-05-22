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

  virtual bool IsSame(const Type* that) const = 0;
  virtual std::string str() const = 0;

  virtual const Void* AsVoid() const { return nullptr; }
  virtual const Bool* AsBool() const { return nullptr; }
  virtual const Integer* AsInteger() const { return nullptr; }
  virtual const Float* AsFloat() const { return nullptr; }
  virtual const Vector* AsVector() const { return nullptr; }
  virtual const Matrix* AsMatrix() const { return nullptr; }
  virtual const Array* AsArray() const { return nullptr; }
  virtual const RuntimeArray* AsRuntimeArray() const { return nullptr; }
  virtual const Struct* AsStruct() const { return nullptr; }
  virtual const Pointer* AsPointer() const { return nullptr; }
  virtual const Function* AsFunction() const { return nullptr; }
};

class Void : public Type {
 public:
  bool IsSame(const Type* that) const override { return that->AsVoid(); }
  std::string str() const override { return "void"; }

  const Void* AsVoid() const override { return this; }
};

class Bool : public Type {
 public:
  bool IsSame(const Type* that) const override { return that->AsBool(); }
  std::string str() const override { return "bool"; }

  const Bool* AsBool() const override { return this; }
};

class Integer : public Type {
 public:
  Integer(uint32_t width, bool is_signed) : width_(width), signed_(is_signed) {}
  Integer(const Integer& that) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Integer* AsInteger() const override { return this; }

 private:
  uint32_t width_;
  bool signed_;
};

class Float : public Type {
 public:
  Float(uint32_t width) : width_(width) {}
  Float(const Float&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Float* AsFloat() const override { return this; }

 private:
  uint32_t width_;
};

class Vector : public Type {
 public:
  Vector(const Type* element_type, uint32_t count);
  Vector(const Vector&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Vector* AsVector() const override { return this; }

 private:
  const Type* element_type_;
  uint32_t count_;
};

class Matrix : public Type {
 public:
  Matrix(const Type* element_type, uint32_t count);
  Matrix(const Matrix&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Matrix* AsMatrix() const override { return this; }

 private:
  const Type* element_type_;
  uint32_t count_;
};

class Array : public Type {
 public:
  Array(const Type* element_type, uint32_t length_id);
  Array(const Array&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Array* AsArray() const override { return this; }

 private:
  const Type* element_type_;
  uint32_t length_id_;
};

class RuntimeArray : public Type {
 public:
  RuntimeArray(const Type* element_type);
  RuntimeArray(const RuntimeArray&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const RuntimeArray* AsRuntimeArray() const override { return this; }

 private:
  const Type* element_type_;
};

class Struct : public Type {
 public:
  Struct(const std::vector<const Type*>& element_types);
  Struct(const Struct&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Struct* AsStruct() const override { return this; }

 private:
  std::vector<const Type*> element_types_;
};

class Pointer : public Type {
 public:
  Pointer(const Type* pointee_type, SpvStorageClass storage_class);
  Pointer(const Pointer&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Pointer* AsPointer() const override { return this; }

 private:
  const Type* pointee_type_;
  SpvStorageClass storage_class_;
};

class Function : public Type {
 public:
  Function(const Type* return_type,
           const std::vector<const Type*>& param_types);
  Function(const Function&) = default;

  bool IsSame(const Type* that) const override;
  std::string str() const override;

  const Function* AsFunction() const override { return this; }

 private:
  const Type* return_type_;
  const std::vector<const Type*> param_types_;
};

}  // namespace type
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_TYPES_H_
