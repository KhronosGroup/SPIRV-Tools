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

#include <algorithm>
#include <cassert>
#include <sstream>

#include "types.h"

namespace spvtools {
namespace opt {
namespace type {

using U32VecVec = std::vector<std::vector<uint32_t>>;

namespace {

bool CompareTwoVectors(const U32VecVec a, const U32VecVec b) {
  const auto size = a.size();
  if (size != b.size()) return false;

  // Shortcut for common cases.
  if (size == 0 || size == 1) return a == b;

  // TODO(antiagainst): The following code, involving a shallow copying, is a
  // little complicated. Find a better way.
  std::vector<const std::vector<uint32_t> *> a_ptrs, b_ptrs;
  a_ptrs.reserve(size);
  a_ptrs.reserve(size);
  for (uint32_t i = 0; i < size; ++i) {
    a_ptrs.push_back(&a[i]);
    b_ptrs.push_back(&b[i]);
  }

  const auto cmp = [](const std::vector<uint32_t>* m,
                      const std::vector<uint32_t>* n) {
    return m->front() < n->front();
  };

  std::sort(a_ptrs.begin(), a_ptrs.end(), cmp);
  std::sort(b_ptrs.begin(), b_ptrs.end(), cmp);

  for (uint32_t i = 0; i < size; ++i) {
    if (*a_ptrs[i] != *b_ptrs[i]) return false;
  }
  return true;
}

}  // anonymous namespace

bool Type::HasSameDecorations(const Type* that) const {
  return CompareTwoVectors(decorations_, that->decorations_);
}

bool Integer::IsSame(Type* that) const {
  const Integer* it = that->AsInteger();
  return it && width_ == it->width_ && signed_ == it->signed_ &&
         HasSameDecorations(that);
}

std::string Integer::str() const {
  std::ostringstream oss;
  oss << (signed_ ? "s" : "u") << "int" << width_;
  return oss.str();
}

bool Float::IsSame(Type* that) const {
  const Float* ft = that->AsFloat();
  return ft && width_ == ft->width_ && HasSameDecorations(that);
}

std::string Float::str() const {
  std::ostringstream oss;
  oss << "float" << width_;
  return oss.str();
}

Vector::Vector(Type* type, uint32_t count)
    : element_type_(type), count_(count) {
  assert(type->AsBool() || type->AsInteger() || type->AsFloat());
}

bool Vector::IsSame(Type* that) const {
  const Vector* vt = that->AsVector();
  if (!vt) return false;
  return count_ == vt->count_ && element_type_->IsSame(vt->element_type_) &&
         HasSameDecorations(that);
}

std::string Vector::str() const {
  std::ostringstream oss;
  oss << "<" << element_type_->str() << ", " << count_ << ">";
  return oss.str();
}

Matrix::Matrix(Type* type, uint32_t count)
    : element_type_(type), count_(count) {
  assert(type->AsVector());
}

bool Matrix::IsSame(Type* that) const {
  const Matrix* mt = that->AsMatrix();
  if (!mt) return false;
  return count_ == mt->count_ && element_type_->IsSame(mt->element_type_) &&
         HasSameDecorations(that);
}

std::string Matrix::str() const {
  std::ostringstream oss;
  oss << "<" << element_type_->str() << ", " << count_ << ">";
  return oss.str();
}

Array::Array(Type* type, uint32_t length_id)
    : element_type_(type), length_id_(length_id) {
  assert(!type->AsVoid());
}

bool Array::IsSame(Type* that) const {
  const Array* at = that->AsArray();
  if (!at) return false;
  return length_id_ == at->length_id_ &&
         element_type_->IsSame(at->element_type_) && HasSameDecorations(that);
}

std::string Array::str() const {
  std::ostringstream oss;
  oss << "[" << element_type_->str() << ", id(" << length_id_ << ")]";
  return oss.str();
}

RuntimeArray::RuntimeArray(Type* type) : element_type_(type) {
  assert(!type->AsVoid());
}

bool RuntimeArray::IsSame(Type* that) const {
  const RuntimeArray* rat = that->AsRuntimeArray();
  if (!rat) return false;
  return element_type_->IsSame(rat->element_type_) && HasSameDecorations(that);
}

std::string RuntimeArray::str() const {
  std::ostringstream oss;
  oss << "[" << element_type_->str() << "]";
  return oss.str();
}

Struct::Struct(const std::vector<Type*>& types) : element_types_(types) {
  for (auto* t : types) {
    (void)t;
    assert(!t->AsVoid());
  }
}

void Struct::AddMemeberDecoration(uint32_t index,
                                  std::vector<uint32_t>&& decoration) {
  if (index >= element_types_.size()) {
    assert(0 && "index out of bound");
    return;
  }

  element_decorations_[index].push_back(std::move(decoration));
}

bool Struct::IsSame(Type* that) const {
  const Struct* st = that->AsStruct();
  if (!st) return false;
  if (element_types_.size() != st->element_types_.size()) return false;
  for (size_t i = 0; i < element_types_.size(); ++i) {
    if (!element_types_[i]->IsSame(st->element_types_[i])) return false;
  }

  if (!HasSameDecorations(that)) return false;

  const auto size = element_decorations_.size();
  if (size != st->element_decorations_.size()) return false;
  for (const auto& p : element_decorations_) {
    if (!CompareTwoVectors(p.second, st->element_decorations_.at(p.first)))
      return false;
  }
  return true;
}

std::string Struct::str() const {
  std::ostringstream oss;
  oss << "{";
  const size_t count = element_types_.size();
  for (size_t i = 0; i < count; ++i) {
    oss << element_types_[i]->str();
    if (i + 1 != count) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

Pointer::Pointer(Type* type, SpvStorageClass storage_class)
    : pointee_type_(type), storage_class_(storage_class) {
  assert(!type->AsVoid());
}

bool Pointer::IsSame(Type* that) const {
  const Pointer* pt = that->AsPointer();
  if (!pt) return false;
  if (storage_class_ != pt->storage_class_) return false;
  if (!pointee_type_->IsSame(pt->pointee_type_)) return false;
  return HasSameDecorations(that);
}

std::string Pointer::str() const { return pointee_type_->str() + "*"; }

Function::Function(Type* return_type, const std::vector<Type*>& param_types)
    : return_type_(return_type), param_types_(param_types) {
  for (auto* t : param_types) {
    (void)t;
    assert(!t->AsVoid());
  }
}

bool Function::IsSame(Type* that) const {
  const Function* ft = that->AsFunction();
  if (!ft) return false;
  if (!return_type_->IsSame(ft->return_type_)) return false;
  if (param_types_.size() != ft->param_types_.size()) return false;
  for (size_t i = 0; i < param_types_.size(); ++i) {
    if (!param_types_[i]->IsSame(ft->param_types_[i])) return false;
  }
  return HasSameDecorations(that);
}

std::string Function::str() const {
  std::ostringstream oss;
  const size_t count = param_types_.size();
  oss << "(";
  for (size_t i = 0; i < count; ++i) {
    oss << param_types_[i]->str();
    if (i + 1 != count) oss << ", ";
  }
  oss << ") -> " << return_type_->str();
  return oss.str();
}

}  // namespace type
}  // namespace opt
}  // namespace spvtools
