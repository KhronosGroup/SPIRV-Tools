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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>

#include "types.h"

namespace spvtools {
namespace opt {
namespace analysis {

using U32VecVec = std::vector<std::vector<uint32_t>>;

namespace {

// Returns true if the two vector of vectors are identical.
bool CompareTwoVectors(const U32VecVec a, const U32VecVec b) {
  const auto size = a.size();
  if (size != b.size()) return false;

  if (size == 0) return true;
  if (size == 1) return a.front() == b.front();

  std::vector<const std::vector<uint32_t>*> a_ptrs, b_ptrs;
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

std::string Type::GetDecorationStr() const {
  std::ostringstream oss;
  oss << "[[";
  for (const auto& decoration : decorations_) {
    oss << "(";
    for (size_t i = 0; i < decoration.size(); ++i) {
      oss << (i > 0 ? ", " : "");
      oss << decoration.at(i);
    }
    oss << ")";
  }
  oss << "]]";
  return oss.str();
}

bool Type::HasSameDecorations(const Type* that) const {
  return CompareTwoVectors(decorations_, that->decorations_);
}

bool Type::operator==(const Type& other) const {
  if (kind_ != other.kind_) return false;

  switch (kind_) {
#define DeclareKindCase(kind) \
  case k##kind:               \
    return As##kind()->IsSame(&other);
    DeclareKindCase(Void);
    DeclareKindCase(Bool);
    DeclareKindCase(Integer);
    DeclareKindCase(Float);
    DeclareKindCase(Vector);
    DeclareKindCase(Matrix);
    DeclareKindCase(Image);
    DeclareKindCase(Sampler);
    DeclareKindCase(SampledImage);
    DeclareKindCase(Array);
    DeclareKindCase(RuntimeArray);
    DeclareKindCase(Struct);
    DeclareKindCase(Opaque);
    DeclareKindCase(Pointer);
    DeclareKindCase(Function);
    DeclareKindCase(Event);
    DeclareKindCase(DeviceEvent);
    DeclareKindCase(ReserveId);
    DeclareKindCase(Queue);
    DeclareKindCase(Pipe);
    DeclareKindCase(ForwardPointer);
    DeclareKindCase(PipeStorage);
    DeclareKindCase(NamedBarrier);
#undef DeclareKindCase
    default:
      assert(false && "Unhandled type");
      return false;
  }
}

void Type::GetHashWords(std::vector<uint32_t>* words) const {
  words->push_back(kind_);
  for (auto d : decorations_) {
    for (auto w : d) {
      words->push_back(w);
    }
  }

  switch (kind_) {
#define DeclareKindCase(type)             \
  case k##type:                           \
    As##type()->GetExtraHashWords(words); \
    break;
    DeclareKindCase(Void);
    DeclareKindCase(Bool);
    DeclareKindCase(Integer);
    DeclareKindCase(Float);
    DeclareKindCase(Vector);
    DeclareKindCase(Matrix);
    DeclareKindCase(Image);
    DeclareKindCase(Sampler);
    DeclareKindCase(SampledImage);
    DeclareKindCase(Array);
    DeclareKindCase(RuntimeArray);
    DeclareKindCase(Struct);
    DeclareKindCase(Opaque);
    DeclareKindCase(Pointer);
    DeclareKindCase(Function);
    DeclareKindCase(Event);
    DeclareKindCase(DeviceEvent);
    DeclareKindCase(ReserveId);
    DeclareKindCase(Queue);
    DeclareKindCase(Pipe);
    DeclareKindCase(ForwardPointer);
    DeclareKindCase(PipeStorage);
    DeclareKindCase(NamedBarrier);
#undef DeclareKindCase
    default:
      assert(false && "Unhandled type");
      break;
  }
}

size_t Type::HashValue() const {
  std::u32string h;
  std::vector<uint32_t> words;
  GetHashWords(&words);
  for (auto w : words) {
    h.push_back(w);
  }

  return std::hash<std::u32string>()(h);
}

bool Integer::IsSame(const Type* that) const {
  const Integer* it = that->AsInteger();
  return it && width_ == it->width_ && signed_ == it->signed_ &&
         HasSameDecorations(that);
}

std::string Integer::str() const {
  std::ostringstream oss;
  oss << (signed_ ? "s" : "u") << "int" << width_;
  return oss.str();
}

void Integer::GetExtraHashWords(std::vector<uint32_t>* words) const {
  words->push_back(width_);
  words->push_back(signed_);
}

bool Float::IsSame(const Type* that) const {
  const Float* ft = that->AsFloat();
  return ft && width_ == ft->width_ && HasSameDecorations(that);
}

std::string Float::str() const {
  std::ostringstream oss;
  oss << "float" << width_;
  return oss.str();
}

void Float::GetExtraHashWords(std::vector<uint32_t>* words) const {
  words->push_back(width_);
}

Vector::Vector(Type* type, uint32_t count)
    : Type(kVector), element_type_(type), count_(count) {
  assert(type->AsBool() || type->AsInteger() || type->AsFloat());
}

bool Vector::IsSame(const Type* that) const {
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

void Vector::GetExtraHashWords(std::vector<uint32_t>* words) const {
  element_type_->GetHashWords(words);
  words->push_back(count_);
}

Matrix::Matrix(Type* type, uint32_t count)
    : Type(kMatrix), element_type_(type), count_(count) {
  assert(type->AsVector());
}

bool Matrix::IsSame(const Type* that) const {
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

void Matrix::GetExtraHashWords(std::vector<uint32_t>* words) const {
  element_type_->GetHashWords(words);
  words->push_back(count_);
}

Image::Image(Type* sampled_type, SpvDim dim, uint32_t depth, uint32_t arrayed,
             uint32_t ms, uint32_t sampled, SpvImageFormat format,
             SpvAccessQualifier access_qualifier)
    : Type(kImage),
      sampled_type_(sampled_type),
      dim_(dim),
      depth_(depth),
      arrayed_(arrayed),
      ms_(ms),
      sampled_(sampled),
      format_(format),
      access_qualifier_(access_qualifier) {
  // TODO(antiagainst): check sampled_type
}

bool Image::IsSame(const Type* that) const {
  const Image* it = that->AsImage();
  if (!it) return false;
  return dim_ == it->dim_ && depth_ == it->depth_ && arrayed_ == it->arrayed_ &&
         ms_ == it->ms_ && sampled_ == it->sampled_ && format_ == it->format_ &&
         access_qualifier_ == it->access_qualifier_ &&
         sampled_type_->IsSame(it->sampled_type_) && HasSameDecorations(that);
}

std::string Image::str() const {
  std::ostringstream oss;
  oss << "image(" << sampled_type_->str() << ", " << dim_ << ", " << depth_
      << ", " << arrayed_ << ", " << ms_ << ", " << sampled_ << ", " << format_
      << ", " << access_qualifier_ << ")";
  return oss.str();
}

void Image::GetExtraHashWords(std::vector<uint32_t>* words) const {
  sampled_type_->GetHashWords(words);
  words->push_back(dim_);
  words->push_back(depth_);
  words->push_back(arrayed_);
  words->push_back(ms_);
  words->push_back(sampled_);
  words->push_back(format_);
  words->push_back(access_qualifier_);
}

bool SampledImage::IsSame(const Type* that) const {
  const SampledImage* sit = that->AsSampledImage();
  if (!sit) return false;
  return image_type_->IsSame(sit->image_type_) && HasSameDecorations(that);
}

std::string SampledImage::str() const {
  std::ostringstream oss;
  oss << "sampled_image(" << image_type_->str() << ")";
  return oss.str();
}

void SampledImage::GetExtraHashWords(std::vector<uint32_t>* words) const {
  image_type_->GetHashWords(words);
}

Array::Array(Type* type, uint32_t length_id)
    : Type(kArray), element_type_(type), length_id_(length_id) {
  assert(!type->AsVoid());
}

bool Array::IsSame(const Type* that) const {
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

void Array::GetExtraHashWords(std::vector<uint32_t>* words) const {
  element_type_->GetHashWords(words);
  words->push_back(length_id_);
}

RuntimeArray::RuntimeArray(Type* type)
    : Type(kRuntimeArray), element_type_(type) {
  assert(!type->AsVoid());
}

bool RuntimeArray::IsSame(const Type* that) const {
  const RuntimeArray* rat = that->AsRuntimeArray();
  if (!rat) return false;
  return element_type_->IsSame(rat->element_type_) && HasSameDecorations(that);
}

std::string RuntimeArray::str() const {
  std::ostringstream oss;
  oss << "[" << element_type_->str() << "]";
  return oss.str();
}

void RuntimeArray::GetExtraHashWords(std::vector<uint32_t>* words) const {
  element_type_->GetHashWords(words);
}

Struct::Struct(const std::vector<Type*>& types)
    : Type(kStruct), element_types_(types) {
  for (auto* t : types) {
    (void)t;
    assert(!t->AsVoid());
  }
}

void Struct::AddMemberDecoration(uint32_t index,
                                 std::vector<uint32_t>&& decoration) {
  if (index >= element_types_.size()) {
    assert(0 && "index out of bound");
    return;
  }

  element_decorations_[index].push_back(std::move(decoration));
}

bool Struct::IsSame(const Type* that) const {
  const Struct* st = that->AsStruct();
  if (!st) return false;
  if (element_types_.size() != st->element_types_.size()) return false;
  const auto size = element_decorations_.size();
  if (size != st->element_decorations_.size()) return false;
  if (!HasSameDecorations(that)) return false;

  for (size_t i = 0; i < element_types_.size(); ++i) {
    if (!element_types_[i]->IsSame(st->element_types_[i])) return false;
  }
  for (const auto& p : element_decorations_) {
    if (st->element_decorations_.count(p.first) == 0) return false;
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

void Struct::GetExtraHashWords(std::vector<uint32_t>* words) const {
  for (auto t : element_types_) {
    t->GetHashWords(words);
  }
  for (auto pair : element_decorations_) {
    words->push_back(pair.first);
    for (auto d : pair.second) {
      for (auto w : d) {
        words->push_back(w);
      }
    }
  }
}

bool Opaque::IsSame(const Type* that) const {
  const Opaque* ot = that->AsOpaque();
  if (!ot) return false;
  return name_ == ot->name_ && HasSameDecorations(that);
}

std::string Opaque::str() const {
  std::ostringstream oss;
  oss << "opaque('" << name_ << "')";
  return oss.str();
}

void Opaque::GetExtraHashWords(std::vector<uint32_t>* words) const {
  for (auto c : name_) {
    words->push_back(static_cast<char32_t>(c));
  }
}

Pointer::Pointer(Type* type, SpvStorageClass storage_class)
    : Type(kPointer), pointee_type_(type), storage_class_(storage_class) {
  assert(!type->AsVoid());
}

bool Pointer::IsSame(const Type* that) const {
  const Pointer* pt = that->AsPointer();
  if (!pt) return false;
  if (storage_class_ != pt->storage_class_) return false;
  if (!pointee_type_->IsSame(pt->pointee_type_)) return false;
  return HasSameDecorations(that);
}

std::string Pointer::str() const { return pointee_type_->str() + "*"; }

void Pointer::GetExtraHashWords(std::vector<uint32_t>* words) const {
  pointee_type_->GetHashWords(words);
  words->push_back(storage_class_);
}

Function::Function(Type* return_type, const std::vector<Type*>& param_types)
    : Type(kFunction), return_type_(return_type), param_types_(param_types) {
  for (auto* t : param_types) {
    (void)t;
    assert(!t->AsVoid());
  }
}

bool Function::IsSame(const Type* that) const {
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

void Function::GetExtraHashWords(std::vector<uint32_t>* words) const {
  return_type_->GetHashWords(words);
  for (auto t : param_types_) {
    t->GetHashWords(words);
  }
}

bool Pipe::IsSame(const Type* that) const {
  const Pipe* pt = that->AsPipe();
  if (!pt) return false;
  return access_qualifier_ == pt->access_qualifier_ && HasSameDecorations(that);
}

std::string Pipe::str() const {
  std::ostringstream oss;
  oss << "pipe(" << access_qualifier_ << ")";
  return oss.str();
}

void Pipe::GetExtraHashWords(std::vector<uint32_t>* words) const {
  words->push_back(access_qualifier_);
}

bool ForwardPointer::IsSame(const Type* that) const {
  const ForwardPointer* fpt = that->AsForwardPointer();
  if (!fpt) return false;
  return target_id_ == fpt->target_id_ &&
         storage_class_ == fpt->storage_class_ && HasSameDecorations(that);
}

std::string ForwardPointer::str() const {
  std::ostringstream oss;
  oss << "forward_pointer(";
  if (pointer_ != nullptr) {
    oss << pointer_->str();
  } else {
    oss << target_id_;
  }
  oss << ")";
  return oss.str();
}

void ForwardPointer::GetExtraHashWords(std::vector<uint32_t>* words) const {
  words->push_back(target_id_);
  words->push_back(storage_class_);
  pointer_->GetHashWords(words);
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
