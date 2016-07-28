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

// This file provides a class hierarchy for representing SPIR-V types.

#ifndef LIBSPIRV_OPT_TYPES_H_
#define LIBSPIRV_OPT_TYPES_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"

namespace spvtools {
namespace opt {
namespace analysis {

class Void;
class Bool;
class Integer;
class Float;
class Vector;
class Matrix;
class Image;
class Sampler;
class SampledImage;
class Array;
class RuntimeArray;
class Struct;
class Opaque;
class Pointer;
class Function;
class Event;
class DeviceEvent;
class ReserveId;
class Queue;
class Pipe;
class ForwardPointer;
class PipeStorage;
class NamedBarrier;

// Abstract class for a SPIR-V type. It has a bunch of As<sublcass>() methods,
// which is used as a way to probe the actual <subclass>.
class Type {
 public:
  virtual ~Type() {}

  // Attaches a decoration directly on this type.
  void AddDecoration(std::vector<uint32_t>&& d) {
    decorations_.push_back(std::move(d));
  }
  // Returns the decorations on this type as a string.
  std::string GetDecorationStr() const;
  // Returns true if this type has exactly the same decorations as |that| type.
  bool HasSameDecorations(const Type* that) const;
  // Returns true if this type is exactly the same as |that| type, including
  // decorations.
  virtual bool IsSame(Type* that) const = 0;
  // Returns a human-readable string to represent this type.
  virtual std::string str() const = 0;

// A bunch of methods for casting this type to a given type. Returns this if the
// cast can be done, nullptr otherwise.
#define DeclareCastMethod(target) \
  virtual target* As##target() { return nullptr; }
  DeclareCastMethod(Void);
  DeclareCastMethod(Bool);
  DeclareCastMethod(Integer);
  DeclareCastMethod(Float);
  DeclareCastMethod(Vector);
  DeclareCastMethod(Matrix);
  DeclareCastMethod(Image);
  DeclareCastMethod(Sampler);
  DeclareCastMethod(SampledImage);
  DeclareCastMethod(Array);
  DeclareCastMethod(RuntimeArray);
  DeclareCastMethod(Struct);
  DeclareCastMethod(Opaque);
  DeclareCastMethod(Pointer);
  DeclareCastMethod(Function);
  DeclareCastMethod(Event);
  DeclareCastMethod(DeviceEvent);
  DeclareCastMethod(ReserveId);
  DeclareCastMethod(Queue);
  DeclareCastMethod(Pipe);
  DeclareCastMethod(ForwardPointer);
  DeclareCastMethod(PipeStorage);
  DeclareCastMethod(NamedBarrier);
#undef DeclareCastMethod

 protected:
  // Decorations attached to this type. Each decoration is encoded as a vector
  // of uint32_t numbers. The first uint32_t number is the decoration value,
  // and the rest are the parameters to the decoration (if exists).
  std::vector<std::vector<uint32_t>> decorations_;
};

class Integer : public Type {
 public:
  Integer(uint32_t width, bool is_signed) : width_(width), signed_(is_signed) {}
  Integer(const Integer&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Integer* AsInteger() override { return this; }

 private:
  uint32_t width_;  // bit width
  bool signed_;     // true if this integer is signed
};

class Float : public Type {
 public:
  Float(uint32_t width) : width_(width) {}
  Float(const Float&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Float* AsFloat() override { return this; }

 private:
  uint32_t width_;  // bit width
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

class Image : public Type {
 public:
  Image(Type* sampled_type, SpvDim dim, uint32_t depth, uint32_t arrayed,
        uint32_t ms, uint32_t sampled, SpvImageFormat format,
        SpvAccessQualifier access_qualifier = SpvAccessQualifierReadOnly);
  Image(const Image&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Image* AsImage() override { return this; }

 private:
  Type* sampled_type_;
  SpvDim dim_;
  uint32_t depth_;
  uint32_t arrayed_;
  uint32_t ms_;
  uint32_t sampled_;
  SpvImageFormat format_;
  SpvAccessQualifier access_qualifier_;
};

class SampledImage : public Type {
 public:
  SampledImage(Type* image_type) : image_type_(image_type) {}
  SampledImage(const SampledImage&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  SampledImage* AsSampledImage() override { return this; }

 private:
  Type* image_type_;
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
  // We can attach decorations to struct members and that should not affect the
  // underlying element type. So we need an extra data structure here to keep
  // track of element type decorations.
  std::unordered_map<uint32_t, std::vector<std::vector<uint32_t>>>
      element_decorations_;
};

class Opaque : public Type {
 public:
  Opaque(std::string name) : name_(std::move(name)) {}
  Opaque(const Opaque&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Opaque* AsOpaque() override { return this; }

 private:
  std::string name_;
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

class Pipe : public Type {
 public:
  Pipe(SpvAccessQualifier access_qualifier)
      : access_qualifier_(access_qualifier) {}
  Pipe(const Pipe&) = default;

  bool IsSame(Type* that) const override;
  std::string str() const override;

  Pipe* AsPipe() override { return this; }

 private:
  SpvAccessQualifier access_qualifier_;
};

class ForwardPointer : public Type {
 public:
  ForwardPointer(uint32_t id, SpvStorageClass storage_class)
      : target_id_(id), storage_class_(storage_class), pointer_(nullptr) {}
  ForwardPointer(const ForwardPointer&) = default;

  uint32_t target_id() const { return target_id_; }
  void SetTargetPointer(Pointer* pointer) { pointer_ = pointer; }

  bool IsSame(Type* that) const override;
  std::string str() const override;

  ForwardPointer* AsForwardPointer() override { return this; }

 private:
  uint32_t target_id_;
  SpvStorageClass storage_class_;
  Pointer* pointer_;
};

#define DefineParameterlessType(type, name)                \
  class type : public Type {                               \
   public:                                                 \
    type() = default;                                      \
    type(const type&) = default;                           \
                                                           \
    bool IsSame(Type* that) const override {               \
      return that->As##type() && HasSameDecorations(that); \
    }                                                      \
    std::string str() const override { return #name; }     \
                                                           \
    type* As##type() override { return this; }             \
  };
DefineParameterlessType(Void, void);
DefineParameterlessType(Bool, bool);
DefineParameterlessType(Sampler, sampler);
DefineParameterlessType(Event, event);
DefineParameterlessType(DeviceEvent, device_event);
DefineParameterlessType(ReserveId, reserve_id);
DefineParameterlessType(Queue, queue);
DefineParameterlessType(PipeStorage, pipe_storage);
DefineParameterlessType(NamedBarrier, named_barrier);
#undef DefineParameterlessType

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_TYPES_H_
