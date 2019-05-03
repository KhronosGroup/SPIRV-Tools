// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/fuzzer_pass_add_useful_constructs.h"
#include "source/fuzz/transformation_add_constant_boolean.h"
#include "source/fuzz/transformation_add_constant_scalar.h"
#include "source/fuzz/transformation_add_type_boolean.h"
#include "source/fuzz/transformation_add_type_float.h"
#include "source/fuzz/transformation_add_type_int.h"
#include "source/fuzz/transformation_add_type_pointer.h"

namespace spvtools {
namespace fuzz {

using opt::IRContext;

void FuzzerPassAddUsefulConstructs::MaybeAddIntConstant(
    opt::IRContext* ir_context, FuzzerContext* fuzzer_context,
    FactManager* fact_manager,
    protobufs::TransformationSequence* transformations, uint32_t width,
    bool is_signed, std::vector<uint32_t> data) const {
  opt::analysis::Integer temp_int_type(width, is_signed);
  assert(ir_context->get_type_mgr()->GetId(&temp_int_type) &&
         "int type should already be registered.");
  auto registered_int_type = ir_context->get_type_mgr()
                                 ->GetRegisteredType(&temp_int_type)
                                 ->AsInteger();
  auto int_type_id = ir_context->get_type_mgr()->GetId(registered_int_type);
  assert(int_type_id &&
         "The relevant int type should have been added to the module already.");
  opt::analysis::IntConstant int_constant(registered_int_type, data);
  if (!ir_context->get_constant_mgr()->FindConstant(&int_constant)) {
    protobufs::TransformationAddConstantScalar add_constant_int =
        transformation::MakeTransformationAddConstantScalar(
            fuzzer_context->FreshId(), int_type_id, data);
    assert(transformation::IsApplicable(add_constant_int, ir_context,
                                        *fact_manager) &&
           "Should be applicable by construction.");
    transformation::Apply(add_constant_int, ir_context, fact_manager);
    *transformations->add_transformation()->mutable_add_constant_scalar() =
        add_constant_int;
  }
}

void FuzzerPassAddUsefulConstructs::MaybeAddFloatConstant(
    opt::IRContext* ir_context, FuzzerContext* fuzzer_context,
    FactManager* fact_manager,
    protobufs::TransformationSequence* transformations, uint32_t width,
    std::vector<uint32_t> data) const {
  opt::analysis::Float temp_float_type(width);
  assert(ir_context->get_type_mgr()->GetId(&temp_float_type) &&
         "float type should already be registered.");
  auto registered_float_type = ir_context->get_type_mgr()
                                   ->GetRegisteredType(&temp_float_type)
                                   ->AsFloat();
  auto float_type_id = ir_context->get_type_mgr()->GetId(registered_float_type);
  assert(
      float_type_id &&
      "The relevant float type should have been added to the module already.");
  opt::analysis::FloatConstant float_constant(registered_float_type, data);
  if (!ir_context->get_constant_mgr()->FindConstant(&float_constant)) {
    protobufs::TransformationAddConstantScalar add_constant_int =
        transformation::MakeTransformationAddConstantScalar(
            fuzzer_context->FreshId(), float_type_id, data);
    assert(transformation::IsApplicable(add_constant_int, ir_context,
                                        *fact_manager) &&
           "Should be applicable by construction.");
    transformation::Apply(add_constant_int, ir_context, fact_manager);
    *transformations->add_transformation()->mutable_add_constant_scalar() =
        add_constant_int;
  }
}

void FuzzerPassAddUsefulConstructs::Apply(
    IRContext* ir_context, FactManager* fact_manager,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations) {
  {
    // Add boolean type if not present.
    opt::analysis::Bool temp_bool_type;
    if (!ir_context->get_type_mgr()->GetId(&temp_bool_type)) {
      protobufs::TransformationAddTypeBoolean add_type_boolean =
          transformation::MakeTransformationAddTypeBoolean(
              fuzzer_context->FreshId());
      assert(transformation::IsApplicable(add_type_boolean, ir_context,
                                          *fact_manager) &&
             "Should be applicable by construction.");
      transformation::Apply(add_type_boolean, ir_context, fact_manager);
      *transformations->add_transformation()->mutable_add_type_boolean() =
          add_type_boolean;
    }
  }

  {
    // Add signed and unsigned 32-bit integer types if not present.
    for (auto is_signed : {true, false}) {
      opt::analysis::Integer temp_int_type(32, is_signed);
      if (!ir_context->get_type_mgr()->GetId(&temp_int_type)) {
        protobufs::TransformationAddTypeInt add_type_int =
            transformation::MakeTransformationAddTypeInt(
                fuzzer_context->FreshId(), 32, is_signed);
        assert(transformation::IsApplicable(add_type_int, ir_context,
                                            *fact_manager) &&
               "Should be applicable by construction.");
        transformation::Apply(add_type_int, ir_context, fact_manager);
        *transformations->add_transformation()->mutable_add_type_int() =
            add_type_int;
      }
    }
  }

  {
    // Add 32-bit float type if not present.
    opt::analysis::Float temp_float_type(32);
    if (!ir_context->get_type_mgr()->GetId(&temp_float_type)) {
      protobufs::TransformationAddTypeFloat add_type_float =
          transformation::MakeTransformationAddTypeFloat(
              fuzzer_context->FreshId(), 32);
      assert(transformation::IsApplicable(add_type_float, ir_context,
                                          *fact_manager) &&
             "Should be applicable by construction.");
      transformation::Apply(add_type_float, ir_context, fact_manager);
      *transformations->add_transformation()->mutable_add_type_float() =
          add_type_float;
    }
  }

  // Add boolean constants true and false if not present.
  opt::analysis::Bool temp_bool_type;
  auto bool_type =
      ir_context->get_type_mgr()->GetRegisteredType(&temp_bool_type)->AsBool();
  for (auto boolean_value : {true, false}) {
    // Add OpConstantTrue/False if not already there.
    opt::analysis::BoolConstant bool_constant(bool_type, boolean_value);
    if (!ir_context->get_constant_mgr()->FindConstant(&bool_constant)) {
      protobufs::TransformationAddConstantBoolean add_constant_boolean =
          transformation::MakeTransformationAddConstantBoolean(
              fuzzer_context->FreshId(), boolean_value);
      assert(transformation::IsApplicable(add_constant_boolean, ir_context,
                                          *fact_manager) &&
             "Should be applicable by construction.");
      transformation::Apply(add_constant_boolean, ir_context, fact_manager);
      *transformations->add_transformation()->mutable_add_constant_boolean() =
          add_constant_boolean;
    }
  }

  // Add signed and unsigned 32-bit integer constants 0 and 1 if not present.
  for (auto is_signed : {true, false}) {
    for (auto value : {0u, 1u}) {
      MaybeAddIntConstant(ir_context, fuzzer_context, fact_manager,
                          transformations, 32, is_signed, {value});
    }
  }

  // Add 32-bit float constants 0.0 and 1.0 if not present.
  uint32_t uint_data[2];
  float float_data[2] = {0.0, 1.0};
  memcpy(uint_data, float_data, sizeof(float_data));
  for (unsigned int& datum : uint_data) {
    MaybeAddFloatConstant(ir_context, fuzzer_context, fact_manager,
                          transformations, 32, {datum});
  }

  // For every known-to-be-constant uniform, make sure we have instructions
  // declaring:
  // - a pointer type with uniform storage class, whose pointee type is the type
  // of the element
  // - a signed integer constant for each index required to access the element
  // - a constant for the constant value itself
  for (auto type : fact_manager->GetTypesForWhichUniformValuesAreKnown()) {
    opt::analysis::Pointer uniform_pointer(type, SpvStorageClassUniform);
    if (!ir_context->get_type_mgr()->GetId(&uniform_pointer)) {
      auto base_type_id = ir_context->get_type_mgr()->GetId(type);
      assert(base_type_id &&
             "All relevant scalar types should already have been added.");
      auto add_pointer = transformation::MakeTransformationAddTypePointer(
          fuzzer_context->FreshId(), SpvStorageClassUniform, base_type_id);
      assert(transformation::IsApplicable(add_pointer, ir_context,
                                          *fact_manager) &&
             "Should be applicable by construction.");
      transformation::Apply(add_pointer, ir_context, fact_manager);
    }
    for (const opt::analysis::ScalarConstant* constant :
         fact_manager->GetConstantsAvailableFromUniformsForType(*type)) {
      if (constant->AsIntConstant()) {
        auto int_constant = constant->AsIntConstant();
        auto int_type = int_constant->type()->AsInteger();
        MaybeAddIntConstant(ir_context, fuzzer_context, fact_manager,
                            transformations, int_type->width(),
                            int_type->IsSigned(), int_constant->words());
      } else {
        assert(constant->AsFloatConstant() &&
               "Known uniform values must be integer or floating-point.");
        auto float_constant = constant->AsFloatConstant();
        auto float_type = float_constant->type()->AsFloat();
        MaybeAddFloatConstant(ir_context, fuzzer_context, fact_manager,
                              transformations, float_type->width(),
                              float_constant->words());
      }
      for (auto& uniform_buffer_element_descriptor :
           *fact_manager->GetUniformDescriptorsForConstant(*constant)) {
        for (auto index : uniform_buffer_element_descriptor.index()) {
          MaybeAddIntConstant(ir_context, fuzzer_context, fact_manager,
                              transformations, 32, true, {index});
        }
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
