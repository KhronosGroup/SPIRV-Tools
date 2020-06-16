// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/fuzzer_pass_add_image_sample_unused_components.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_image_sample_unused_components.h"
#include "source/fuzz/transformation_composite_construct.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddImageSampleUnusedComponents::
    FuzzerPassAddImageSampleUnusedComponents(
        opt::IRContext* ir_context,
        TransformationContext* transformation_context,
        FuzzerContext* fuzzer_context,
        protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddImageSampleUnusedComponents::
    ~FuzzerPassAddImageSampleUnusedComponents() = default;

void FuzzerPassAddImageSampleUnusedComponents::Apply() {
  GetIRContext()->module()->ForEachInst([this](opt::Instruction* instruction) {
    if (!spvOpcodeIsImageSample(instruction->opcode())) {
      return;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()
                ->GetChanceOfAddingImageSampleUnusedComponents())) {
      return;
    }

    // Gets image sample coordinate information.
    uint32_t coordinate_id = instruction->GetSingleWordInOperand(1);
    auto coordinate_instruction =
        GetIRContext()->get_def_use_mgr()->GetDef(coordinate_id);
    auto coordinate_type = GetIRContext()->get_type_mgr()->GetType(
        coordinate_instruction->type_id());

    // If the coordinate is a 4-dimensional vector, then no unused components
    // may be added.
    if (coordinate_type->AsVector() &&
        coordinate_type->AsVector()->element_count() == 4) {
      return;
    }

    // If the coordinate is a scalar, then at most 3 unused components may be
    // added. If the coordinate is a vector, then the maximum number of unused
    // components depends on the vector size.
    uint32_t max_unused_component_count =
        coordinate_type->AsInteger() || coordinate_type->AsFloat()
            ? 3
            : 4 - coordinate_type->AsVector()->element_count();
    uint32_t unused_component_count =
        GetFuzzerContext()->GetRandomUnusedComponentCountForImageSample(
            max_unused_component_count);

    // Gets a type for the zero-unused components.
    uint32_t zero_constant_type_id;
    switch (unused_component_count) {
      case 1:
        // If the coordinate is an integer or float, then the unused components
        // type is the same as the coordinate. If the coordinate is a vector,
        // then the unused components type is the same as the vector components
        // type.
        zero_constant_type_id =
            coordinate_type->AsInteger() || coordinate_type->AsFloat()
                ? coordinate_instruction->type_id()
                : GetIRContext()->get_type_mgr()->GetId(
                      coordinate_type->AsVector()->element_type());
        break;
      case 2:
      case 3:
        // If the coordinate is an integer or float, then the unused components
        // type is the same as the coordinate. If the coordinate is a vector,
        // then the unused components type is the same as the coordinate
        // components type.
        zero_constant_type_id =
            coordinate_type->AsInteger() || coordinate_type->AsFloat()
                ? FindOrCreateVectorType(coordinate_instruction->type_id(),
                                         unused_component_count)
                : FindOrCreateVectorType(
                      GetIRContext()->get_type_mgr()->GetId(
                          coordinate_type->AsVector()->element_type()),
                      unused_component_count);
        break;
      default:
        break;
    }

    // Gets |coordinate_type| again because the module has changed.
    coordinate_type = GetIRContext()->get_type_mgr()->GetType(
        coordinate_instruction->type_id());

    // If the new vector type with unused components does not exist, then create
    // it.
    uint32_t coordinate_with_unused_components_type_id =
        coordinate_type->AsInteger() || coordinate_type->AsFloat()
            ? FindOrCreateVectorType(coordinate_instruction->type_id(),
                                     1 + unused_component_count)
            : FindOrCreateVectorType(
                  GetIRContext()->get_type_mgr()->GetId(
                      coordinate_type->AsVector()->element_type()),
                  coordinate_type->AsVector()->element_count() +
                      unused_component_count);

    // Inserts an OpCompositeConstruct instruction which
    // represents the coordinate with unused components.
    uint32_t coordinate_with_unused_components_id =
        GetFuzzerContext()->GetFreshId();
    ApplyTransformation(TransformationCompositeConstruct(
        coordinate_with_unused_components_type_id,
        {coordinate_instruction->result_id(),
         FindOrCreateZeroConstant(zero_constant_type_id)},
        MakeInstructionDescriptor(GetIRContext(), instruction),
        coordinate_with_unused_components_id));

    // Tries to add unused components to the image sample coordinate.
    ApplyTransformation(TransformationAddImageSampleUnusedComponents(
        coordinate_with_unused_components_id,
        MakeInstructionDescriptor(GetIRContext(), instruction)));
  });
}

}  // namespace fuzz
}  // namespace spvtools
