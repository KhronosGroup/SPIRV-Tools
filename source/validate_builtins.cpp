// Copyright (c) 2017 Google Inc.
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

#include "validate.h"

#include <algorithm>
#include <string>

#include "diagnostic.h"
#include "opcode.h"
#include "val/validation_state.h"

using libspirv::Decoration;
using libspirv::DiagnosticStream;
using libspirv::Instruction;
using libspirv::ValidationState_t;

namespace { // utils

bool isIncludedIn(uint32_t elem, std::vector<uint32_t> vec) {
  return std::find(vec.begin(),vec.end(),elem) != vec.end();  
}

// Returns whether the given instruction has a BuiltIn decoration
bool isBuiltIn(Instruction inst, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(inst.id());
  return std::any_of(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type();
      });
}

// Returns the SpvBuiltIn_enum of the first BuiltIn decoration of a instruction
uint32_t getBuiltInEnum(Instruction inst, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(inst.id());
  auto it = std::find_if(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type();
      });
  assert(it != decorations.end());
  return it->params()[0];
}

// Returns the Storage class of the instruction
uint32_t getStorageClass(Instruction inst) {
  switch (inst.opcode()) {
    case SpvOpTypePointer: return inst.words()[2];
    case SpvOpTypeForwardPointer: return inst.words()[2];
    case SpvOpVariable: return inst.words()[3];
    case SpvOpGenericCastToPtrExplicit: return inst.words()[4];
    default: return SpvStorageClassMax;
  }
}

// TODO(jcaraban): how to find the entry_points in a more direct way?
// Returns the execution models of the entry points linked to the builtin
std::vector<uint32_t> getExecutionModels(
      ValidationState_t& vstate, Instruction inst) {
  std::vector<uint32_t> exec_vec;
  // Walks the entry functions in the module
  for (auto entry_func : vstate.entry_points()) {
    const auto &interfaces = vstate.entry_point_interfaces(entry_func);
    // Filtering those interfacing the built-in of interest
    if (isIncludedIn(inst.id(),interfaces)) {
      auto func_inst = vstate.FindDef(entry_func);
      assert(func_inst);
      // The entry function uses link to the entry points
      for (auto pair : func_inst->uses()) {
        auto entry_point = pair.first;
        if (entry_point->opcode() == SpvOpEntryPoint) {
          // The entry points contain the execution model
          auto exec_model = entry_point->words()[1];
          exec_vec.push_back(exec_model);
        }
      }
    }
  }
  return exec_vec;
}

// Ensures the excution model is included in the list of compatible models
spv_result_t checkExecutionModel(ValidationState_t& vstate,
      uint32_t exec_model, std::vector<uint32_t> models) {
  if (not isIncludedIn(exec_model,models)) {
    return vstate.diag(SPV_ERROR_INVALID_ID)
        << "Built-in variables are restricted to only certain"
           "EXECUTION MODELS (see e.g. Vulkan specification)";
  }
  return SPV_SUCCESS;
}

// Ensures the pair 'exec_model','storage_class' are compatible
spv_result_t checkStorageClass(ValidationState_t& vstate,
      uint32_t exec_model, uint32_t storage_class,
      std::vector<uint32_t> models, std::vector<uint32_t> storages)
{
  // If 'exec_model' is included the list of 'models'
  if (isIncludedIn(exec_model,models)) {
    // But 'storage_class' is not one of the compatible 'storages'
    if (not isIncludedIn(storage_class,storages)) {
      return vstate.diag(SPV_ERROR_INVALID_ID)
          << "Built-in variables must match the STORAGE CLASS "
             "of the built-in (see e.g. Vulkan specification)";
    }
  }
  return SPV_SUCCESS;
}

// Ensures the variable type matches the built-in type
spv_result_t checkBuiltInType(ValidationState_t& vstate, Instruction inst,
      SpvOp elem_type, SpvOp comp_type=SpvOpNop, uint32_t comp_size=0)
{
  auto error = [&](ValidationState_t& state) -> spv_result_t {
    return state.diag(SPV_ERROR_INVALID_ID)
        << "Built-in variables must match the DATA TYPE "
           "of the built-in (see e.g. Vulkan specification)";
  };
  auto *curr_inst = &inst;
  // Gets instruction type
  curr_inst = vstate.FindDef(curr_inst->type_id());
  assert(curr_inst);
  // The built-in variable must be of OpType Pointer
  if (curr_inst->opcode() != SpvOpTypePointer) {
    return error(vstate);
  }
  // The Pointer type is stored in its 3rd word
  curr_inst = vstate.FindDef(curr_inst->words()[3]);
  assert(curr_inst);
  // If the type is composed (i.e. Vector or Array)
  if (comp_type != SpvOpNop) {
    // It must match the composed type...
    if (curr_inst->opcode() != comp_type) {
      return error(vstate);
    }
    // ... and it must match the number of components / size
    uint32_t size;
    if (comp_type == SpvOpTypeVector) { // size as literal
      size = curr_inst->words()[3];
    } else if (comp_type == SpvOpTypeArray) { // size as constant
      auto *size_inst = vstate.FindDef(curr_inst->words()[3]);
      assert(size_inst);
      size = size_inst->words()[3];
      // if comp_size = 0, any size is ok
      size = (comp_size == 0) ? 0 : size;
    } else {
      assert(false);
    }
    if (size != comp_size) {
      return error(vstate);
    }
    // Gets primitive type
    curr_inst = vstate.FindDef(curr_inst->words()[2]);
    assert(curr_inst);
  }
  // Composed or not, it must match the primitive type
  if (curr_inst->opcode() != elem_type) {
    return error(vstate);
  }
  return SPV_SUCCESS;
}

} // anonymous namespace

namespace { // built-in checks

spv_result_t CheckPosition(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The Position decoration must be used only within vertex, tessellation
  // control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a vertex shader, any variable decorated with Position 
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
  
  // In a tessellation control, tessellation evaluation, or geometry shader,
  // any variable decorated with Position must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with Position must be declared as a
  // four-component vector of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 4;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPointSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The Position decoration must be used only within vertex, tessellation
  // control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a vertex shader, any variable decorated with Position 
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
  
  // In a tessellation control, tessellation evaluation, or geometry shader,
  // any variable decorated with Position must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with PointSize must be declared as a
  // scalar 32-bit floating-point value.
  SpvOp elem_type = SpvOpTypeFloat;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckClipDistance(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{  
  std::vector<uint32_t> exec_vec, stor_vec;

  // The ClipDistance decoration must be used only within vertex, fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In vertex shaders, any variable decorated with ClipDistance
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In fragment shaders, any variable decorated with ClipDistance
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In tessellation control, tessellation evaluation, or geometry shaders,
  // any variable decorated with ClipDistance must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
                SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with ClipDistance must be declared as
  // an array of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeArray;
  uint32_t comp_size = 0; // any size
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckCullDistance(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The CullDistance decoration must be used only within vertex, fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In vertex shaders, any variable decorated with CullDistance
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In fragment shaders, any variable decorated with CullDistance
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In tessellation control, tessellation evaluation, or geometry shaders,
  // any variable decorated with CullDistance must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
                SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with CullDistance must be declared as
  // an array of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeArray;
  uint32_t comp_size = 0; // any size
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckVertexId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The VertexId decoration must be used only within vertex shaders
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // VertexId requires Vertex execution and Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
    
  // VertexId must be declared as a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckInstanceId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The InstanceId decoration must be used only within vertex shaders
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // InstanceId requires Vertex execution and Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
    
  // InstanceId must be declared as a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeFloat;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPrimitiveId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // If a geometry shader is present and the fragment shader reads from
  // an input variable decorated with PrimitiveId, then the geometry
  // shader must write to an output variable decorated with PrimitiveId
  // in all execution paths.

  // TODO (jcaraban)

  // The PrimitiveId decoration must be used only within fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a tessellation control or tessellation evaluation shader,
  // any variable decorated with PrimitiveId must be declared
  // using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In a geometry shader, any variable decorated with PrimitiveId must
  // be declared using either the Input or Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In fragment shaders, any variable decorated with PrimitiveId
  // must be declared using the Input storage class ...
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
    
  // ... and either the Geometry or Tessellation capability must also be declared.

  // TODO(jcaraban): this part is covered by 'validate_instructions.cpp',
  //                 but shouldn't it be treated in the built-ins file?

  /*
  if (not vstate.HasCapability(SpvCapabilityGeometry) &&
      not vstate.HasCapability(SpvCapabilityTessellation)) {
    return vstate.diag(SPV_ERROR_INVALID_ID)
        << "In a fragment shader, any variable decorated with "
           "PrimitiveId must be declared using the Input storage "
           "class, and either the Geometry or Tessellation "
           "capability must also be declared.";
  }*/

  // Any variable decorated with PrimitiveId must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS; 
}

spv_result_t CheckInvocationId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The InvocationId decoration must be used only within
  // tessellation control and geometry shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelGeometry };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with InvocationId must be declared
  // using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with InvocationId must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLayer(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // If a vertex processing stage shader entry point’s interface includes a
  // variable decorated with Layer, it must write the same value to Layer
  // for all output vertices of a given primitive.

  // TODO ?

  // The Layer decoration must be used only with geometry and fragment shaders
  exec_vec = { SpvExecutionModelGeometry,
               SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a geometry shader, any variable decorated with Layer
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In a fragment shader, any variable decorated with Layer
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with Layer must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckViewportIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The ViewportIndex decoration must be used only within
  // geometry, and fragment shaders.
  exec_vec = { SpvExecutionModelGeometry,
               SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a geometry shader, any variable decorated with ViewportIndex
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In a fragment shader, any variable decorated with ViewportIndex
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
    
  // Any variable decorated with ViewportIndex must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckTessLevelOuter(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessLevelOuter decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a tessellation control shader, any variable decorated with
  // TessLevelOuter must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In a tessellation evaluation shader, any variable decorated with
  // TessLevelOuter must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with TessLevelOuter must be declared as
  // an array of size four, containing 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeArray;
  uint32_t comp_size = 4;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckTessLevelInner(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessLevelInner decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // In a tessellation control shader, any variable decorated with
  // TessLevelInner must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // In a tessellation evaluation shader, any variable decorated with
  // TessLevelInner must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with TessLevelOuter must be declared as
  // an array of size two, containing 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeArray;
  uint32_t comp_size = 2;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckTessCoord(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessCoord decoration must be used only within tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;
  
  // The variable decorated with TessCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with TessCoord must be declared as
  // three-component vector of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPatchVertices(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The PatchVertices decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with PatchVertices must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with PatchVertices must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckFragCoord(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The FragCoord decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with FragCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with FragCoord must be declared as
  // a four-component vector of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 4;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPointCoord(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The PointCoord decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with PointCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  //  The variable decorated with PointCoord must be declared as
  // two-component vector of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 2;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckFrontFacing(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The FrontFacing decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with FrontFacing must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with FrontFacing must be declared as a boolean.
  SpvOp elem_type = SpvOpTypeBool;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;  
}

spv_result_t CheckSampleId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SampleId decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with SampleId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with SampleId must be declared as
  // a scalar 32-bit.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;
  
  return SPV_SUCCESS;
}

spv_result_t CheckSamplePosition(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SamplePosition decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with SamplePosition must be
  // declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with SamplePosition must be declared as
  // a two-component vector of 32-bit floating-point values.
  SpvOp elem_type = SpvOpTypeFloat;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 2;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSampleMask(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SampleMask decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // Any variable decorated with SampleMask must be declared
  // using either the Input or Output storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // Any variable decorated with SampleMask must be declared as
  // an array of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeArray;
  uint32_t comp_size = 0; // Any size
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckFragDepth(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // To write to FragDepth, a shader must declare the DepthReplacing execution mode.

  // TODO (jcaraban)

  // The FragDepth decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with FragDepth must be declared
  // using the Output storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with FragDepth must be declared as
  // a scalar 32-bit floating-point value.
  SpvOp elem_type = SpvOpTypeFloat;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckHelperInvocation(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The HelperInvocation decoration must be used only within fragment shaders
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with HelperInvocation must be declared
  // using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with HelperInvocation must be declared as a boolean
  SpvOp elem_type = SpvOpTypeBool;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckNumWorkgroups(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec; 

  // The NumWorkgroups decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with NumWorkgroups must be declared
  // using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with NumWorkgroups must be declared as
  // a three-component vector of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;
  
  return SPV_SUCCESS;
}

spv_result_t CheckWorkgroupSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // If an object is decorated with the WorkgroupSize decoration,
  // this must take precedence over any execution mode set for LocalSize

  // TODO(jcaraban)

  // The WorkgroupSize decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The object decorated with WorkgroupSize must be
  // a specialization constant or a constant.
  storage_class = storage_class;
  // TODO(jcaraban)

  // The object decorated with WorkgroupSize must be declared as
  // a three-component vector of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckWorkgroupId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The WorkgroupId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with WorkgroupId must be declared
  // using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with WorkgroupId must be declared as
  // a three-component vector of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLocalInvocationId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The LocalInvocationId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with LocalInvocationId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;
    
  // The variable decorated with LocalInvocationId must be declared as
  // a three-component vector of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckGlobalInvocationId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The GlobalInvocationId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;
    
  // The variable decorated with GlobalInvocationId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with GlobalInvocationId must be declared as
  // a three-component vector of 32-bit integers.
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLocalInvocationIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The LocalInvocationIndex decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;
    
  // The variable decorated with LocalInvocationIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with GlobalInvocationId must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}
/*
spv_result_t CheckWorkDim(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // WorkDim is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;
    
  // WorkDim is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // WorkDim is a scalar 32-bit integer
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckGlobalSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // GlobalSize is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;
    
  // GlobalSize is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // GlobalSize is a three-component vector of 32-bit integers
  SpvOp elem_type = SpvOpTypeInt;
  SpvOp comp_type = SpvOpTypeVector;
  uint32_t comp_size = 3;
  if (auto error = checkBuiltInType(vstate,inst,elem_type,comp_type,comp_size))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckEnqueuedWorkgroupSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  return CheckWorkgroupSize(vstate,inst,storage_class,exec_model);
}

spv_result_t CheckGlobalOffset(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  return checkGlobalSize(vstate,inst,storage_class,exec_model);
}

spv_result_t CheckGlobalLinearId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  return CheckLocalInvocationIndex(vstate,inst,storage_class,exec_model);
}

spv_result_t CheckSubgroupSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  return CheckWorkgroupSize(vstate,inst,storage_class,exec_model);
}

spv_result_t CheckSubgroupMaxSize(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  return checkSubgroupSize(vstate,inst,storage_class,exec_model);
}

spv_result_t CheckNumSubgroups(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckNumEnqueuedSubgroups(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupLocalInvocationId(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}
*/
spv_result_t CheckVertexIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  std::vector<uint32_t> exec_vec, stor_vec;

  // The VertexIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = checkExecutionModel(vstate,exec_model,exec_vec))
    return error;

  // The variable decorated with VertexIndex must be declared
  // using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = checkStorageClass(vstate,exec_model,storage_class,exec_vec,stor_vec))
    return error;

  // The variable decorated with VertexIndex must be declared as
  // a scalar 32-bit integer.
  SpvOp elem_type = SpvOpTypeInt;
  if (auto error = checkBuiltInType(vstate,inst,elem_type))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckInstanceIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban): many built-ins show same logic, e.g. Vertex/InstanceIndex
  return CheckVertexIndex(vstate,inst,storage_class,exec_model);
}
/*
spv_result_t CheckSubgroupEqMaskKHR(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupGeMaskKHR(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupGtMaskKHR(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupLeMaskKHR(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSubgroupLtMaskKHR(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaseVertex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaseInstance(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckDrawIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckDeviceIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckViewIndex(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordNoPerspAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordNoPerspCentroidAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordNoPerspSampleAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordSmoothAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordSmoothCentroidAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordSmoothSampleAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckBaryCoordPullModelAMD(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckFragStencilRefEXT(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckViewportMaskNV(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSecondaryPositionNV(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckSecondaryViewportMaskNV(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckPositionPerViewNV(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}

spv_result_t CheckViewportMaskPerViewNV(ValidationState_t& vstate, Instruction inst,
    uint32_t storage_class, uint32_t exec_model)
{
  // TODO(jcaraban)
}
*/

}  // anonymous namespace

namespace libspirv {

#define BUILTIN_CASE(bltin) \
  case SpvBuiltIn ## bltin: \
    return Check ## bltin (_,inst,storage_class,execution_model);

// Validates that decorations have been applied properly.
spv_result_t ValidateBuiltIns(ValidationState_t& _) {
  // For every instruction decorated with a built-in...
  for (const auto& inst : _.ordered_instructions()) {
    if (not isBuiltIn(inst,_))
      continue;
    auto storage_class = getStorageClass(inst);
    auto built_in = getBuiltInEnum(inst,_);
    // For every 'execution model' affecting the built-in, check its validty
    for (auto execution_model : getExecutionModels(_,inst)) {
      switch (built_in) {
        BUILTIN_CASE( Position                    )
        BUILTIN_CASE( PointSize                   )
        BUILTIN_CASE( ClipDistance                )
        BUILTIN_CASE( CullDistance                )
        BUILTIN_CASE( VertexId                    )
        BUILTIN_CASE( InstanceId                  )
        BUILTIN_CASE( PrimitiveId                 )
        BUILTIN_CASE( InvocationId                )
        BUILTIN_CASE( Layer                       )
        BUILTIN_CASE( ViewportIndex               )
        BUILTIN_CASE( TessLevelOuter              )
        BUILTIN_CASE( TessLevelInner              )
        BUILTIN_CASE( TessCoord                   )
        BUILTIN_CASE( PatchVertices               )
        BUILTIN_CASE( FragCoord                   )
        BUILTIN_CASE( PointCoord                  )
        BUILTIN_CASE( FrontFacing                 )
        BUILTIN_CASE( SampleId                    )
        BUILTIN_CASE( SamplePosition              )
        BUILTIN_CASE( SampleMask                  )
        BUILTIN_CASE( FragDepth                   )
        BUILTIN_CASE( HelperInvocation            )
        BUILTIN_CASE( NumWorkgroups               )
        BUILTIN_CASE( WorkgroupSize               )
        BUILTIN_CASE( WorkgroupId                 )
        BUILTIN_CASE( LocalInvocationId           )
        BUILTIN_CASE( GlobalInvocationId          )
        BUILTIN_CASE( LocalInvocationIndex        )
        // TODO(jcaraban): the built-ins below are not clearly documented
        //BUILTIN_CASE( WorkDim                     )
        //BUILTIN_CASE( GlobalSize                  )
        //BUILTIN_CASE( EnqueuedWorkgroupSize       )
        //BUILTIN_CASE( GlobalOffset                )
        //BUILTIN_CASE( GlobalLinearId              )
        //BUILTIN_CASE( SubgroupSize                )
        //BUILTIN_CASE( SubgroupMaxSize             )
        //BUILTIN_CASE( NumSubgroups                )
        //BUILTIN_CASE( NumEnqueuedSubgroups        )
        //BUILTIN_CASE( SubgroupId                  )
        //BUILTIN_CASE( SubgroupLocalInvocationId   )
        BUILTIN_CASE( VertexIndex                 )
        BUILTIN_CASE( InstanceIndex               )
        //BUILTIN_CASE( SubgroupEqMaskKHR           )
        //BUILTIN_CASE( SubgroupGeMaskKHR           )
        //BUILTIN_CASE( SubgroupGtMaskKHR           )
        //BUILTIN_CASE( SubgroupLeMaskKHR           )
        //BUILTIN_CASE( SubgroupLtMaskKHR           )
        //BUILTIN_CASE( BaseVertex                  )
        //BUILTIN_CASE( BaseInstance                )
        //BUILTIN_CASE( DrawIndex                   )
        //BUILTIN_CASE( DeviceIndex                 )
        //BUILTIN_CASE( ViewIndex                   )
        //BUILTIN_CASE( BaryCoordNoPerspAMD         )
        //BUILTIN_CASE( BaryCoordNoPerspCentroidAMD )
        //BUILTIN_CASE( BaryCoordNoPerspSampleAMD   )
        //BUILTIN_CASE( BaryCoordSmoothAMD          )
        //BUILTIN_CASE( BaryCoordSmoothCentroidAMD  )
        //BUILTIN_CASE( BaryCoordSmoothSampleAMD    )
        //BUILTIN_CASE( BaryCoordPullModelAMD       )
        //BUILTIN_CASE( FragStencilRefEXT           )
        //BUILTIN_CASE( ViewportMaskNV              )
        //BUILTIN_CASE( SecondaryPositionNV         )
        //BUILTIN_CASE( SecondaryViewportMaskNV     )
        //BUILTIN_CASE( PositionPerViewNV           )
        //BUILTIN_CASE( ViewportMaskPerViewNV       )
      }
    }
  }
  return SPV_SUCCESS;
}

#undef BUILTIN_CASE

}  // namespace libspirv

