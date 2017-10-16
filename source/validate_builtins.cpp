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

using std::string;
using libspirv::Decoration;
using libspirv::DiagnosticStream;
using libspirv::Instruction;
using libspirv::ValidationState_t;

namespace { // utils

enum AnyValue { kAnyBits=-3, kAnySize=-2, kAnySign=-1 };
enum Signedness { kUnsigned=0, kSigned=1 };

// Struct modeling the OpTypeInt/Float/Bool "elemental" types 
struct ElemType {
  SpvOp type; // Elemental type (e.g. int, float, bool)
  int bits; // Bits number (e.g. 32, 16, 64), 0 means any number
  int sign; // Signedness (0=unsigned, 1=signed, -1=no applicable)

  ElemType(SpvOp type_, int bits_=kAnyBits, int sign_=kAnySign)
    : type(type_), bits(bits_), sign(sign_) { }
};

// Struct modeling the OpTypeVector/Array "composed" types
struct CompType {
  SpvOp type; // Composed type (e.g. vector, array)
  int size; // Number of components, 0 means any number

  CompType(SpvOp type_=SpvOpNop, int size_=kAnySize)
    : type(type_), size(size_) { }
};

// Returns the integer signedness of the target environment
int Sign(ValidationState_t& vstate) {
  switch (vstate.context()->target_env) {
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_OPENGL_4_0:
    case SPV_ENV_OPENGL_4_1:
    case SPV_ENV_OPENGL_4_2:
    case SPV_ENV_OPENGL_4_3:
    case SPV_ENV_OPENGL_4_5:
      return kSigned;
    case SPV_ENV_OPENCL_2_1:
    case SPV_ENV_OPENCL_2_2:
      return kUnsigned;
    default:
      return kAnySign;
  }
  return kAnySign;
}

// Returns true if |elem| is in the given vector
bool IsIncluded(uint32_t elem, const std::vector<uint32_t>& vec) {
  return std::find(vec.begin(), vec.end(), elem) != vec.end();  
}

// Returns true if |inst| presents a BuiltIn decoration
bool IsBuiltIn(const Instruction* inst, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(inst->id());
  return std::any_of(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type();
      });
}

// Returns the SpvBuiltIn enum of the first BuiltIn decoration of |inst|
uint32_t GetBuiltInEnum(const Instruction* inst, ValidationState_t& vstate) {
  const auto& decorations = vstate.id_decorations(inst->id());
  auto it = std::find_if(
      decorations.begin(), decorations.end(), [](const Decoration& d) {
        return SpvDecorationBuiltIn == d.dec_type();
      });
  assert(it != decorations.end());
  return it->params()[0];
}

// Returns the Storage Class of the instruction |inst|
uint32_t GetStorageClass(const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpTypePointer: return inst->word(2);
    case SpvOpTypeForwardPointer: return inst->word(2);
    case SpvOpVariable: return inst->word(3);
    case SpvOpGenericCastToPtrExplicit: return inst->word(4);
    default: return SpvStorageClassMax;
  }
}

std::vector<const Instruction*> getEntryPoints(
      ValidationState_t& vstate, const Instruction* inst) {
  std::vector<const Instruction*> entry_vec;
  // Walks the entry functions in the module
  for (auto entry_func : vstate.entry_points()) {
    const auto& interfaces = vstate.entry_point_interfaces(entry_func);
    // Filters those interfacing the instruction of interest
    if (IsIncluded(inst->id(),interfaces)) {
      auto* func_inst = vstate.FindDef(entry_func);
      assert(func_inst);
      // The entry function use-chains reveal the entry points
      for (auto pair : func_inst->uses()) {
        auto entry_point = pair.first;
        if (entry_point->opcode() == SpvOpEntryPoint) {
          entry_vec.push_back(entry_point);
        }
      }
    }
  }
  return entry_vec;
}

// Returns the execution models of the entry points interfacing |inst|
std::vector<uint32_t> getExecutionModels(
      ValidationState_t& vstate, const Instruction* inst) {
  // Gets and walks the entry points of |inst|
  auto entry_vec = getEntryPoints(vstate,inst);
  std::vector<uint32_t> model_vec;
  for (auto entry_point : entry_vec) {
    uint32_t exec_model = entry_point->word(1);
    // Accumulates the execution models
    model_vec.push_back(exec_model);
  }
  return model_vec;
}

// Returns the execution modes of the given |entry_point|
#if 0
std::vector<uint32_t> getExecutionModes(const Instruction* entry_point) {
  // Follows the use-chains to the execution modes
  std::vector<uint32_t> mode_vec;
  for (auto pair : entry_point->uses()) {
    auto mode_inst = pair.first;
    if (mode_inst->opcode() == SpvOpExecutionMode) {
      uint32_t exec_mode = mode_inst->word(2);
      // Accumulates the execution modes of the entry
      mode_vec.push_back(exec_mode);
    }
  }
  return mode_vec;
}
#endif

// Ensures |exec_model| is included in the list of compatible |models|
spv_result_t CheckExecutionModel(ValidationState_t& vstate, const string& name,
      uint32_t exec_model, const std::vector<uint32_t>& models) {
  if (not IsIncluded(exec_model, models)) {
    return vstate.diag(SPV_ERROR_INVALID_ID) << name
        << " built-in is restricted to certain EXECUTION MODELS"
           " (see SPIR-V, Vulkan, OpenGL, OpenCL specifications)";
  }
  return SPV_SUCCESS;
}

// Ensures the pair |exec_model,storage_class| are compatible
spv_result_t CheckStorageClass(ValidationState_t& vstate,
      const string& name, uint32_t exec_model, uint32_t storage_class,
      std::vector<uint32_t> models, std::vector<uint32_t> storages) {
  // If 'exec_model' is included in the list of 'models'
  if (IsIncluded(exec_model, models)) {
    // But 'storage_class' is not one of the compatible 'storages'
    if (not IsIncluded(storage_class, storages)) {
      return vstate.diag(SPV_ERROR_INVALID_ID) << name
          << " built-in is restricted to certain STORAGE CLASSES,"
             " depending on the execution model (see SPIR-V spec)";
    }
  }
  return SPV_SUCCESS;
}

// Ensures the variable type matches the built-in type, which could be an
// elemental type (e.g. int, float) or composed (e.g. 4-componets vector)
spv_result_t CheckBuiltInType(ValidationState_t& vstate,
      const string& name, const Instruction* inst,
      ElemType elem, CompType comp=CompType()) {
  // Lambda returning the personalized type error
  auto error = [&](ValidationState_t& state) -> spv_result_t {
    return state.diag(SPV_ERROR_INVALID_ID) << name
        << " built-in must match the DATA TYPE defined"
           " in the specification (see SPIR-V specification)";
  };
  // Gets instruction type
  inst = vstate.FindDef(inst->type_id());
  assert(inst);
  // The built-in variable must be of OpType Pointer
  if (inst->opcode() != SpvOpTypePointer) {
    return error(vstate);
  }
  // The type pointed to is given in word 3
  inst = vstate.FindDef(inst->word(3));
  assert(inst);
  // If the type is composed (i.e. Vector or Array)
  if (comp.type != SpvOpNop) {
    // It must match the composed type...
    if (inst->opcode() != comp.type) {
      return error(vstate);
    }
    // ... and it must match the number of components / size
    int size = kAnySize;
    if (comp.type == SpvOpTypeVector) { // size as literal
      uint32_t literal = inst->word(3);
      size = static_cast<int>(literal);
    } else if (comp.type == SpvOpTypeArray) { // size as constant
      auto* const_inst = vstate.FindDef(inst->word(3));
      assert(const_inst);
      uint32_t literal = const_inst->word(3);
      size = static_cast<int>(literal);
    } else {
      assert(false);
    }
    if (comp.size != kAnySize)
      if (size != comp.size)
        return error(vstate);
    // Gets elemental type
    inst = vstate.FindDef(inst->word(2));
    assert(inst);
  }
  // Composed or not, it must match the elemental type...
  if (inst->opcode() != elem.type) {
    return error(vstate);
  }
  // ... and it must match the number of bits...
  int bits = static_cast<int>(inst->word(2));
  if (elem.bits != kAnyBits)
    if (bits != elem.bits)
      return error(vstate);
  // ... and the sign, if applicable
  if (elem.type == SpvOpTypeInt && elem.sign != kAnySign) {
    int sign = static_cast<int>(inst->word(3));
    if (sign != elem.sign)
      return error(vstate);
  }
  return SPV_SUCCESS;
}

} // anonymous namespace

namespace { // built-in checks

spv_result_t CheckPosition(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "Position";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The Position decoration must be used only within vertex, tessellation
  // control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a vertex shader, any variable decorated with Position 
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
  
  // In a tessellation control, tessellation evaluation, or geometry shader,
  // any variable decorated with Position must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with Position must be declared as a
  // four-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPointSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "PointSize";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The Position decoration must be used only within vertex, tessellation
  // control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a vertex shader, any variable decorated with Position 
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
  
  // In a tessellation control, tessellation evaluation, or geometry shader,
  // any variable decorated with Position must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with PointSize must be declared as a
  // scalar 32-bit floating-point value.
  ElemType elem = { SpvOpTypeFloat, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckClipDistance(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{  
  const string name = "ClipDistance";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The ClipDistance decoration must be used only within vertex, fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In vertex shaders, any variable decorated with ClipDistance
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In fragment shaders, any variable decorated with ClipDistance
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In tessellation control, tessellation evaluation, or geometry shaders,
  // any variable decorated with ClipDistance must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
                SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with ClipDistance must be declared as
  // an array of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeArray };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckCullDistance(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "CullDistance";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The CullDistance decoration must be used only within vertex, fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In vertex shaders, any variable decorated with CullDistance
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In fragment shaders, any variable decorated with CullDistance
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In tessellation control, tessellation evaluation, or geometry shaders,
  // any variable decorated with CullDistance must not be declared in a
  // storage class other than Input or Output.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
                SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with CullDistance must be declared as
  // an array of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeArray };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckVertexId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "VertexId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The VertexId decoration must be used only within vertex shaders
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // VertexId requires Vertex execution and Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
    
  // VertexId must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckInstanceId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "InstanceId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The InstanceId decoration must be used only within vertex shaders
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // InstanceId requires Vertex execution and Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
    
  // InstanceId must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPrimitiveId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "PrimitiveId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // If a geometry shader is present and the fragment shader reads from
  // an input variable decorated with PrimitiveId, then the geometry
  // shader must write to an output variable decorated with PrimitiveId
  // in all execution paths.

  // TODO (jcaraban): shall we check the above? how?

  // The PrimitiveId decoration must be used only within fragment,
  // tessellation control, tessellation evaluation, and geometry shaders.
  exec_vec = { SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a tessellation control or tessellation evaluation shader,
  // any variable decorated with PrimitiveId must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In a geometry shader, any variable decorated with PrimitiveId must
  // be declared using either the Input or Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In fragment shaders, any variable decorated with PrimitiveId
  // must be declared using the Input storage class ...
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
    
  // ... and either the Geometry or Tessellation capability must also be declared.

  // TODO(jcaraban): this seems to be covered by 'validate_instructions.cpp',
  //                 should it be tested in validate_builtins.cpp instead?

  #if 0
  if (not vstate.HasCapability(SpvCapabilityGeometry) &&
      not vstate.HasCapability(SpvCapabilityTessellation)) {
    return vstate.diag(SPV_ERROR_INVALID_ID)
        << "In a fragment shader, any variable decorated with "
           "PrimitiveId must be declared using the Input storage "
           "class, and either the Geometry or Tessellation "
           "capability must also be declared.";
  }
  #endif

  // PrimitiveId must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS; 
}

spv_result_t CheckInvocationId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "InvocationId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The InvocationId decoration must be used only within
  // tessellation control and geometry shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // InvocationId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // InvocationId must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLayer(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "Layer";
  std::vector<uint32_t> exec_vec, stor_vec;

  // If a vertex processing stage shader entry point’s interface includes a
  // variable decorated with Layer, it must write the same value to Layer
  // for all output vertices of a given primitive.

  // TODO(jcaraban): can this be checked here? how?

  // Layer may also be used as output in the Vertex and Tessellation
  // Execution Models under the ShaderViewportIndexLayerNV capability.
  bool capable = vstate.HasCapability(SpvCapabilityShaderViewportIndexLayerNV);

  // The Layer decoration must be used only with geometry and fragment shaders
  exec_vec = { SpvExecutionModelGeometry,
               SpvExecutionModelFragment };
  if (capable) {
    exec_vec.push_back( SpvExecutionModelVertex );
    exec_vec.push_back( SpvExecutionModelTessellationControl );
    exec_vec.push_back( SpvExecutionModelTessellationEvaluation );
  }
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a geometry shader, any variable decorated with Layer
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  if (capable) {
    exec_vec.push_back( SpvExecutionModelVertex );
    exec_vec.push_back( SpvExecutionModelTessellationControl );
    exec_vec.push_back( SpvExecutionModelTessellationEvaluation );
  }
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In a fragment shader, any variable decorated with Layer
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Layer must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckViewportIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "ViewportIndex";
  std::vector<uint32_t> exec_vec, stor_vec;

  // ViewportIndex may also be used as output in the Vertex and Tessellation
  // Execution Models under the ShaderViewportIndexLayerNV capability.
  bool capable = vstate.HasCapability(SpvCapabilityShaderViewportIndexLayerNV);
  
  // The ViewportIndex decoration must be used only within
  // geometry, and fragment shaders.
  exec_vec = { SpvExecutionModelGeometry,
               SpvExecutionModelFragment };
  if (capable) {
    exec_vec.push_back( SpvExecutionModelVertex );
    exec_vec.push_back( SpvExecutionModelTessellationControl );
    exec_vec.push_back( SpvExecutionModelTessellationEvaluation );
  }
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a geometry shader, any variable decorated with ViewportIndex
  // must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelGeometry };
  if (capable) {
    exec_vec.push_back( SpvExecutionModelVertex );
    exec_vec.push_back( SpvExecutionModelTessellationControl );
    exec_vec.push_back( SpvExecutionModelTessellationEvaluation );
  }
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In a fragment shader, any variable decorated with ViewportIndex
  // must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
    
  // ViewportIndex must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckTessLevelOuter(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "TessLevelOuter";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessLevelOuter decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a tessellation control shader, any variable decorated with
  // TessLevelOuter must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In a tessellation evaluation shader, any variable decorated with
  // TessLevelOuter must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with TessLevelOuter must be declared as
  // an array of size four, containing 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeArray, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckTessLevelInner(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "TessLevelInner";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessLevelInner decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // In a tessellation control shader, any variable decorated with
  // TessLevelInner must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelTessellationControl };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // In a tessellation evaluation shader, any variable decorated with
  // TessLevelInner must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // Any variable decorated with TessLevelOuter must be declared as
  // an array of size two, containing 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeArray, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckTessCoord(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "TessCoord";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The TessCoord decoration must be used only within tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
  
  // TessCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // TessCoord must be declared as three-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPatchVertices(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "PatchVertices";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The PatchVertices decoration must be used only within
  // tessellation control and tessellation evaluation shaders.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // PatchVertices must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // PatchVertices must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckFragCoord(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "FragCoord";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The FragCoord decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // FragCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // FragCoord must be declared as 
  // a four-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckPointCoord(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "PointCoord";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The PointCoord decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // PointCoord must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // PointCoord must be declared as
  // two-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckFrontFacing(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "FrontFacing";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The FrontFacing decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // FrontFacing must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // FrontFacing must be declared as a boolean.
  ElemType elem = { SpvOpTypeBool };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;  
}

spv_result_t CheckSampleId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SampleId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SampleId decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // SampleId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SampleId must be declared as a scalar 32-bit.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;
  
  return SPV_SUCCESS;
}

spv_result_t CheckSamplePosition(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SamplePosition";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SamplePosition decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // SamplePosition must be
  // declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SamplePosition must be declared as
  // a two-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSampleMask(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SampleMask";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The SampleMask decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // Any variable decorated with SampleMask must be declared using either the Input or Output storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SampleMask must be declared as an array of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeArray };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckFragDepth(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "FragDepth";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // To write to FragDepth, a shader must declare the DepthReplacing execution mode.

  // TODO(jcaraban): when exactly shall we check for DepthReplacing?
  //                 when an OpStore targets the built-in variable?
  #if 0
  auto entry_vec = getEntryPoints(vstate,inst);
  for (auto entry_point : entry_vec) {
    auto mode_vec = getExecutionModes(entry_point);
    if (not IsIncluded(SpvExecutionModeDepthReplacing,mode_vec)) {
      return vstate.diag(SPV_ERROR_INVALID_ID)
          << "To write to FragDepth, a shader must declare the "
            "DepthReplacing execution mode (see Vulkan spec)";
    }
  }
  #endif

  // The FragDepth decoration must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // FragDepth must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // FragDepth must be declared as a scalar 32-bit floating-point value.
  ElemType elem = { SpvOpTypeFloat, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckHelperInvocation(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "HelperInvocation";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The HelperInvocation decoration must be used only within fragment shaders
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // HelperInvocation must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // HelperInvocation must be declared as a boolean
  ElemType elem = SpvOpTypeBool;
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckNumWorkgroups(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "NumWorkgroups";
  std::vector<uint32_t> exec_vec, stor_vec; 

  // The NumWorkgroups decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // NumWorkgroups must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // NumWorkgroups must be declared as
  // a three-component vector of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;
  
  // TODO(jcaraban): depending on SPV_ENV_VULKAN / CL, the int is signed or unsgined

  return SPV_SUCCESS;
}

spv_result_t CheckWorkgroupSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "WorkgroupSize";
  std::vector<uint32_t> exec_vec, stor_vec;

  // If an object is decorated with the WorkgroupSize decoration,
  // this must take precedence over any execution mode set for LocalSize

  // TODO(jcaraban): can this be verified now?
  //                 i.e. is this a compile or runtime restriction?

  // The WorkgroupSize decoration must be used only within compute shaders
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // The object decorated with WorkgroupSize must be
  // a specialization constant or a constant.
  if (inst->opcode() != SpvOpConstantComposite &&
      inst->opcode() != SpvOpSpecConstantComposite) {
    return vstate.diag(SPV_ERROR_INVALID_ID)
        << "The object decorated with WorkgroupSize must be "
           "a specialization constant or a constant";
  }

  // WrokgroupSize should not have storage class, and if it does,
  // other validation should complain before reaching this line
  assert(storage_class == SpvStorageClassMax);
  
  // The object decorated with WorkgroupSize must be declared as
  // a three-component vector of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckWorkgroupId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "WorkgroupId";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The WorkgroupId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // WorkgroupId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // WorkgroupId must be declared as
  // a three-component vector of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLocalInvocationId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "LocalInvocationId";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The LocalInvocationId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // LocalInvocationId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;
    
  // LocalInvocationId must be declared as
  // a three-component vector of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckGlobalInvocationId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "GlobalInvocationId";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The GlobalInvocationId decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // GlobalInvocationId must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // GlobalInvocationId must be declared as
  // a three-component vector of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckLocalInvocationIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "LocalInvocationIndex";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The LocalInvocationIndex decoration must be used only within compute shaders.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // LocalInvocationIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelGLCompute,
               SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // GlobalInvocationId must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32, Sign(vstate) };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckWorkDim(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "WorkDim";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // WorkDim is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // WorkDim is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // WorkDim is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckGlobalSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "GlobalSize";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // GlobalSize is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // GlobalSize is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // GlobalSize is a three-component vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckEnqueuedWorkgroupSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "EnqueuedWorkgroupSize";
  std::vector<uint32_t> exec_vec, stor_vec;

  // EnqueuedWorkgroupSize is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // EnqueuedWorkgroupSize is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // EnqueuedWorkgroupSize is a three-component vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckGlobalOffset(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "GlobalOffset";
  std::vector<uint32_t> exec_vec, stor_vec;
  
    // GlobalOffset is part of the OpenCL specification
    exec_vec = { SpvExecutionModelKernel };
    if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
      return error;
      
    // GlobalOffset is an input argument to the kernels
    exec_vec = { SpvExecutionModelKernel };
    stor_vec = { SpvStorageClassInput };
    if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
      return error;
  
    // GlobalOffset is a three-component vector of 32-bit integers
    ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
    CompType comp = { SpvOpTypeVector, 3 };
    if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
      return error;
  
    return SPV_SUCCESS;
}

spv_result_t CheckGlobalLinearId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "GlobalLinearId";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // GlobalLinearId is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // GlobalLinearId is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // GlobalLinearId is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupSize";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupSize is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupSize is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupSize is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupMaxSize(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupMaxSize";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupMaxSize is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupMaxSize is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupMaxSize is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckNumSubgroups(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "NumSubgroups";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // NumSubgroups is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // NumSubgroups is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // NumSubgroups is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckNumEnqueuedSubgroups(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "NumEnqueuedSubgroups";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // NumEnqueuedSubgroups is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // NumEnqueuedSubgroups is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // NumEnqueuedSubgroups is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupId";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupId is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupId is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupId is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupLocalInvocationId(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupLocalInvocationId";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupLocalInvocationId is part of the OpenCL specification
  exec_vec = { SpvExecutionModelKernel };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupLocalInvocationId is an input argument to the kernels
  exec_vec = { SpvExecutionModelKernel };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupLocalInvocationId is a scalar 32-bit integer
  ElemType elem = { SpvOpTypeInt, 32, kUnsigned };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckVertexIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "VertexIndex";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The VertexIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // VertexIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // VertexIndex must be declared as
  // a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckInstanceIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "InstanceIndex";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The VertexIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // VertexIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // VertexIndex must be declared as
  // a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupEqMaskKHR(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupEqMaskKHR";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupEqMaskKHR is part of the SPV_KHR_shader_ballot extension
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupEqMaskKHR is an input argument
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupEqMaskKHR is a four-components vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupGeMaskKHR(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupGeMaskKHR";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupGeMaskKHR is part of the SPV_KHR_shader_ballot extension
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupGeMaskKHR is an input argument
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupGeMaskKHR is a four-components vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupGtMaskKHR(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupGtMaskKHR";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupGtMaskKHR is part of the SPV_KHR_shader_ballot extension
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupGtMaskKHR is an input argument
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupGtMaskKHR is a four-components vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupLeMaskKHR(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupLeMaskKHR";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupLeMaskKHR is part of the SPV_KHR_shader_ballot extension
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupLeMaskKHR is an input argument
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupLeMaskKHR is a four-components vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSubgroupLtMaskKHR(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SubgroupLtMaskKHR";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SubgroupLtMaskKHR is part of the SPV_KHR_shader_ballot extension
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;
    
  // SubgroupLtMaskKHR is an input argument
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelFragment,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SubgroupLtMaskKHR is a four-components vector of 32-bit integers
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaseVertex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaseVertex";
  std::vector<uint32_t> exec_vec, stor_vec;

  // The BaseVertex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaseVertex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaseVertex must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
     return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaseInstance(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaseInstance";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The BaseInstance decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaseInstance must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaseInstance must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
      return error;

  return SPV_SUCCESS;
}

spv_result_t CheckDrawIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "DrawIndex";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The DrawIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // DrawIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // DrawIndex must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
      return error;

  return SPV_SUCCESS;
}

spv_result_t CheckDeviceIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "DeviceIndex";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The VertexIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex,
                SpvExecutionModelFragment,
                SpvExecutionModelTessellationControl,
                SpvExecutionModelTessellationEvaluation,
                SpvExecutionModelGeometry,
                SpvExecutionModelGLCompute };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // VertexIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex,
                SpvExecutionModelFragment,
                SpvExecutionModelTessellationControl,
                SpvExecutionModelTessellationEvaluation,
                SpvExecutionModelGeometry,
                SpvExecutionModelGLCompute };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // VertexIndex must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
      return error;

  return SPV_SUCCESS;
}

spv_result_t CheckViewIndex(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "ViewIndex";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // The VertexIndex decoration must be used only within vertex shaders.
  exec_vec = { SpvExecutionModelVertex,
                SpvExecutionModelFragment,
                SpvExecutionModelTessellationControl,
                SpvExecutionModelTessellationEvaluation,
                SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // VertexIndex must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex,
                SpvExecutionModelFragment,
                SpvExecutionModelTessellationControl,
                SpvExecutionModelTessellationEvaluation,
                SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // VertexIndex must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
      return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordNoPerspAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordNoPerspAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordNoPerspCentroidAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordNoPerspCentroidAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordNoPerspSampleAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordNoPerspSampleAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordSmoothAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordSmoothAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordSmoothCentroidAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordSmoothCentroidAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordSmoothSampleAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordSmoothSampleAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a two-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 2 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckBaryCoordPullModelAMD(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "BaryCoordPullModelAMD";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // BaryCoordNoPerspAMD must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // BaryCoordNoPerspAMD must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassInput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a three-components vector of 32-bit floats
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 3 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckFragStencilRefEXT(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "FragStencilRefEXT";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // FragStencilRefEXT must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelFragment };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // FragStencilRefEXT must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelFragment };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // BaryCoordNoPerspAMD is a scalar integer of any-bits
  ElemType elem = { SpvOpTypeInt };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckViewportMaskNV(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "ViewportMaskNV";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // ViewportMaskNV must be used within Vertex, Tessellation or Geometry.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // ViewportMaskNV must be declared using the Output storage class.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // ViewportMaskNV must be declared as a scalar 32-bit integer.
  ElemType elem = { SpvOpTypeInt, 32 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem) )
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSecondaryPositionNV(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SecondaryPositionNV";
  std::vector<uint32_t> exec_vec, stor_vec;

  // SecondaryPositionNV must be used only within Vertex, Tessellation or Geometry.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // SecondaryPositionNV must be Output within Vertex.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SecondaryPositionNV must be Output within Tessellation and Geometry.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SecondaryPositionNV must be declared as a
  // four-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckSecondaryViewportMaskNV(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "SecondaryViewportMaskNV";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // SecondaryViewportMaskNV must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // SecondaryViewportMaskNV must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // SecondaryViewportMaskNV must be declared as an array of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeArray };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error; 

  return SPV_SUCCESS;
}

spv_result_t CheckPositionPerViewNV(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "PositionPerViewNV";
  std::vector<uint32_t> exec_vec, stor_vec;

  // PositionPerViewNV must be used only within Vertex, Tessellation or Geometry.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // PositionPerViewNV must be Output within Vertex.
  exec_vec = { SpvExecutionModelVertex };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // PositionPerViewNV must be Output within Tessellation and Geometry.
  exec_vec = { SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassInput,
               SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // PositionPerViewNV must be declared as a
  // four-component vector of 32-bit floating-point values.
  ElemType elem = { SpvOpTypeFloat, 32 };
  CompType comp = { SpvOpTypeVector, 4 };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error;

  return SPV_SUCCESS;
}

spv_result_t CheckViewportMaskPerViewNV(ValidationState_t& vstate, const Instruction* inst,
    uint32_t storage_class, uint32_t exec_model)
{
  const string name = "ViewportMaskPerViewNV";
  std::vector<uint32_t> exec_vec, stor_vec;
  
  // ViewportMaskPerViewNV must be used only within fragment shaders.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  if (auto error = CheckExecutionModel(vstate, name, exec_model, exec_vec))
    return error;

  // ViewportMaskPerViewNV must be declared using the Input storage class.
  exec_vec = { SpvExecutionModelVertex,
               SpvExecutionModelTessellationControl,
               SpvExecutionModelTessellationEvaluation,
               SpvExecutionModelGeometry };
  stor_vec = { SpvStorageClassOutput };
  if (auto error = CheckStorageClass(vstate, name, exec_model, storage_class, exec_vec, stor_vec))
    return error;

  // ViewportMaskPerViewNV must be declared as an array of 32-bit integers.
  ElemType elem = { SpvOpTypeInt, 32 };
  CompType comp = { SpvOpTypeArray };
  if (auto error = CheckBuiltInType(vstate, name, inst, elem, comp))
    return error; 

  return SPV_SUCCESS;
}

}  // anonymous namespace

namespace libspirv {

#define BUILTIN_CASE(bltin) \
  case SpvBuiltIn ## bltin: \
    return Check ## bltin (_,inst,storage_class,execution_model);

// Validates that decorations have been applied properly.
spv_result_t ValidateBuiltIns(ValidationState_t& _)
{
  // For every instruction decorated with a built-in...
  for (const auto& inst_ref : _.ordered_instructions()) {
    const Instruction *inst = &inst_ref;
    if (not IsBuiltIn(inst, _))
      continue;
    auto storage_class = GetStorageClass(inst);
    auto built_in = GetBuiltInEnum(inst, _);
    // For every 'execution model' affecting the built-in, check its validty
    for (auto execution_model : getExecutionModels(_, inst)) {
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
        BUILTIN_CASE( WorkDim                     )
        BUILTIN_CASE( GlobalSize                  )
        BUILTIN_CASE( EnqueuedWorkgroupSize       )
        BUILTIN_CASE( GlobalOffset                )
        BUILTIN_CASE( GlobalLinearId              )
        BUILTIN_CASE( SubgroupSize                )
        BUILTIN_CASE( SubgroupMaxSize             )
        BUILTIN_CASE( NumSubgroups                )
        BUILTIN_CASE( NumEnqueuedSubgroups        )
        BUILTIN_CASE( SubgroupId                  )
        BUILTIN_CASE( SubgroupLocalInvocationId   )
        BUILTIN_CASE( VertexIndex                 )
        BUILTIN_CASE( InstanceIndex               )
        BUILTIN_CASE( SubgroupEqMaskKHR           )
        BUILTIN_CASE( SubgroupGeMaskKHR           )
        BUILTIN_CASE( SubgroupGtMaskKHR           )
        BUILTIN_CASE( SubgroupLeMaskKHR           )
        BUILTIN_CASE( SubgroupLtMaskKHR           )
        BUILTIN_CASE( BaseVertex                  )
        BUILTIN_CASE( BaseInstance                )
        BUILTIN_CASE( DrawIndex                   )
        BUILTIN_CASE( DeviceIndex                 )
        BUILTIN_CASE( ViewIndex                   )
        BUILTIN_CASE( BaryCoordNoPerspAMD         )
        BUILTIN_CASE( BaryCoordNoPerspCentroidAMD )
        BUILTIN_CASE( BaryCoordNoPerspSampleAMD   )
        BUILTIN_CASE( BaryCoordSmoothAMD          )
        BUILTIN_CASE( BaryCoordSmoothCentroidAMD  )
        BUILTIN_CASE( BaryCoordSmoothSampleAMD    )
        BUILTIN_CASE( BaryCoordPullModelAMD       )
        BUILTIN_CASE( FragStencilRefEXT           )
        BUILTIN_CASE( ViewportMaskNV              )
        BUILTIN_CASE( SecondaryPositionNV         )
        BUILTIN_CASE( SecondaryViewportMaskNV     )
        BUILTIN_CASE( PositionPerViewNV           )
        BUILTIN_CASE( ViewportMaskPerViewNV       )
      }
    }
  }
  return SPV_SUCCESS;
}

#undef BUILTIN_CASE

}  // namespace libspirv

