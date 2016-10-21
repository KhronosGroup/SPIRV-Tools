
// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "validate.h"

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "opcode.h"
#include "val/Type.h"
#include "val/ValidationState.h"

using std::begin;
using std::end;
using std::find_if;
using std::get;
using std::string;
using std::tuple;
using std::vector;
using std::unordered_map;

using libspirv::ValidationState_t;

namespace {

// clang-format off
// Returns a string representation of the type category
string TypeFlagToString(spv_type_category_t flag) {
  string ret;
  switch (flag) {
    case SPV_TYPE_CATEGORY_TYPE_VOID:           ret = "TypeVoid"; break;
    case SPV_TYPE_CATEGORY_TYPE_BOOL:           ret = "TypeBoolean"; break;
    case SPV_TYPE_CATEGORY_TYPE_INT:            ret = "TypeInteger"; break;
    case SPV_TYPE_CATEGORY_TYPE_FLOAT:          ret = "TypeFloatingPoint"; break;
    case SPV_TYPE_CATEGORY_TYPE_IMAGE:          ret = "TypeImage"; break;
    case SPV_TYPE_CATEGORY_TYPE_SAMPLER:        ret = "TypeSampler"; break;
    case SPV_TYPE_CATEGORY_TYPE_VECTOR:         ret = "TypeVector"; break;
    case SPV_TYPE_CATEGORY_TYPE_MATRIX:         ret = "TypeMatrix"; break;
    case SPV_TYPE_CATEGORY_TYPE_ARRAY:          ret = "TypeArray"; break;
    case SPV_TYPE_CATEGORY_TYPE_RUNTIMEARRAY:   ret = "TypeRuntimeArray"; break;
    case SPV_TYPE_CATEGORY_TYPE_STRUCT:         ret = "TypeStructure"; break;
    case SPV_TYPE_CATEGORY_TYPE_SAMPLEDIMAGE:   ret = "TypeSampledImage"; break;
    case SPV_TYPE_CATEGORY_TYPE_FUNCTION:       ret = "TypeFunction"; break;
    case SPV_TYPE_CATEGORY_TYPE_OPAQUE:         ret = "TypeOpaque"; break;
    case SPV_TYPE_CATEGORY_TYPE_QUEUE:          ret = "TypeQueue"; break;
    case SPV_TYPE_CATEGORY_TYPE_POINTER:        ret = "TypePointer"; break;
    case SPV_TYPE_CATEGORY_TYPE_EVENT:          ret = "TypeEvent"; break;
    case SPV_TYPE_CATEGORY_TYPE_DEVICEEVENT:    ret = "TypeDeviceEvent"; break;
    case SPV_TYPE_CATEGORY_TYPE_RESERVEID:      ret = "TypeReserveId"; break;
    case SPV_TYPE_CATEGORY_TYPE_PIPE:           ret = "TypePipe"; break;
    case SPV_TYPE_CATEGORY_TYPE_FORWARDPOINTER: ret = "TypeForwardPointer"; break;
    case SPV_TYPE_CATEGORY_NUMERICAL:           ret = "TypeNumerical"; break;
    case SPV_TYPE_CATEGORY_SCALAR:              ret = "TypeScalar"; break;
    case SPV_TYPE_CATEGORY_AGGREGATE:           ret = "TypeAggregate"; break;
    case SPV_TYPE_CATEGORY_COMPOSITE:           ret = "TypeComposite"; break;
    case SPV_TYPE_CATEGORY_CONCRETE:            ret = "TypeConcrete"; break;
    case SPV_TYPE_CATEGORY_ABSTRACT:            ret = "TypeAbstract"; break;
    case SPV_TYPE_CATEGORY_OPAQUE:              ret = "TypeOpaque"; break;
    case SPV_TYPE_CATEGORY_ANY:                 ret = "TypeCategoryAny"; break;
    case SPV_TYPE_CATEGORY_NONE:                ret = "NONE"; break;
  }
  return ret;
}

// clang-format off
// Returns a string representation of a vector of type category flags
string TypeFlagsToString(const vector<spv_type_category_t>& flags) {
  string out = "{";
  for (auto flag : flags) {
    out += " " + TypeFlagToString(flag);
  }
  return out + " }";
}
// clang-format on

struct SpvOpHash {
  size_t operator()(SpvOp op) const {
    return std::hash<uint32_t>()(static_cast<uint32_t>(op));
  }
};

// TODO(umar): This representation for the type descriptors is not complete
// and I don't think it is robust enough to handle everything in the standard.
// TODO(umar): Support capability dependant checks
// TODO(umar): Support decoration dependant checks
// TODO(umar): Support type width and sign dependant checks
struct spv_type_desc_t {
  const SpvOp opcode;
  const vector<spv_type_category_t> result_type;
};

// NOTE: This will be generated at build time for a specific version
// of spirv. Ideally this should be generated by the spec and added
// to the build script as an external dependency.
spv_type_desc_t result_types[] = {
    {SpvOpUndef, {SPV_TYPE_CATEGORY_ANY}},
    {SpvOpConstantTrue, {SPV_TYPE_CATEGORY_TYPE_BOOL}},
    {SpvOpConstantFalse, {SPV_TYPE_CATEGORY_TYPE_BOOL}},
    // {SpvOpConstant,      TypeCategoryNumerical} // NOTE: checked by binay
    // parser
    {SpvOpConstantComposite, {SPV_TYPE_CATEGORY_COMPOSITE}},
    {SpvOpConstantSampler, {SPV_TYPE_CATEGORY_TYPE_SAMPLER}},
    {SpvOpConstantNull,
     {SPV_TYPE_CATEGORY_SCALAR, SPV_TYPE_CATEGORY_TYPE_VECTOR,
      SPV_TYPE_CATEGORY_TYPE_POINTER, SPV_TYPE_CATEGORY_TYPE_EVENT,
      SPV_TYPE_CATEGORY_TYPE_DEVICEEVENT, SPV_TYPE_CATEGORY_TYPE_RESERVEID,
      SPV_TYPE_CATEGORY_TYPE_QUEUE, SPV_TYPE_CATEGORY_COMPOSITE}}};

// Checks the result type of the instruction.
spv_result_t CheckResultType(ValidationState_t& _, SpvOp opcode,
                             uint32_t operand) {
  auto found = find_if(
      begin(result_types), end(result_types),
      [opcode](const spv_type_desc_t& type) { return type.opcode == opcode; });

  if (found != end(result_types)) {
    auto* type = _.types(operand);
    if (!type->IsTypeAny(found->result_type)) {
      return _.diag(SPV_ERROR_INVALID_TYPE)
             << "Opcode " << spvOpcodeString(opcode)
             << " requires the result type to be " +
                    TypeFlagsToString(found->result_type) + " found " +
                    TypeFlagToString(type->category());
    }
  } else {
    // TODO(umar): Add assert once the result_types object is complete
  }
  return SPV_SUCCESS;
}
}  // namespace

namespace libspirv {

// clang-format off
// Returns a type category of a type generating opcode
spv_type_category_t OpcodeToTypeFlag(SpvOp opcode) {
  spv_type_category_t ret = SPV_TYPE_CATEGORY_NONE;
  switch (opcode) {
    case SpvOpTypeVoid:           ret = SPV_TYPE_CATEGORY_TYPE_VOID; break;
    case SpvOpTypeBool:           ret = SPV_TYPE_CATEGORY_TYPE_BOOL; break;
    case SpvOpTypeInt:            ret = SPV_TYPE_CATEGORY_TYPE_INT; break;
    case SpvOpTypeFloat:          ret = SPV_TYPE_CATEGORY_TYPE_FLOAT; break;
    case SpvOpTypeImage:          ret = SPV_TYPE_CATEGORY_TYPE_IMAGE; break;
    case SpvOpTypeSampler:        ret = SPV_TYPE_CATEGORY_TYPE_SAMPLER; break;
    case SpvOpTypeVector:         ret = SPV_TYPE_CATEGORY_TYPE_VECTOR; break;
    case SpvOpTypeMatrix:         ret = SPV_TYPE_CATEGORY_TYPE_MATRIX; break;
    case SpvOpTypeArray:          ret = SPV_TYPE_CATEGORY_TYPE_ARRAY; break;
    case SpvOpTypeRuntimeArray:   ret = SPV_TYPE_CATEGORY_TYPE_RUNTIMEARRAY; break;
    case SpvOpTypeStruct:         ret = SPV_TYPE_CATEGORY_TYPE_STRUCT; break;
    case SpvOpTypeSampledImage:   ret = SPV_TYPE_CATEGORY_TYPE_SAMPLEDIMAGE; break;
    case SpvOpTypePointer:        ret = SPV_TYPE_CATEGORY_TYPE_POINTER; break;
    case SpvOpTypeFunction:       ret = SPV_TYPE_CATEGORY_TYPE_FUNCTION; break;
    case SpvOpTypeOpaque:         ret = SPV_TYPE_CATEGORY_TYPE_OPAQUE; break;
    case SpvOpTypeEvent:          ret = SPV_TYPE_CATEGORY_TYPE_EVENT; break;
    case SpvOpTypeDeviceEvent:    ret = SPV_TYPE_CATEGORY_TYPE_DEVICEEVENT; break;
    case SpvOpTypeReserveId:      ret = SPV_TYPE_CATEGORY_TYPE_RESERVEID; break;
    case SpvOpTypeQueue:          ret = SPV_TYPE_CATEGORY_TYPE_QUEUE; break;
    case SpvOpTypePipe:           ret = SPV_TYPE_CATEGORY_TYPE_PIPE; break;
    case SpvOpTypeForwardPointer: ret = SPV_TYPE_CATEGORY_TYPE_FORWARDPOINTER; break;
    default:
      assert(1 == 0 && "Not Defined");
  }
  return ret;
}
// clang-format on

spv_result_t TypePass(ValidationState_t& _,
                      const spv_parsed_instruction_t* inst) {
  const SpvOp opcode = static_cast<SpvOp>(inst->opcode);
  if (spvOpcodeGeneratesType(opcode)) {
    // Aggregate types can contain aliases
    if (opcode != SpvOpTypeArray && opcode != SpvOpTypeStruct &&
        opcode != SpvOpTypeRuntimeArray) {
      if (const auto* alias = _.GetTypeAlias(*inst)) {
        return _.diag(SPV_ERROR_INVALID_TYPE)
               << "Type " << _.getIdName(inst->result_id)
               << " is an alias to type " << _.getIdName(alias->id());
      }
    }
    _.AddType(*inst);
  }

  if (auto error = CheckResultType(_, opcode, inst->type_id)) return error;
  // TODO(umar): Add checks for operands
  return SPV_SUCCESS;
}

}  // namespace libspirv
