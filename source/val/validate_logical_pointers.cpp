// Copyright (c) 2025 Google LLC.
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

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "source/opcode.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if inst is a logical pointer.
bool IsLogicalPointer(const ValidationState_t& _, const Instruction* inst) {
  if (!_.IsPointerType(inst->type_id())) {
    return false;
  }

  // Physical storage buffer pointers are not logical pointers.
  auto type_inst = _.FindDef(inst->type_id());
  auto sc = type_inst->GetOperandAs<spv::StorageClass>(1);
  if (sc == spv::StorageClass::PhysicalStorageBuffer) {
    return false;
  }

  return true;
}

bool IsVariablePointer(const ValidationState_t& _,
                       std::unordered_map<uint32_t, bool>& variable_pointers,
                       const Instruction* inst) {
  const auto iter = variable_pointers.find(inst->id());
  if (iter != variable_pointers.end()) {
    return iter->second;
  }

  bool is_var_ptr = false;
  switch (inst->opcode()) {
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpUntypedPtrAccessChainKHR:
    case spv::Op::OpUntypedInBoundsPtrAccessChainKHR:
    case spv::Op::OpLoad:
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpConstantNull:
      is_var_ptr = true;
      break;
    case spv::Op::OpFunctionParameter:
      // Special case: skip to function calls.
      if (IsLogicalPointer(_, inst)) {
        auto func = inst->function();
        auto func_inst = _.FindDef(func->id());

        const auto param_inst_num = inst - &_.ordered_instructions()[0];
        uint32_t param_index = 0;
        uint32_t inst_index = 1;
        while (_.ordered_instructions()[param_inst_num - inst_index].opcode() !=
               spv::Op::OpFunction) {
          if (_.ordered_instructions()[param_inst_num - inst_index].opcode() ==
              spv::Op::OpFunctionParameter) {
            param_index++;
          }
          ++inst_index;
        }

        for (const auto& use_pair : func_inst->uses()) {
          const auto use_inst = use_pair.first;
          if (use_inst->opcode() == spv::Op::OpFunctionCall) {
            const auto arg_id =
                use_inst->GetOperandAs<uint32_t>(3 + param_index);
            const auto arg_inst = _.FindDef(arg_id);
            is_var_ptr |= IsVariablePointer(_, variable_pointers, arg_inst);
          }
        }
      }
      break;
    default: {
      for (uint32_t i = 0; i < inst->operands().size(); ++i) {
        if (inst->operands()[i].type != SPV_OPERAND_TYPE_ID) {
          continue;
        }

        auto op_inst = _.FindDef(inst->GetOperandAs<uint32_t>(i));
        if (IsLogicalPointer(_, op_inst)) {
          is_var_ptr |= IsVariablePointer(_, variable_pointers, op_inst);
        }
      }
      break;
    }
  }
  variable_pointers[inst->id()] = is_var_ptr;
  return is_var_ptr;
}

spv_result_t ValidateLogicalPointerOperands(ValidationState_t& _,
                                            const Instruction* inst) {
  if (inst->type_id() == 0) {
    return SPV_SUCCESS;
  }

  bool has_pointer_operand = false;
  spv::StorageClass sc = spv::StorageClass::Function;
  for (uint32_t i = 0; i < inst->operands().size(); ++i) {
    if (inst->operands()[i].type != SPV_OPERAND_TYPE_ID) {
      continue;
    }

    auto op_inst = _.FindDef(inst->GetOperandAs<uint32_t>(i));
    if (IsLogicalPointer(_, op_inst)) {
      has_pointer_operand = true;

      // Assume that there are not mixed storage classes in the instruction.
      // This is not true for OpCopyMemory and OpCopyMemorySized, but they allow
      // all storage classes.
      auto type_inst = _.FindDef(op_inst->type_id());
      sc = type_inst->GetOperandAs<spv::StorageClass>(1);
      break;
    }
  }

  if (!has_pointer_operand) {
    return SPV_SUCCESS;
  }

  switch (inst->opcode()) {
    // The following instructions allow logical pointer operands in all cases
    // without capabilities.
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyObject:
    case spv::Op::OpArrayLength:
    case spv::Op::OpExtInst:
    // Core spec bugs
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpEntryPoint:
    // SPV_KHR_untyped_pointers
    case spv::Op::OpUntypedArrayLengthKHR:
    case spv::Op::OpUntypedAccessChainKHR:
    case spv::Op::OpUntypedInBoundsAccessChainKHR:
    case spv::Op::OpCopyMemorySized:
    // Cooperative matrix KHR/NV
    case spv::Op::OpCooperativeMatrixLoadKHR:
    case spv::Op::OpCooperativeMatrixLoadNV:
    case spv::Op::OpCooperativeMatrixStoreKHR:
    case spv::Op::OpCooperativeMatrixStoreNV:
    // SPV_KHR_ray_query (spec bugs)
    case spv::Op::OpRayQueryInitializeKHR:
    case spv::Op::OpRayQueryTerminateKHR:
    case spv::Op::OpRayQueryGenerateIntersectionKHR:
    case spv::Op::OpRayQueryProceedKHR:
    case spv::Op::OpRayQueryGetIntersectionTypeKHR:
    case spv::Op::OpRayQueryGetRayTMinKHR:
    case spv::Op::OpRayQueryGetRayFlagsKHR:
    case spv::Op::OpRayQueryGetIntersectionTKHR:
    case spv::Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR:
    case spv::Op::OpRayQueryGetIntersectionInstanceIdKHR:
    case spv::Op::
        OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:
    case spv::Op::OpRayQueryGetIntersectionGeometryIndexKHR:
    case spv::Op::OpRayQueryGetIntersectionPrimitiveIndexKHR:
    case spv::Op::OpRayQueryGetIntersectionBarycentricsKHR:
    case spv::Op::OpRayQueryGetIntersectionFrontFaceKHR:
    case spv::Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR:
    case spv::Op::OpRayQueryGetIntersectionObjectRayDirectionKHR:
    case spv::Op::OpRayQueryGetIntersectionObjectRayOriginKHR:
    case spv::Op::OpRayQueryGetWorldRayDirectionKHR:
    case spv::Op::OpRayQueryGetWorldRayOriginKHR:
    case spv::Op::OpRayQueryGetIntersectionObjectToWorldKHR:
    case spv::Op::OpRayQueryGetIntersectionWorldToObjectKHR:
    // SPV_KHR_ray_tracing_position_fetch (spec bugs)
    case spv::Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR:
    // SPV_NV_cluster_acceleration_structure (spec bugs)
    case spv::Op::OpRayQueryGetClusterIdNV:
    case spv::Op::OpHitObjectGetClusterIdNV:
    // SPV_NV_linear_swept_spheres (spec bugs)
    case spv::Op::OpRayQueryGetIntersectionSpherePositionNV:
    case spv::Op::OpRayQueryGetIntersectionSphereRadiusNV:
    case spv::Op::OpRayQueryGetIntersectionLSSPositionsNV:
    case spv::Op::OpRayQueryGetIntersectionLSSRadiiNV:
    case spv::Op::OpRayQueryGetIntersectionLSSHitValueNV:
    case spv::Op::OpRayQueryIsSphereHitNV:
    case spv::Op::OpRayQueryIsLSSHitNV:
    case spv::Op::OpHitObjectGetSpherePositionNV:
    case spv::Op::OpHitObjectGetSphereRadiusNV:
    case spv::Op::OpHitObjectGetLSSPositionsNV:
    case spv::Op::OpHitObjectGetLSSRadiiNV:
    case spv::Op::OpHitObjectIsSphereHitNV:
    case spv::Op::OpHitObjectIsLSSHitNV:
    // SPV_NV_shader_invocation_reorder (spec bugs)
    case spv::Op::OpReorderThreadWithHitObjectNV:
    case spv::Op::OpHitObjectTraceRayNV:
    case spv::Op::OpHitObjectTraceRayMotionNV:
    case spv::Op::OpHitObjectRecordHitNV:
    case spv::Op::OpHitObjectRecordHitMotionNV:
    case spv::Op::OpHitObjectRecordHitWithIndexNV:
    case spv::Op::OpHitObjectRecordHitWithIndexMotionNV:
    case spv::Op::OpHitObjectRecordMissNV:
    case spv::Op::OpHitObjectRecordMissMotionNV:
    case spv::Op::OpHitObjectRecordEmptyNV:
    case spv::Op::OpHitObjectExecuteShaderNV:
    case spv::Op::OpHitObjectGetCurrentTimeNV:
    case spv::Op::OpHitObjectGetAttributesNV:
    case spv::Op::OpHitObjectGetHitKindNV:
    case spv::Op::OpHitObjectGetPrimitiveIndexNV:
    case spv::Op::OpHitObjectGetGeometryIndexNV:
    case spv::Op::OpHitObjectGetInstanceIdNV:
    case spv::Op::OpHitObjectGetInstanceCustomIndexNV:
    case spv::Op::OpHitObjectGetObjectRayOriginNV:
    case spv::Op::OpHitObjectGetObjectRayDirectionNV:
    case spv::Op::OpHitObjectGetWorldRayDirectionNV:
    case spv::Op::OpHitObjectGetWorldRayOriginNV:
    case spv::Op::OpHitObjectGetObjectToWorldNV:
    case spv::Op::OpHitObjectGetWorldToObjectNV:
    case spv::Op::OpHitObjectGetRayTMaxNV:
    case spv::Op::OpHitObjectGetRayTMinNV:
    case spv::Op::OpHitObjectGetShaderBindingTableRecordIndexNV:
    case spv::Op::OpHitObjectGetShaderRecordBufferHandleNV:
    case spv::Op::OpHitObjectIsEmptyNV:
    case spv::Op::OpHitObjectIsHitNV:
    case spv::Op::OpHitObjectIsMissNV:
    // SPV_NV_raw_access_chains
    case spv::Op::OpRawAccessChainNV:
    // SPV_NV_cooperative_matrix2
    case spv::Op::OpCooperativeMatrixLoadTensorNV:
    case spv::Op::OpCooperativeMatrixStoreTensorNV:
    // SPV_NV_cooperative_vector
    case spv::Op::OpCooperativeVectorLoadNV:
    case spv::Op::OpCooperativeVectorStoreNV:
    case spv::Op::OpCooperativeVectorMatrixMulNV:
    case spv::Op::OpCooperativeVectorMatrixMulAddNV:
    case spv::Op::OpCooperativeVectorOuterProductAccumulateNV:
    case spv::Op::OpCooperativeVectorReduceSumAccumulateNV:
      return SPV_SUCCESS;
    // The following cases require a variable pointer capability. Since all
    // instructions are for variable pointers, the storage class and capability
    // are also checked.
    case spv::Op::OpReturnValue:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
    case spv::Op::OpPtrDiff:
    // Core spec bugs
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpVariable:
    // SPV_KHR_untyped_pointers
    case spv::Op::OpUntypedPtrAccessChainKHR:
      if ((_.HasCapability(spv::Capability::VariablePointersStorageBuffer) &&
           sc == spv::StorageClass ::StorageBuffer) ||
          (_.HasCapability(spv::Capability::VariablePointers) &&
           sc == spv::StorageClass::Workgroup)) {
        return SPV_SUCCESS;
      }
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Instruction may only have a logical pointer operand in the "
                "StorageBuffer or Workgroup storage classes with appropriate "
                "variable pointers capability";
    default:
      if (spvOpcodeIsAtomicOp(inst->opcode())) {
        return SPV_SUCCESS;
      }
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Instruction may not have a logical pointer operand";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateLogicalPointerReturns(ValidationState_t& _,
                                           const Instruction* inst)
{
  if (!IsLogicalPointer(_, inst)) {
    return SPV_SUCCESS;
  }

  const auto type_inst = _.FindDef(inst->type_id());
  const auto sc = type_inst->GetOperandAs<spv::StorageClass>(1u);

  switch (inst->opcode()) {
    // Core spec without an variable pointer capability.
    case spv::Op::OpVariable:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpImageTexelPointer:
    case spv::Op::OpCopyObject:
    // Core spec bugs
    case spv::Op::OpUndef:
    // SPV_KHR_untyped_pointers
    case spv::Op::OpUntypedAccessChainKHR:
    case spv::Op::OpUntypedInBoundsAccessChainKHR:
    case spv::Op::OpUntypedVariableKHR:
    // SPV_NV_raw_access_chains
    case spv::Op::OpRawAccessChainNV:
      return SPV_SUCCESS;
    // Core spec with variable pointer capability. Check storage classes since
    // variable pointers can only be in certain storage classes.
    case spv::Op::OpSelect:
    case spv::Op::OpPhi:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpLoad:
    case spv::Op::OpConstantNull:
    case spv::Op::OpFunction:
    // SPV_KHR_untyped_pointers
    case spv::Op::OpUntypedPtrAccessChainKHR:
      if ((_.HasCapability(spv::Capability::VariablePointersStorageBuffer) &&
           sc == spv::StorageClass ::StorageBuffer) ||
          (_.HasCapability(spv::Capability::VariablePointers) &&
           sc == spv::StorageClass::Workgroup)) {
        return SPV_SUCCESS;
      }
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Instruction may only return a logical pointer in the "
                "StorageBuffer or Workgroup storage classes with appropriate "
                "variable pointers capability";
    default:
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Instruction may not return a logical pointer";
  }

  return SPV_SUCCESS;
}

spv_result_t CheckMatrixElementTyped(ValidationState_t& _,
                                     const Instruction* inst)
{
  std::vector<const Instruction*> stack;
  std::unordered_set<const Instruction*> seen;
  stack.push_back(inst);
  while (!stack.empty()) {
    const Instruction* trace_inst = stack.back();
    stack.pop_back();

    if (!seen.insert(trace_inst).second) {
      continue;
    }

    switch (trace_inst->opcode()) {
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpPtrAccessChain: {
        // Get the type of the base operand.
        uint32_t start_index =
            trace_inst->opcode() == spv::Op::OpPtrAccessChain ? 4 : 3;
        const auto access_type_id = _.GetOperandTypeId(trace_inst, 2);
        auto access_type = _.FindDef(access_type_id);
        access_type = _.FindDef(access_type->GetOperandAs<uint32_t>(2));
        // If the base operand is a matrix, then it was definitely pointing to a
        // sub-component.
        if (access_type->opcode() == spv::Op::OpTypeMatrix) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Variable pointer must not point to a column or a "
                    "component of a column of a matrix. Occurs due to:\n"
                 << _.Disassemble(*trace_inst);
        }

        // Otherwise, step through the indices to see if we pass a matrix.
        for (uint32_t i = start_index; i < trace_inst->operands().size(); ++i) {
          const auto index = trace_inst->GetOperandAs<uint32_t>(i);
          if (access_type->opcode() == spv::Op::OpTypeStruct) {
            uint64_t val = 0;
            _.EvalConstantValUint64(index, &val);
            access_type =
                _.FindDef(access_type->GetOperandAs<uint32_t>(1 + val));
          } else {
            access_type = _.FindDef(_.GetComponentType(access_type->id()));
          }

          if (access_type->opcode() == spv::Op::OpTypeMatrix) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << "Variable pointer must not point to a column or a "
                      "component of a column of a matrix. Occurs due to:\n"
                   << _.Disassemble(*trace_inst);
          }
          stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(2)));
        }
        break;
      }
      case spv::Op::OpPhi:
        for (uint32_t i = 2; i < trace_inst->operands().size(); i += 2) {
          stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(i)));
        }
        break;
      case spv::Op::OpSelect:
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(3)));
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(4)));
        break;
      case spv::Op::OpFunctionParameter: {
        // Jump to function calls
        auto func = trace_inst->function();
        auto func_inst = _.FindDef(func->id());

        const auto param_inst_num = trace_inst - &_.ordered_instructions()[0];
        uint32_t param_index = 0;
        uint32_t inst_index = 1;
        while (_.ordered_instructions()[param_inst_num - inst_index].opcode() !=
               spv::Op::OpFunction) {
          if (_.ordered_instructions()[param_inst_num - inst_index].opcode() ==
              spv::Op::OpFunctionParameter) {
            param_index++;
          }
          ++inst_index;
        }

        for (const auto& use_pair : func_inst->uses()) {
          const auto use_inst = use_pair.first;
          if (use_inst->opcode() == spv::Op::OpFunctionCall) {
            const auto arg_id =
                use_inst->GetOperandAs<uint32_t>(3 + param_index);
            const auto arg_inst = _.FindDef(arg_id);
            stack.push_back(arg_inst);
          }
        }
        break;
      }
      case spv::Op::OpFunctionCall: {
        // Jump to return values.
        const auto* func = _.function(trace_inst->GetOperandAs<uint32_t>(2));
        for (auto* bb : func->ordered_blocks()) {
          const auto* terminator = bb->terminator();
          if (terminator->opcode() == spv::Op::OpReturnValue) {
            stack.push_back(terminator);
          }
        }
        break;
      }
      case spv::Op::OpReturnValue:
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(0)));
        break;
      case spv::Op::OpCopyObject:
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(2)));
        break;
      case spv::Op::OpLoad:
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(2)));
        break;
      case spv::Op::OpStore:
        stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(0)));
        break;
      case spv::Op::OpVariable: {
        const auto sc = trace_inst->GetOperandAs<spv::StorageClass>(2);
        if (sc == spv::StorageClass::Function ||
            sc == spv::StorageClass::Private) {
          // Add the initializer
          if (trace_inst->operands().size() > 3) {
            stack.push_back(_.FindDef(trace_inst->GetOperandAs<uint32_t>(3)));
          }
          // Jump to stores
          std::vector<std::pair<const Instruction*, uint32_t>> store_stack(
              trace_inst->uses());
          std::unordered_set<const Instruction*> store_seen;
          while (!store_stack.empty()) {
            const auto& use = store_stack.back();
            store_stack.pop_back();

            if (!store_seen.insert(use.first).second) {
              continue;
            }

            // If the use is a store pointer, trace the store object.
            // Note: use.second is a word index.
            if (use.first->opcode() == spv::Op::OpStore && use.second == 1) {
              stack.push_back(_.FindDef(use.first->GetOperandAs<uint32_t>(1)));
            } else {
              // Most likely a gep so keep tracing.
              for (auto& next_use : use.first->uses()) {
                store_stack.push_back(next_use);
              }
            }
          }
        }
        break;
      }
      default:
        break;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateVariablePointers(ValidationState_t& _,
                                      std::unordered_map<uint32_t, bool>& variable_pointers,
                                      const Instruction* inst) {
  // Variable pointers cannot be operands to array length.
  if (inst->opcode() == spv::Op::OpArrayLength ||
      inst->opcode() == spv::Op::OpUntypedArrayLengthKHR) {
    const auto ptr_index = inst->opcode() == spv::Op::OpArrayLength ? 2 : 3;
    const auto ptr_id = inst->GetOperandAs<uint32_t>(ptr_index);
    const auto ptr_inst = _.FindDef(ptr_id);
    if (IsVariablePointer(_, variable_pointers, ptr_inst)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Pointer operand must not be a variable pointer";
    }
    return SPV_SUCCESS;
  }

  if (!IsLogicalPointer(_, inst)) {
    return SPV_SUCCESS;
  }

  if (!IsVariablePointer(_, variable_pointers, inst)) {
    return SPV_SUCCESS;
  }

  // Variable pointers must not:
  // * point to array of Block- or BufferBlock-decorated structs
  // * point to an object that is or contains a matrix
  // * point to a column, or component in a column, of a matrix
  auto type_inst = _.FindDef(inst->type_id());
  if (type_inst->opcode() == spv::Op::OpTypePointer) {
    const auto pointee_type = _.FindDef(type_inst->GetOperandAs<uint32_t>(2));
    if (pointee_type->opcode() == spv::Op::OpTypeArray ||
        pointee_type->opcode() == spv::Op::OpTypeRuntimeArray) {
      const auto element_type =
          _.FindDef(pointee_type->GetOperandAs<uint32_t>(1));
      if (element_type->opcode() == spv::Op::OpTypeStruct &&
          (_.HasDecoration(element_type->id(), spv::Decoration::Block) ||
           _.HasDecoration(element_type->id(), spv::Decoration::BufferBlock))) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Variable pointer must not point to an array of Block- or "
                  "BufferBlock-decorated structs";
      }
    } else if (_.ContainsType(
                   pointee_type->id(),
                   [](const Instruction* inst) {
                     return inst->opcode() == spv::Op::OpTypeMatrix;
                   },
                   /* traverse_all_types = */ false)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Variable pointer must not point to an object that is or "
                "contains a matrix";
    } else if (_.IsFloatScalarOrVectorType(pointee_type->id())) {
      // Pointing to a column or component in a column is trickier to detect.
      // Trace backwards and check encountered access chains to determine if
      // this pointer is pointing into a matrix.
      if (auto error = CheckMatrixElementTyped(_, inst)) {
        return error;
      }
    }
  } else {
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ValidateLogicalPointers(ValidationState_t& _) {
  // Only the following addressing models have logical pointers.
  if (_.addressing_model() != spv::AddressingModel::Logical &&
      _.addressing_model() != spv::AddressingModel::PhysicalStorageBuffer64) {
    return SPV_SUCCESS;
  }

  if (_.options()->relax_logical_pointer) {
    return SPV_SUCCESS;
  }

  // Cache all variable pointers
  std::unordered_map<uint32_t, bool> variable_pointers;
  for (auto& inst : _.ordered_instructions()) {
    if (!IsLogicalPointer(_, &inst)) {
      continue;
    }

    IsVariablePointer(_, variable_pointers, &inst);
  }

  for (auto& inst : _.ordered_instructions()) {
    // if (IsVariablePointer(_, variable_pointers, &inst)) {
    //   std::cerr << "Variable pointer: " << _.Disassemble(inst) << "\n";
    // }

    if (auto error = ValidateLogicalPointerOperands(_, &inst)) {
      return error;
    }
    if (auto error = ValidateLogicalPointerReturns(_, &inst)) {
      return error;
    }
    if (auto error = ValidateVariablePointers(_, variable_pointers, &inst)) {
      return error;
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
