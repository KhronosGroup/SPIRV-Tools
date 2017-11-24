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

#ifndef LIBSPIRV_ENUM_STRING_MAPPING_H_
#define LIBSPIRV_ENUM_STRING_MAPPING_H_

#include <string>

// clang-format off
#include "latest_spirv.h"
// clang-format on

#include "extensions.h"

namespace libspirv {

// Finds Extension enum corresponding to |str|. Returns false if not found.
bool GetExtensionFromString(const std::string& str, Extension* extension);

// Returns text string corresponding to |extension|.
std::string ExtensionToString(Extension extension);

// Returns text string corresponding to the given enumerant.
std::string SourceLanguageToString(SpvSourceLanguage);
std::string ExecutionModelToString(SpvExecutionModel);
std::string AddressingModelToString(SpvAddressingModel);
std::string MemoryModelToString(SpvMemoryModel);
std::string ExecutionModeToString(SpvExecutionMode);
std::string StorageClassToString(SpvStorageClass);
std::string DimToString(SpvDim);
std::string SamplerAddressingModeToString(SpvSamplerAddressingMode);
std::string SamplerFilterModeToString(SpvSamplerFilterMode);
std::string ImageFormatToString(SpvImageFormat);
std::string ImageChannelOrderToString(SpvImageChannelOrder);
std::string ImageChannelDataTypeToString(SpvImageChannelDataType);
std::string FPRoundingModeToString(SpvFPRoundingMode);
std::string LinkageTypeToString(SpvLinkageType);
std::string AccessQualifierToString(SpvAccessQualifier);
std::string FunctionParameterAttributeToString(SpvFunctionParameterAttribute);
std::string DecorationToString(SpvDecoration);
std::string BuiltInToString(SpvBuiltIn);
std::string ScopeToString(SpvScope);
std::string GroupOperationToString(SpvGroupOperation);
std::string KernelEnqueueFlagsToString(SpvKernelEnqueueFlags);
std::string CapabilityToString(SpvCapability);

}  // namespace libspirv

#endif  // LIBSPIRV_ENUM_STRING_MAPPING_H_
