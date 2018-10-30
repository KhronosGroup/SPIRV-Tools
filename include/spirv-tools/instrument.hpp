// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#ifndef INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_
#define INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_

namespace spvtools {

// Error Codes
// These are utilized in debug output record streams and identify which
// validation error has occurred and written a specific record.
static const uint32_t kInstErrorBindlessBounds = 0;
static const uint32_t kInstErrorBindlessUninitialized = 1;

// Debug Buffer Bindings
static const uint32_t kDebugOutputBindingStream = 0;

// Debug Buffer Offsets
static const int kDebugOutputSizeOffset = 0;
static const int kDebugOutputDataOffset = 1;

// Common Output Record Offsets
static const int kInstCommonOutSize = 0;
static const int kInstCommonOutShaderId = 1;
static const int kInstCommonOutInstructionIdx = 2;
static const int kInstCommonOutStageIdx = 3;
static const int kInstCommonOutCnt = 4;

// Vertex Shader Output Record Offsets
static const int kInstVertOutVertexId = kInstCommonOutCnt;
static const int kInstVertOutInstanceId = kInstCommonOutCnt + 1;

// Frag Shader Output Record Offsets
static const int kInstFragOutFragCoordX = kInstCommonOutCnt;
static const int kInstFragOutFragCoordY = kInstCommonOutCnt + 1;

// Compute Shader Output Record Offsets
static const int kInstCompOutGlobalInvocationId = kInstCommonOutCnt;
static const int kInstCompOutUnused = kInstCommonOutCnt + 1;

// Tessellation Shader Output Record Offsets
static const int kInstTessOutInvocationId = kInstCommonOutCnt;
static const int kInstTessOutUnused = kInstCommonOutCnt + 1;

// Geometry Shader Output Record Offsets
static const int kInstGeomOutPrimitiveId = kInstCommonOutCnt;
static const int kInstGeomOutInvocationId = kInstCommonOutCnt + 1;

// Size of Common and Stage-specific Members
static const int kInstStageOutCnt = kInstCommonOutCnt + 2;

// Bindless-specific Output Record Offsets
static const int kInstBindlessOutError = kInstStageOutCnt;
static const int kInstBindlessOutDescIndex = kInstStageOutCnt + 1;
static const int kInstBindlessOutDescBound = kInstStageOutCnt + 2;
static const int kInstBindlessOutRecordSize = kInstStageOutCnt + 3;

// Maximum Output Record Member Count
static const int kInstMaxOutCnt = kInstStageOutCnt + 3;

}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_INSTRUMENT_HPP_
