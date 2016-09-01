// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef LIBSPIRV_BINARY_H_
#define LIBSPIRV_BINARY_H_

#include "spirv-tools/libspirv.h"
#include "spirv/1.1/spirv.h"
#include "spirv_definition.h"

// Functions

// Grabs the header from the SPIR-V module given in the binary parameter. The
// endian parameter specifies the endianness of the binary module. On success,
// returns SPV_SUCCESS and writes the parsed header into *header.
spv_result_t spvBinaryHeaderGet(const spv_const_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t* header);

#endif  // LIBSPIRV_BINARY_H_
