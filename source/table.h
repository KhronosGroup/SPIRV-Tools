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

#ifndef SOURCE_TABLE_H_
#define SOURCE_TABLE_H_

#include "source/extensions.h"
#include "source/latest_version_spirv_header.h"
#include "source/util/index_range.h"
#include "spirv-tools/libspirv.hpp"

// NOTE: Instruction and operand tables have moved to table2.{h|cpp},
// where they are represented more compactly.
// TODO(dneto): Move the remaining tables to the new scheme.

// Define the static tables that describe the grammatical structure
// of SPIR-V instructions and their operands. These tables are populated
// by reading the grammar files from SPIRV-Headers.
//
// Most clients access these tables indirectly via an spv_context_t object.
//
// The overall structure among containers (i.e. skipping scalar data members)
// is as follows:
//
//    An spv_context_t:
//      - points to spv_ext_inst_table_t  = array of spv_ext_inst_group_t
//
//    An spv_ext_inst_group_t has:
//      - array of spv_ext_inst_desc_t
//
//    An spv_ext_inst_desc_t has:
//      - a name string
//      - array of spv::Capability
//      - array of spv_operand_type_t

typedef struct spv_ext_inst_desc_t {
  const char* name;
  const uint32_t ext_inst;
  const uint32_t numCapabilities;
  const spv::Capability* capabilities;
  const spv_operand_type_t operandTypes[40];  // vksp needs at least 40
} spv_ext_inst_desc_t;

typedef struct spv_ext_inst_group_t {
  const spv_ext_inst_type_t type;
  const uint32_t count;
  const spv_ext_inst_desc_t* entries;
} spv_ext_inst_group_t;

typedef struct spv_ext_inst_table_t {
  const uint32_t count;
  const spv_ext_inst_group_t* groups;
} spv_ext_inst_table_t;

typedef const spv_ext_inst_desc_t* spv_ext_inst_desc;

typedef const spv_ext_inst_table_t* spv_ext_inst_table;

struct spv_context_t {
  const spv_target_env target_env;
  const spv_ext_inst_table ext_inst_table;
  spvtools::MessageConsumer consumer;
};

namespace spvtools {

// Sets the message consumer to |consumer| in the given |context|. The original
// message consumer will be overwritten.
void SetContextMessageConsumer(spv_context context, MessageConsumer consumer);
}  // namespace spvtools

// Populates *table with entries for env.
spv_result_t spvExtInstTableGet(spv_ext_inst_table* table, spv_target_env env);

#endif  // SOURCE_TABLE_H_
