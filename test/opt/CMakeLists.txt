# Copyright (c) 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_spvtools_unittest(TARGET instruction
  SRCS test_instruction.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET ir_loader
  SRCS test_ir_loader.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_manager
  SRCS module_utils.h
       test_pass_manager.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_strip_debug_info
  SRCS test_strip_debug_info.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_freeze_spec_const
  SRCS test_freeze_spec_const.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_eliminate_dead_const
  SRCS test_eliminate_dead_const.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_utils
  SRCS test_utils.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET def_use
  SRCS test_def_use.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET assembly_builder
  SRCS test_assembly_builder.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET types
  SRCS test_types.cpp
  LIBS SPIRV-Tools-opt
)

add_spvtools_unittest(TARGET type_manager
  SRCS test_type_manager.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET iterator
  SRCS test_iterator.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET module
  SRCS module_utils.h
       test_module.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_fold_spec_const_op_composite
  SRCS test_fold_spec_const_op_composite.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_unify_const
  SRCS test_unify_const.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET pass_set_spec_const_default_value
  SRCS test_set_spec_const_default_value.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET line_debug_info
  SRCS test_line_debug_info.cpp pass_utils.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS}
)
