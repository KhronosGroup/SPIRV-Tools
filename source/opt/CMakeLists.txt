# Copyright (c) 2016 Google Inc.

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
add_library(SPIRV-Tools-opt
  basic_block.h
  build_module.h
  compact_ids_pass.h
  constants.h
  def_use_manager.h
  eliminate_dead_constant_pass.h
  flatten_decoration_pass.h
  function.h
  fold_spec_constant_op_and_composite_pass.h
  freeze_spec_constant_value_pass.h
  inline_pass.h
  instruction.h
  ir_loader.h
  log.h
  module.h
  null_pass.h
  reflect.h
  pass.h
  passes.h
  pass_manager.h
  set_spec_constant_default_value_pass.h
  strip_debug_info_pass.h
  types.h
  type_manager.h
  unify_const_pass.h

  basic_block.cpp
  build_module.cpp
  compact_ids_pass.cpp
  def_use_manager.cpp
  eliminate_dead_constant_pass.cpp
  flatten_decoration_pass.cpp
  function.cpp
  fold_spec_constant_op_and_composite_pass.cpp
  freeze_spec_constant_value_pass.cpp
  inline_pass.cpp
  instruction.cpp
  ir_loader.cpp
  module.cpp
  set_spec_constant_default_value_pass.cpp
  optimizer.cpp
  pass_manager.cpp
  strip_debug_info_pass.cpp
  types.cpp
  type_manager.cpp
  unify_const_pass.cpp
)

spvtools_default_compile_options(SPIRV-Tools-opt)
target_include_directories(SPIRV-Tools-opt
  PUBLIC ${spirv-tools_SOURCE_DIR}/include
  PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# We need the assembling and disassembling functionalities in the main library.
target_link_libraries(SPIRV-Tools-opt
  PUBLIC ${SPIRV_TOOLS})

set_property(TARGET SPIRV-Tools-opt PROPERTY FOLDER "SPIRV-Tools libraries")

install(TARGETS SPIRV-Tools-opt
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
