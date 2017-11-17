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
  aggressive_dead_code_elim_pass.h
  basic_block.h
  block_merge_pass.h
  build_module.h
  cfg_cleanup_pass.h
  cfg.h
  common_uniform_elim_pass.h
  compact_ids_pass.h
  constants.h
  dead_branch_elim_pass.h
  dead_variable_elimination.h
  decoration_manager.h
  def_use_manager.h
  eliminate_dead_constant_pass.h
  eliminate_dead_functions_pass.h
  flatten_decoration_pass.h
  fold.h
  fold_spec_constant_op_and_composite_pass.h
  freeze_spec_constant_value_pass.h
  function.h
  inline_exhaustive_pass.h
  inline_opaque_pass.h
  inline_pass.h
  insert_extract_elim.h
  instruction.h
  ir_context.h
  ir_loader.h
  local_access_chain_convert_pass.h
  local_redundancy_elimination.h
  local_single_block_elim_pass.h
  local_single_store_elim_pass.h
  local_ssa_elim_pass.h
  log.h
  mem_pass.h
  merge_return_pass.h
  module.h
  null_pass.h
  passes.h
  pass.h
  pass_manager.h
  propagator.h
  reflect.h
  remove_duplicates_pass.h
  set_spec_constant_default_value_pass.h
  strength_reduction_pass.h
  strip_debug_info_pass.h
  type_manager.h
  types.h
  unify_const_pass.h
  value_number_table.h

  aggressive_dead_code_elim_pass.cpp
  basic_block.cpp
  block_merge_pass.cpp
  build_module.cpp
  cfg_cleanup_pass.cpp
  cfg.cpp
  common_uniform_elim_pass.cpp
  compact_ids_pass.cpp
  dead_branch_elim_pass.cpp
  dead_variable_elimination.cpp
  decoration_manager.cpp
  def_use_manager.cpp
  eliminate_dead_constant_pass.cpp
  eliminate_dead_functions_pass.cpp
  flatten_decoration_pass.cpp
  fold.cpp
  fold_spec_constant_op_and_composite_pass.cpp
  freeze_spec_constant_value_pass.cpp
  function.cpp
  inline_exhaustive_pass.cpp
  inline_opaque_pass.cpp
  inline_pass.cpp
  insert_extract_elim.cpp
  instruction.cpp
  instruction_list.cpp
  ir_context.cpp
  ir_loader.cpp
  local_access_chain_convert_pass.cpp
  local_redundancy_elimination.cpp
  local_single_block_elim_pass.cpp
  local_single_store_elim_pass.cpp
  local_ssa_elim_pass.cpp
  mem_pass.cpp
  merge_return_pass.cpp
  module.cpp
  optimizer.cpp
  pass.cpp
  pass_manager.cpp
  propagator.cpp
  remove_duplicates_pass.cpp
  set_spec_constant_default_value_pass.cpp
  strength_reduction_pass.cpp
  strip_debug_info_pass.cpp
  type_manager.cpp
  types.cpp
  unify_const_pass.cpp
  value_number_table.cpp
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
spvtools_check_symbol_exports(SPIRV-Tools-opt)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-opt
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
    
