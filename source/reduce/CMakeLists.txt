# Copyright (c) 2018 Google LLC

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
set(SPIRV_TOOLS_REDUCE_SOURCES
        change_operand_reduction_opportunity.h
        change_operand_to_undef_reduction_opportunity.h
        conditional_branch_to_simple_conditional_branch_opportunity_finder.h
        conditional_branch_to_simple_conditional_branch_reduction_opportunity.h
        merge_blocks_reduction_opportunity.h
        merge_blocks_reduction_opportunity_finder.h
        operand_to_const_reduction_opportunity_finder.h
        operand_to_undef_reduction_opportunity_finder.h
        operand_to_dominating_id_reduction_opportunity_finder.h
        reducer.h
        reduction_opportunity.h
        reduction_opportunity_finder.h
        reduction_pass.h
        reduction_util.h
        remove_block_reduction_opportunity.h
        remove_block_reduction_opportunity_finder.h
        remove_function_reduction_opportunity.h
        remove_function_reduction_opportunity_finder.h
        remove_instruction_reduction_opportunity.h
        remove_selection_reduction_opportunity.h
        remove_selection_reduction_opportunity_finder.h
        remove_struct_member_reduction_opportunity.h
        remove_unused_instruction_reduction_opportunity_finder.h
        remove_unused_struct_member_reduction_opportunity_finder.h
        simple_conditional_branch_to_branch_opportunity_finder.h
        simple_conditional_branch_to_branch_reduction_opportunity.h
        structured_construct_to_block_reduction_opportunity.h
        structured_construct_to_block_reduction_opportunity_finder.h
        structured_loop_to_selection_reduction_opportunity.h
        structured_loop_to_selection_reduction_opportunity_finder.h

        change_operand_reduction_opportunity.cpp
        change_operand_to_undef_reduction_opportunity.cpp
        conditional_branch_to_simple_conditional_branch_opportunity_finder.cpp
        conditional_branch_to_simple_conditional_branch_reduction_opportunity.cpp
        merge_blocks_reduction_opportunity.cpp
        merge_blocks_reduction_opportunity_finder.cpp
        operand_to_const_reduction_opportunity_finder.cpp
        operand_to_undef_reduction_opportunity_finder.cpp
        operand_to_dominating_id_reduction_opportunity_finder.cpp
        reducer.cpp
        reduction_opportunity.cpp
        reduction_opportunity_finder.cpp
        reduction_pass.cpp
        reduction_util.cpp
        remove_block_reduction_opportunity.cpp
        remove_block_reduction_opportunity_finder.cpp
        remove_function_reduction_opportunity.cpp
        remove_function_reduction_opportunity_finder.cpp
        remove_instruction_reduction_opportunity.cpp
        remove_selection_reduction_opportunity.cpp
        remove_selection_reduction_opportunity_finder.cpp
        remove_struct_member_reduction_opportunity.cpp
        remove_unused_instruction_reduction_opportunity_finder.cpp
        remove_unused_struct_member_reduction_opportunity_finder.cpp
        simple_conditional_branch_to_branch_opportunity_finder.cpp
        simple_conditional_branch_to_branch_reduction_opportunity.cpp
        structured_construct_to_block_reduction_opportunity.cpp
        structured_construct_to_block_reduction_opportunity_finder.cpp
        structured_loop_to_selection_reduction_opportunity.cpp
        structured_loop_to_selection_reduction_opportunity_finder.cpp
)

if(MSVC AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
  # Enable parallel builds across four cores for this lib
  add_definitions(/MP4)
endif()

spvtools_pch(SPIRV_TOOLS_REDUCE_SOURCES pch_source_reduce)

add_library(SPIRV-Tools-reduce STATIC ${SPIRV_TOOLS_REDUCE_SOURCES})
if(SPIRV_INSTALL_STATIC_LIBS)
  set(SPIRV_TOOLS_REDUCE_STATIC SPIRV-Tools-reduce)
endif()

if(BUILD_SHARED_LIBS)
  add_library(SPIRV-Tools-reduce-shared SHARED ${SPIRV_TOOLS_REDUCE_SOURCES})
  set(SPIRV_TOOLS_REDUCE_SHARED SPIRV-Tools-reduce-shared)
endif()

spvtools_default_compile_options(SPIRV-Tools-reduce ${SPIRV_TOOLS_REDUCE_SHARED})

set(SPIRV_TOOLS_REDUCE_TARGET SPIRV-Tools-reduce ${SPIRV_TOOLS_REDUCE_SHARED})

foreach(target ${SPIRV_TOOLS_REDUCE_TARGET})
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${spirv-tools_BINARY_DIR}
  )
endforeach()

# The reducer reuses a lot of functionality from the SPIRV-Tools library.
target_link_libraries(SPIRV-Tools-reduce
  PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY}
  PUBLIC SPIRV-Tools-opt)

if(BUILD_SHARED_LIBS)
  target_link_libraries(SPIRV-Tools-reduce-shared
    PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY}-shared
    PUBLIC SPIRV-Tools-opt-shared)
endif()

set_property(TARGET SPIRV-Tools-reduce ${SPIRV_TOOLS_REDUCE_SHARED} PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-reduce ${SPIRV_TOOLS_REDUCE_SHARED})

if(ENABLE_SPIRV_TOOLS_INSTALL)
  if(BUILD_SHARED_LIBS OR SPIRV_INSTALL_STATIC_LIBS)
    if(BUILD_SHARED_LIBS)
      set_target_properties(SPIRV-Tools-reduce-shared PROPERTIES OUTPUT_NAME SPIRV-Tools-reduce)
    endif()
    if(NOT SPIRV_INSTALL_STATIC_LIBS)
        set_target_properties(SPIRV-Tools-reduce-shared PROPERTIES EXPORT_NAME SPIRV-Tools-reduce)
    endif()
    install(TARGETS ${SPIRV_TOOLS_REDUCE_STATIC} ${SPIRV_TOOLS_REDUCE_SHARED} EXPORT SPIRV-Tools-reduceTarget
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    export(EXPORT SPIRV-Tools-reduceTarget FILE SPIRV-Tools-reduceTargets.cmake)

    spvtools_config_package_dir(SPIRV-Tools-reduce PACKAGE_DIR)
    install(EXPORT SPIRV-Tools-reduceTarget FILE SPIRV-Tools-reduceTargets.cmake
      DESTINATION ${PACKAGE_DIR})

    spvtools_generate_config_file(SPIRV-Tools-reduce)
    install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-reduceConfig.cmake DESTINATION ${PACKAGE_DIR})
  endif()
endif(ENABLE_SPIRV_TOOLS_INSTALL)
