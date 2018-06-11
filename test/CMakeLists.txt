# Copyright (c) 2015-2016 The Khronos Group Inc.
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

# Add a SPIR-V Tools unit test. Signature:
#   add_spvtools_unittest(
#     TARGET target_name
#     SRCS   src_file.h src_file.cpp
#     LIBS   lib1 lib2
#   )

if (NOT "${SPIRV_SKIP_TESTS}")
  if (TARGET gmock_main)
    message(STATUS "Found Google Mock, building tests.")
  else()
    message(STATUS "Did not find googletest, tests will not be built. "
      "To enable tests place googletest in '<spirv-dir>/external/googletest'.")
  endif()
endif()

function(add_spvtools_unittest)
  if (NOT "${SPIRV_SKIP_TESTS}" AND TARGET gmock_main)
    set(one_value_args TARGET)
    set(multi_value_args SRCS LIBS ENVIRONMENT)
    cmake_parse_arguments(
      ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})
    set(target test_${ARG_TARGET})
    add_executable(${target} ${ARG_SRCS})
    spvtools_default_compile_options(${target})
    if(${COMPILER_IS_LIKE_GNU})
      target_compile_options(${target} PRIVATE -Wno-undef)
      # Effcee and RE2 headers exhibit shadowing.
      target_compile_options(${target} PRIVATE -Wno-shadow)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      # Disable C4503 "decorated name length exceeded" warning,
      # triggered by some heavily templated types.
      # We don't care much about that in test code.
      # Important to do since we have warnings-as-errors.
      target_compile_options(${target} PRIVATE /wd4503)
      # Googletest accidentally turns off support for ::testing::Combine
      # in VS 2017.  See https://github.com/google/googletest/issues/1352
      # Forcibly turn it on again.
      target_compile_options(${target} PRIVATE /DGTEST_HAS_COMBINE=1)
    endif()
    target_include_directories(${target} PRIVATE
      ${SPIRV_HEADER_INCLUDE_DIR}
      ${spirv-tools_SOURCE_DIR}
      ${spirv-tools_SOURCE_DIR}/include
      ${spirv-tools_SOURCE_DIR}/test
      ${spirv-tools_BINARY_DIR}
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include
    )
    if (TARGET effcee)
      # If using Effcee for testing, then add its include directory.
      target_include_directories(${target} PRIVATE ${effcee_SOURCE_DIR})
    endif()
    target_link_libraries(${target} PRIVATE ${ARG_LIBS})
    if (TARGET effcee)
      target_link_libraries(${target} PRIVATE effcee)
    endif()
    target_link_libraries(${target} PRIVATE gmock_main)
    add_test(NAME spirv-tools-${target} COMMAND ${target})
    if (DEFINED ARG_ENVIRONMENT)
      set_tests_properties(spirv-tools-${target} PROPERTIES ENVIRONMENT ${ARG_ENVIRONMENT})
    endif()
    set_property(TARGET ${target} PROPERTY FOLDER "SPIRV-Tools tests")
  endif()
endfunction()

set(TEST_SOURCES
  test_fixture.h
  unit_spirv.h

  assembly_context_test.cpp
  assembly_format_test.cpp
  binary_destroy_test.cpp
  binary_endianness_test.cpp
  binary_header_get_test.cpp
  binary_parse_test.cpp
  binary_strnlen_s_test.cpp
  binary_to_text_test.cpp
  binary_to_text.literal_test.cpp
  bit_stream.cpp
  c_interface_test.cpp
  comment_test.cpp
  cpp_interface_test.cpp
  diagnostic_test.cpp
  enum_string_mapping_test.cpp
  enum_set_test.cpp
  ext_inst.debuginfo_test.cpp
  ext_inst.glsl_test.cpp
  ext_inst.opencl_test.cpp
  fix_word_test.cpp
  generator_magic_number_test.cpp
  hex_float_test.cpp
  huffman_codec.cpp
  immediate_int_test.cpp
  libspirv_macros_test.cpp
  link/binary_version_test.cpp
  link/memory_model_test.cpp
  link/entry_points_test.cpp
  link/global_values_amount_test.cpp
  link/ids_limit_test.cpp
  link/matching_imports_to_exports_test.cpp
  link/unique_ids_test.cpp
  link/partial_linkage_test.cpp
  log_test.cpp
  move_to_front_test.cpp
  named_id_test.cpp
  name_mapper_test.cpp
  opcode_make_test.cpp
  opcode_require_capabilities_test.cpp
  opcode_split_test.cpp
  opcode_table_get_test.cpp
  operand_capabilities_test.cpp
  operand_test.cpp
  operand_pattern_test.cpp
  opt/aggressive_dead_code_elim_test.cpp
  opt/assembly_builder_test.cpp
  opt/block_merge_test.cpp
  opt/cfg_cleanup_test.cpp
  opt/compact_ids_test.cpp
  opt/common_uniform_elim_test.cpp
  opt/copy_prop_array_test.cpp
  opt/ccp_test.cpp
  opt/dead_branch_elim_test.cpp
  opt/dead_insert_elim_test.cpp
  opt/dead_variable_elim_test.cpp
  opt/decoration_manager_test.cpp
  opt/def_use_test.cpp
  opt/dominator_tree/common_dominators.cpp
  opt/dominator_tree/generated.cpp
  opt/dominator_tree/nested_ifs.cpp
  opt/dominator_tree/nested_ifs_post.cpp
  opt/dominator_tree/nested_loops.cpp
  opt/dominator_tree/nested_loops_with_unreachables.cpp
  opt/dominator_tree/post.cpp
  opt/dominator_tree/simple.cpp
  opt/dominator_tree/switch_case_fallthrough.cpp
  opt/dominator_tree/unreachable_for.cpp
  opt/dominator_tree/unreachable_for_post.cpp
  opt/eliminate_dead_const_test.cpp
  opt/eliminate_dead_functions_test.cpp
  opt/feature_manager_test.cpp
  opt/flatten_decoration_test.cpp
  opt/fold_spec_const_op_composite_test.cpp
  opt/fold_test.cpp
  opt/freeze_spec_const_test.cpp
  opt/function_utils.cpp
  opt/if_conversion_test.cpp
  opt/inline_test.cpp
  opt/inline_opaque_test.cpp
  opt/insert_extract_elim_test.cpp
  opt/instruction_test.cpp
  opt/instruction_list_test.cpp
  opt/ir_builder.cpp
  opt/ir_context_test.cpp
  opt/ir_loader_test.cpp
  opt/iterator_test.cpp
  opt/line_debug_info_test.cpp
  opt/local_redundancy_elimination_test.cpp
  opt/local_ssa_elim_test.cpp
  opt/local_single_block_elim.cpp
  opt/local_access_chain_convert_test.cpp
  opt/local_single_store_elim_test.cpp
  opt/loop_optimizations/dependence_analysis.cpp
  opt/loop_optimizations/dependence_analysis_helpers.cpp
  opt/loop_optimizations/fusion_compatibility.cpp
  opt/loop_optimizations/fusion_illegal.cpp
  opt/loop_optimizations/fusion_legal.cpp
  opt/loop_optimizations/fusion_pass.cpp
  opt/loop_optimizations/hoist_all_loop_types.cpp
  opt/loop_optimizations/hoist_double_nested_loops.cpp
  opt/loop_optimizations/hoist_from_independent_loops.cpp
  opt/loop_optimizations/hoist_simple_case.cpp
  opt/loop_optimizations/hoist_single_nested_loops.cpp
  opt/loop_optimizations/hoist_without_preheader.cpp
  opt/loop_optimizations/lcssa.cpp
  opt/loop_optimizations/loop_descriptions.cpp
  opt/loop_optimizations/loop_fission.cpp
  opt/loop_optimizations/nested_loops.cpp
  opt/loop_optimizations/peeling.cpp
  opt/loop_optimizations/peeling_pass.cpp
  opt/loop_optimizations/unroll_assumptions.cpp
  opt/loop_optimizations/unroll_simple.cpp
  opt/loop_optimizations/unswitch.cpp
  opt/module_test.cpp
  opt/optimizer_test.cpp
  opt/pass_manager_test.cpp
  opt/pass_merge_return_test.cpp
  opt/pass_remove_duplicates_test.cpp
  opt/pass_test.cpp
  opt/pass_utils.cpp
  opt/private_to_local_test.cpp
  opt/propagator_test.cpp
  opt/reduce_load_size_test.cpp
  opt/redundancy_elimination_test.cpp
  opt/register_liveness.cpp
  opt/replace_invalid_opc_test.cpp
  opt/scalar_analysis.cpp
  opt/scalar_replacement_test.cpp
  opt/set_spec_const_default_value_test.cpp
  opt/simplification_test.cpp
  opt/strength_reduction_test.cpp
  opt/strip_debug_info_test.cpp
  opt/strip_reflect_info_test.cpp
  opt/types_test.cpp
  opt/type_manager_test.cpp
  opt/unify_const_test.cpp
  opt/utils_test.cpp
  opt/value_table_test.cpp
  opt/vector_dce_test.cpp
  opt/workaround1209_test.cpp
  parse_number_test.cpp
  preserve_numeric_ids_test.cpp
  software_version_test.cpp
  stats/stats_aggregate_test.cpp
  stats/stats_analyzer_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/stats/stats_analyzer.cpp

  string_utils_test.cpp
  target_env_test.cpp
  text_advance_test.cpp
  text_destroy_test.cpp
  text_literal_test.cpp
  text_start_new_inst_test.cpp
  text_to_binary.annotation_test.cpp
  text_to_binary.barrier_test.cpp
  text_to_binary.constant_test.cpp
  text_to_binary.control_flow_test.cpp
  text_to_binary_test.cpp
  text_to_binary.debug_test.cpp
  text_to_binary.device_side_enqueue_test.cpp
  text_to_binary.extension_test.cpp
  text_to_binary.function_test.cpp
  text_to_binary.group_test.cpp
  text_to_binary.image_test.cpp
  text_to_binary.literal_test.cpp
  text_to_binary.memory_test.cpp
  text_to_binary.misc_test.cpp
  text_to_binary.mode_setting_test.cpp
  text_to_binary.pipe_storage_test.cpp
  text_to_binary.type_declaration_test.cpp
  text_to_binary.subgroup_dispatch_test.cpp
  text_to_binary.reserved_sampling_test.cpp
  text_word_get_test.cpp

  unit_spirv.cpp
  util/ilist_test.cpp
  util/bit_vector_test.cpp

  val/val_capability_test.cpp
  val/val_cfg_test.cpp
  val/val_id_test.cpp
  val/val_layout_test.cpp
  val/val_ssa_test.cpp
  val/val_storage_test.cpp
  val/val_state_test.cpp
  val/val_data_test.cpp
  val/val_type_unique_test.cpp
  val/val_arithmetics_test.cpp
  val/val_composites_test.cpp
  val/val_conversion_test.cpp
  val/val_derivatives_test.cpp
  val/val_logicals_test.cpp
  val/val_bitwise_test.cpp
  val/val_builtins_test.cpp
  val/val_image_test.cpp
  val/val_atomics_test.cpp
  val/val_barriers_test.cpp
  val/val_primitives_test.cpp
  val/val_ext_inst_test.cpp
  val/val_limits_test.cpp
  val/val_validation_state_test.cpp
  val/val_decoration_test.cpp
  val/val_literals_test.cpp
  val/val_extensions_test.cpp
  val/val_adjacency_test.cpp
  val/val_version_test.cpp
  val/val_non_uniform_test.cpp
)

if (${SPIRV_TIMER_ENABLED})
set(TEST_SOURCES ${TEST_SOURCES}
  timer_test.cpp
)
endif()

if(SPIRV_BUILD_COMPRESSION)
set(TEST_SOURCES ${TEST_SOURCES}
  comp/markv_codec_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/comp/markv_model_factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../tools/comp/markv_model_shader.cpp
)
endif(SPIRV_BUILD_COMPRESSION)

add_spvtools_unittest(
  TARGET spirv_unit_tests
  SRCS ${TEST_SOURCES}
  LIBS ${SPIRV_TOOLS}
       SPIRV-Tools-comp
       SPIRV-Tools-opt
       SPIRV-Tools-link)

add_spvtools_unittest(
  TARGET c_interface_shared
  SRCS c_interface_test.cpp
  LIBS ${SPIRV_TOOLS}-shared
  ENVIRONMENT PATH=$<TARGET_FILE_DIR:${SPIRV_TOOLS}-shared>)
