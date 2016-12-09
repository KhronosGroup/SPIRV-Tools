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

set(GRAMMAR_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_grammar_tables.py")
set(XML_REGISTRY_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_registry_tables.py")

# macro() definitions are used in the following because we need to append .inc
# file paths into some global lists (*_CPP_DEPENDS). And those global lists are
# later used by set_source_files_properties() calls.
# function() definitions are not suitable because they create new scopes.
macro(spvtools_core_tables VERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${VERSION}/spirv.core.grammar.json")
  set(GRAMMAR_INSTS_INC_FILE "${spirv-tools_BINARY_DIR}/core.insts-${VERSION}.inc")
  set(GRAMMAR_KINDS_INC_FILE "${spirv-tools_BINARY_DIR}/operand.kinds-${VERSION}.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INSTS_INC_FILE} ${GRAMMAR_KINDS_INC_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --core-insts-output=${GRAMMAR_INSTS_INC_FILE}
      --operand-kinds-output=${GRAMMAR_KINDS_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for SPIR-V v${VERSION} core instructions and operands.")
  list(APPEND OPCODE_CPP_DEPENDS ${GRAMMAR_INSTS_INC_FILE})
  list(APPEND OPERAND_CPP_DEPENDS ${GRAMMAR_KINDS_INC_FILE})
endmacro(spvtools_core_tables)

macro(spvtools_glsl_tables VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${VERSION}/spirv.core.grammar.json")
  set(GLSL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${VERSION}/extinst.glsl.std.450.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/glsl.std.450.insts-${VERSION}.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${CORE_GRAMMAR_JSON_FILE}
      --extinst-glsl-grammar=${GLSL_GRAMMAR_JSON_FILE}
      --glsl-insts-output=${GRAMMAR_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${GLSL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for GLSL extended instructions and operands v${VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_glsl_tables)

macro(spvtools_opencl_tables VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${VERSION}/spirv.core.grammar.json")
  set(OPENCL_GRAMMAR_JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/extinst-${VERSION}.opencl.std.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/opencl.std.insts-${VERSION}.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${CORE_GRAMMAR_JSON_FILE}
      --extinst-opencl-grammar=${OPENCL_GRAMMAR_JSON_FILE}
      --opencl-insts-output=${GRAMMAR_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${OPENCL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for OpenCL extended instructions and operands v${VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_opencl_tables)

spvtools_core_tables("1.0")
spvtools_core_tables("1.1")
spvtools_opencl_tables("1.0")
spvtools_glsl_tables("1.0")


# Extract the list of known generators from the SPIR-V XML registry file.
set(GENERATOR_INC_FILE ${spirv-tools_BINARY_DIR}/generators.inc)
set(SPIRV_XML_REGISTRY_FILE ${SPIRV_HEADER_INCLUDE_DIR}/spirv/spir-v.xml)
add_custom_command(OUTPUT ${GENERATOR_INC_FILE}
  COMMAND ${PYTHON_EXECUTABLE} ${XML_REGISTRY_PROCESSING_SCRIPT}
    --xml=${SPIRV_XML_REGISTRY_FILE}
    --generator-output=${GENERATOR_INC_FILE}
  DEPENDS ${XML_REGISTRY_PROCESSING_SCRIPT} ${SPIRV_XML_REGISTRY_FILE}
  COMMENT "Generate tables based on the SPIR-V XML registry.")
list(APPEND OPCODE_CPP_DEPENDS ${GENERATOR_INC_FILE})

# The following .cpp files include the above generated .inc files.
# Add those .inc files as their dependencies.
#
# Why using such an awkward way?
# * If we use add_custom_target() to define a target to generate all .inc files
#   and let ${SPIRV_TOOLS} depend on it, then we need to run ninja twice every
#   time the grammar is updated: the first time is for generating those .inc
#   files, and the second time is for rebuilding .cpp files, when ninja finds
#   out that .inc files are updated.
# * If we use add_custom_command() with PRE_BUILD, then the grammar processing
#   script will always run no matter whether the grammar is updated.
# * add_dependencies() is used to add *target* dependencies to a target.
# * The following solution only generates .inc files when the script or the
#   grammar files is updated, and in a single ninja run.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode.cpp
  PROPERTIES OBJECT_DEPENDS "${OPCODE_CPP_DEPENDS}")
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/operand.cpp
  PROPERTIES OBJECT_DEPENDS "${OPERAND_CPP_DEPENDS}")
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_inst.cpp
  PROPERTIES OBJECT_DEPENDS "${EXTINST_CPP_DEPENDS}")

set(SPIRV_TOOLS_BUILD_VERSION_INC
  ${spirv-tools_BINARY_DIR}/build-version.inc)
set(SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR
  ${spirv-tools_SOURCE_DIR}/utils/update_build_version.py)
set(SPIRV_TOOLS_CHANGES_FILE
  ${spirv-tools_SOURCE_DIR}/CHANGES)
add_custom_command(OUTPUT ${SPIRV_TOOLS_BUILD_VERSION_INC}
   COMMAND ${PYTHON_EXECUTABLE}
           ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${spirv-tools_SOURCE_DIR} ${SPIRV_TOOLS_BUILD_VERSION_INC}
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${SPIRV_TOOLS_CHANGES_FILE}
   COMMENT "Update build-version.inc in the SPIRV-Tools build directory (if necessary).")
# Convenience target for standalone generation of the build-version.inc file.
# This is not required for any dependence chain.
add_custom_target(spirv-tools-build-version
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC})

add_subdirectory(opt)

set(SPIRV_SOURCES
  ${spirv-tools_SOURCE_DIR}/include/spirv-tools/libspirv.h

  ${CMAKE_CURRENT_SOURCE_DIR}/util/bitutils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/hex_float.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/parse_number.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assembly_grammar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/enum_set.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_inst.h
  ${CMAKE_CURRENT_SOURCE_DIR}/instruction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/macro.h
  ${CMAKE_CURRENT_SOURCE_DIR}/name_mapper.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode.h
  ${CMAKE_CURRENT_SOURCE_DIR}/operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsed_operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/print.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_endian.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_target_env.h
  ${CMAKE_CURRENT_SOURCE_DIR}/table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/text.h
  ${CMAKE_CURRENT_SOURCE_DIR}/text_handler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/validate.h

  ${CMAKE_CURRENT_SOURCE_DIR}/util/parse_number.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assembly_grammar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/disassemble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_inst.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libspirv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/message.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/name_mapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parsed_operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/software_version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_endian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_target_env.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text_handler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate_cfg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate_id.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate_instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate_datarules.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/validate_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/basic_block.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/construct.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validation_state.cpp)

# The software_version.cpp file includes build-version.inc.
# Rebuild the software_version.cpp object file if it is older than
# build-version.inc or whenever build-version.inc itself is out of
# date.  In the latter case, rebuild build-version.inc first.
# CMake is not smart enough to detect this dependency automatically.
# Without this, the dependency detection system for #included files
# does not kick in on a clean build for the following reason:  The
# build will fail early because it doesn't know how to build the
# missing source file build-version.inc. That occurs before the
# preprocessor is run on software_version.cpp to detect the
# #include dependency.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/software_version.cpp
  PROPERTIES OBJECT_DEPENDS "${SPIRV_TOOLS_BUILD_VERSION_INC}")

add_library(${SPIRV_TOOLS} ${SPIRV_SOURCES})
spvtools_default_compile_options(${SPIRV_TOOLS})
target_include_directories(${SPIRV_TOOLS}
  PUBLIC ${spirv-tools_SOURCE_DIR}/include
  PRIVATE ${spirv-tools_BINARY_DIR}
  PRIVATE ${SPIRV_HEADER_INCLUDE_DIR}
  )

install(TARGETS ${SPIRV_TOOLS}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
