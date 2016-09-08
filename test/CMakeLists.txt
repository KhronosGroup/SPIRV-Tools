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
    message(STATUS "Did not find googletest, tests will not be built."
      "To enable tests place googletest in '<spirv-dir>/external/googletest'.")
  endif()
endif()

function(add_spvtools_unittest)
  if (NOT "${SPIRV_SKIP_TESTS}" AND TARGET gmock_main)
    set(one_value_args TARGET)
    set(multi_value_args SRCS LIBS)
    cmake_parse_arguments(
      ARG "" "${one_value_args}" "${multi_value_args}" ${ARGN})
    set(target test_${ARG_TARGET})
    add_executable(${target} ${ARG_SRCS})
    spvtools_default_compile_options(${target})
    if(${COMPILER_IS_LIKE_GNU})
      target_compile_options(${target} PRIVATE -Wno-undef)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
      # Disable C4503 "decorated name length exceeded" warning,
      # triggered by some heavily templated types.
      # We don't care much about that in test code.
      # Important to do since we have warnings-as-errors.
      target_compile_options(${target} PRIVATE /wd4503)
    endif()
    target_include_directories(${target} PRIVATE
      ${SPIRV_HEADER_INCLUDE_DIR}
      ${spirv-tools_SOURCE_DIR}
      ${spirv-tools_SOURCE_DIR}/include
      ${spirv-tools_SOURCE_DIR}/test
      ${gtest_SOURCE_DIR}/include
      ${gmock_SOURCE_DIR}/include
    )
    target_link_libraries(${target} PRIVATE ${ARG_LIBS})
    target_link_libraries(${target} PRIVATE gmock_main)
    add_test(NAME spirv-tools-${target} COMMAND ${target})
  endif()
endfunction()

set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitSPIRV.h

  ${CMAKE_CURRENT_SOURCE_DIR}/AssemblyContext.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AssemblyFormat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryDestroy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryEndianness.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryHeaderGet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryParse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryToText.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/BinaryToText.Literal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CapabilitySet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Comment.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ExtInstGLSLstd450.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ExtInst.OpenCL.std.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FixWord.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GeneratorMagicNumber.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HexFloat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ImmediateInt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LibspirvMacros.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NamedId.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NameMapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpcodeMake.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpcodeRequiresCapabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpcodeSplit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpcodeTableGet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperandCapabilities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperandPattern.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SoftwareVersion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TargetEnv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextAdvance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextDestroy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextLiteral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextStartsNewInst.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Annotation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Barrier.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.ControlFlow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Debug.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.DeviceSideEnqueue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Extension.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Group.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Literal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.Miscellaneous.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.ModeSetting.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.PipeStorage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.TypeDeclaration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextToBinary.SubgroupDispatch.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TextWordGet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitSPIRV.cpp
)

add_spvtools_unittest(
  TARGET spirv_unit_tests
  SRCS ${TEST_SOURCES}
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET diagnostic
  SRCS diagnostic.cpp
  LIBS ${SPIRV_TOOLS})

add_spvtools_unittest(
  TARGET cpp_interface
  SRCS cpp_interface.cpp
  LIBS SPIRV-Tools-opt ${SPIRV_TOOLS})

add_subdirectory(opt)
add_subdirectory(val)
