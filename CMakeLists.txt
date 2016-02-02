# Copyright (c) 2015-2016 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
# KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
# SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
#    https://www.khronos.org/registry/
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

cmake_minimum_required(VERSION 2.8)
project(spirv-tools)
enable_testing()
set(SPIRV_TOOLS "SPIRV-Tools")

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  add_definitions(-DSPIRV_LINUX)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  add_definitions(-DSPIRV_MAC)
else()
  message(FATAL_ERROR "Your platform '${CMAKE_SYSTEM_NAME}' is not supported!")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

option(SPIRV_WERROR "Enable error on warning" ON)
if(UNIX)
  set(SPIRV_WARNINGS -Wall -Wextra -Wno-missing-field-initializers)

  option(SPIRV_WARN_EVERYTHING "Enable -Weverything" ${SPIRV_WARN_EVERYTHING})
  if(${SPIRV_WARN_EVERYTHING})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(SPIRV_WARNINGS ${SPIRV_WARNINGS}
        -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Wpedantic -pedantic-errors)
    else()
      message(STATUS "Unknown compiler ${CMAKE_CXX_COMPILER_ID}, "
                     "so SPIRV_WARN_EVERYTHING has no effect")
    endif()
  endif()

  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Werror)
  endif()
elseif(WIN32)
  set(SPIRV_WARNINGS -D_CRT_SECURE_NO_WARNINGS /wd4800)

  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} /WX)
  endif()
endif()

option(SPIRV_COLOR_TERMINAL "Enable color terminal output" ON)
if(${SPIRV_COLOR_TERMINAL})
  add_definitions(-DSPIRV_COLOR_TERMINAL)
endif()

function(spvtools_default_compile_options TARGET)
  target_compile_options(${TARGET} PRIVATE ${SPIRV_WARNINGS})
  if (UNIX)
    target_compile_options(${TARGET} PRIVATE
      -std=c++11 -fno-exceptions -fno-rtti)
    target_compile_options(${TARGET} PRIVATE
      -Wall -Wextra -Wno-long-long -Wshadow -Wundef -Wconversion
      -Wno-sign-conversion)
    # For good call stacks in profiles, keep the frame pointers.
    if(NOT "${SPIRV_PERF}" STREQUAL "")
      target_compile_options(${TARGET} PRIVATE -fno-omit-frame-pointer)
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(SPIRV_USE_SANITIZER "" CACHE STRING
        "Use the clang sanitizer [address|memory|thread|...]")
      if(NOT "${SPIRV_USE_SANITIZER}" STREQUAL "")
        target_compile_options(${TARGET} PRIVATE
          -fsanitize=${SPIRV_USE_SANITIZER})
      endif()
    else()
      target_compile_options(${TARGET} PRIVATE
         -Wno-missing-field-initializers)
    endif()
  endif()
endfunction()

set(SPIRV_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/util/bitutils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/util/hex_float.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/assembly_grammar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/binary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/diagnostic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/ext_inst.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/instruction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/opcode.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/print.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/spirv_constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/spirv_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/spirv_endian.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text_handler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/assembly_grammar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/disassemble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/diagnostic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/ext_inst.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/opcode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/spirv_endian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text_handler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_cfg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_id.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_ssa.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_types.cpp)

add_library(${SPIRV_TOOLS} ${SPIRV_SOURCES})
spvtools_default_compile_options(${SPIRV_TOOLS})
target_include_directories(${SPIRV_TOOLS} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/include)

# Defaults to OFF if the user didn't set it.
option(SPIRV_SKIP_EXECUTABLES
       "Skip building the executable and tests along with the library"
       ${SPIRV_SKIP_EXECUTABLES})
set(SPIRV_INSTALL_TARGETS ${SPIRV_TOOLS})
if (NOT ${SPIRV_SKIP_EXECUTABLES})
  list(APPEND SPIRV_INSTALL_TARGETS spirv-as spirv-dis spirv-val)

  add_executable(spirv-as ${CMAKE_CURRENT_SOURCE_DIR}/tools/as/as.cpp)
  spvtools_default_compile_options(spirv-as)
  target_link_libraries(spirv-as PRIVATE ${SPIRV_TOOLS})

  add_executable(spirv-dis ${CMAKE_CURRENT_SOURCE_DIR}/tools/dis/dis.cpp)
  spvtools_default_compile_options(spirv-dis)
  target_link_libraries(spirv-dis PRIVATE ${SPIRV_TOOLS})

  add_executable(spirv-val ${CMAKE_CURRENT_SOURCE_DIR}/tools/val/val.cpp)
  spvtools_default_compile_options(spirv-val)
  target_link_libraries(spirv-val PRIVATE ${SPIRV_TOOLS})

  set(GMOCK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/googlemock)
  if(EXISTS ${GMOCK_DIR})
    if(WIN32)
      option(gtest_force_shared_crt
        "Use shared (DLL) run-time lib even when Google Test is built as static lib."
        ON)
    endif()

    add_subdirectory(${GMOCK_DIR})
  endif()
  if (TARGET gmock)
    message(STATUS "Found Google Mock, building tests.")

    set(TEST_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TestFixture.h
      ${CMAKE_CURRENT_SOURCE_DIR}/test/UnitSPIRV.h

      ${CMAKE_CURRENT_SOURCE_DIR}/test/AssemblyContext.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/AssemblyFormat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryDestroy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryEndianness.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryHeaderGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryParse.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryToText.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryToText.Literal.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Comment.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/DiagnosticDestroy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/DiagnosticPrint.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/DiagnosticStream.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ExtInstGLSLstd450.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ExtInst.OpenCL.std.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/FixWord.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/GeneratorMagicNumber.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/HexFloat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ImmediateInt.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/LibspirvMacros.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/NamedId.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeMake.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeRequiresCapabilities.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeSplit.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeTableGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OperandCapabilities.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Operand.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OperandPattern.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextAdvance.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextDestroy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextLiteral.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextStartsNewInst.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Annotation.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Barrier.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Constant.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.ControlFlow.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Debug.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.DeviceSideEnqueue.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Extension.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Function.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Group.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Image.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Literal.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Memory.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Miscellaneous.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.ModeSetting.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.TypeDeclaration.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextWordGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/UnitSPIRV.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ValidateFixtures.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Validate.Capability.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Validate.Layout.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Validate.Storage.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Validate.SSA.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ValidateID.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ValidationState.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp)

    add_executable(UnitSPIRV ${TEST_SOURCES})
    spvtools_default_compile_options(UnitSPIRV)
    if(UNIX)
      target_compile_options(UnitSPIRV PRIVATE -Wno-undef)
    endif()
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
      # Disable C4503 "decorated name length exceeded" warning, triggered
      # by some heavily templated types.
      # We don't care much about that in test code.
      # Important to do since we have warnings-as-errors.
      target_compile_options(UnitSPIRV PRIVATE /wd4503)
    endif()
    target_include_directories(UnitSPIRV PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${gmock_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/include)
    target_link_libraries(UnitSPIRV PRIVATE ${SPIRV_TOOLS} gmock)
    add_test(NAME spirv-tools-testsuite COMMAND UnitSPIRV)
  else()
    message(STATUS "Did not find googletest, tests will not be built."
      "To enable tests place googletest in '<spirv-dir>/external/googletest'.")
  endif()
endif()

install(TARGETS ${SPIRV_INSTALL_TARGETS}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
  DESTINATION include/libspirv/)
