# Copyright (c) 2015 The Khronos Group Inc.
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

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  add_definitions(-DSPIRV_LINUX)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  add_definitions(-DSPIRV_MAC)
else()
  message(FATAL_ERROR "Your platform '${CMAKE_SYSTEM_NAME}' is not supported!")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
if (UNIX)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
elseif(WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fno-exceptions -fno-rtti")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(SPIRV_USE_SANITIZER "" CACHE STRING
      "Use the clang sanitizer [address|memory|thread|...]")
    if(NOT "${SPIRV_USE_SANITIZER}" STREQUAL "")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=${SPIRV_USE_SANITIZER}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${SPIRV_USE_SANITIZER}")
    endif()
  endif()
endif()

if(UNIX)
  option(SPIRV_COLOR_TERMINAL "Enable color terminal output" ON)
  if(${SPIRV_COLOR_TERMINAL})
    add_definitions(-DSPV_COLOR_TERMINAL)
  endif()
endif()

if(UNIX)
  set(SPIRV_WARNINGS "-Wall -Wextra -Wno-missing-field-initializers")
  option(SPIRV_WARN_EVERYTHING "Enable -Weverything for SPIRV library" OFF)
  if(${SPIRV_WARN_EVERYTHING})
    set(SPIRV_WARNINGS
      "${SPIRV_WARNINGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded")
  endif()
elseif(WIN32)
  set(SPIRV_WARNINGS "-D_CRT_SECURE_NO_WARNINGS /wd4800")
endif()

if(UNIX)
  option(SPIRV_WERROR "Enable error on warning" OFF)
  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS "${SPIRV_WARNINGS} -Werror")
  endif()
endif()

include_directories(SYSTEM
  ${CMAKE_CURRENT_SOURCE_DIR}/external/include)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SPIRV_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/bitwisecast.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/binary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/diagnostic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/opcode.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/print.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text_handler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/source/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/diagnostic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/ext_inst.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/opcode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/text_handler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source/validate_id.cpp)

add_library(SPIRV-TOOLS ${SPIRV_SOURCES})
target_link_libraries(SPIRV-TOOLS ${SPIRV_LIBS})
set_target_properties(SPIRV-TOOLS PROPERTIES COMPILE_FLAGS ${SPIRV_WARNINGS})

option(SPIRV_SKIP_EXECUTABLES "Skip building the executables and tests along with the library" ${SPIRV_SKIP_EXECUTABLES})

set(SPIRV_BUILT_TARGETS SPIRV-TOOLS)
if (NOT ${SPIRV_SKIP_EXECUTABLES})
  list(APPEND SPIRV_BUILT_TARGETS spirv-as spirv-dis spirv-val)
  add_executable(spirv-as
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/as/as.cpp)
  set_target_properties(spirv-as PROPERTIES COMPILE_FLAGS ${SPIRV_WARNINGS})
  target_link_libraries(spirv-as SPIRV-TOOLS)

  add_executable(spirv-dis
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/dis/dis.cpp)
  set_target_properties(spirv-dis PROPERTIES COMPILE_FLAGS ${SPIRV_WARNINGS})
  target_link_libraries(spirv-dis SPIRV-TOOLS)

  add_executable(spirv-val
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/val/val.cpp)
  set_target_properties(spirv-val PROPERTIES COMPILE_FLAGS ${SPIRV_WARNINGS})
  target_link_libraries(spirv-val SPIRV-TOOLS)

  set(GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest)
  if(EXISTS ${GTEST_DIR})
    if(WIN32)
      option(gtest_force_shared_crt
        "Use shared (DLL) run-time lib even when Google Test is built as static lib."
        ON)
    endif()

    add_subdirectory(${GTEST_DIR})
  endif()
  if (TARGET gmock)
    message(STATUS "Found Google Mock, building tests.")

    include_directories(SYSTEM
      ${gmock_SOURCE_DIR}/include
      ${gtest_SOURCE_DIR}/include)

    add_executable(UnitSPIRV
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TestFixture.h
      ${CMAKE_CURRENT_SOURCE_DIR}/test/UnitSPIRV.h

      ${CMAKE_CURRENT_SOURCE_DIR}/test/AssemblyFormat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryDestroy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryEndianness.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryHeaderGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/BinaryToText.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Comment.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/DiagnosticPrint.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ExtInstGLSLstd450.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/FixWord.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ImmediateInt.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/LibspirvMacros.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/NamedId.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OperandPattern.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeMake.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeRequiresCapabilities.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeSplit.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OpcodeTableGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/OperandTableGet.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextAdvance.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextDestroy.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextLiteral.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextStartsNewInst.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Annotation.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Barrier.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Constant.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.ControlFlow.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.Debug.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/TextToBinary.DeviceSideEnqueue.cpp
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
      ${CMAKE_CURRENT_SOURCE_DIR}/test/Validate.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/ValidateID.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp)
    target_link_libraries(UnitSPIRV SPIRV-TOOLS gmock)
  else()
    message(STATUS "Did not find googletest, tests will not be built."
      "To enable tests place googletest in '<spirv-dir>/external/googletest'.")
  endif()
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
install(TARGETS ${SPIRV_BUILT_TARGETS}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/libspirv/libspirv.h
  DESTINATION include/libspirv/)
