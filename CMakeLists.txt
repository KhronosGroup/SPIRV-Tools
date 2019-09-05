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

cmake_minimum_required(VERSION 3.0)
cmake_policy(VERSION 3.0)
project(spirv-tools VERSION 2019.5 LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()


list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPIRV_ALLOW_TIMERS "Allow timers via clock_gettime on supported platforms" ON)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  add_definitions(-DSPIRV_LINUX)
  set(SPIRV_TIMER_ENABLED ${SPIRV_ALLOW_TIMERS})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
  add_definitions(-DSPIRV_WINDOWS)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  add_definitions(-DSPIRV_MAC)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
  add_definitions(-DSPIRV_ANDROID)
  set(SPIRV_TIMER_ENABLED ${SPIRV_ALLOW_TIMERS})
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
  add_definitions(-DSPIRV_FREEBSD)
else()
  message(FATAL_ERROR "Your platform '${CMAKE_SYSTEM_NAME}' is not supported!")
endif()

if (${SPIRV_TIMER_ENABLED})
  add_definitions(-DSPIRV_TIMER_ENABLED)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

if(DEFINED SKIP_SPIRV_TOOLS_INSTALL)
  message(DEPRECATION "SKIP_SPIRV_TOOLS_INSTALL has been removed. "
                      "It should be removed from your build options.")
endif()

if(DEFINED SPIRV_BUILD_COMPRESSION)
  message(FATAL_ERROR "SPIR-V compression codec has been removed from SPIR-V tools. "
          "Please remove SPIRV_BUILD_COMPRESSION from your build options.")
endif()

option(SPIRV_WERROR "Enable error on warning" ON)
if(("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU") OR (("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") AND (NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")))
  set(COMPILER_IS_LIKE_GNU TRUE)
endif()
if(${COMPILER_IS_LIKE_GNU})
  set(SPIRV_WARNINGS -Wall -Wextra -Wnon-virtual-dtor -Wno-missing-field-initializers)

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} -Wno-self-assign)
  endif()

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
elseif(MSVC)
  set(SPIRV_WARNINGS -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /wd4800)

  if(${SPIRV_WERROR})
    set(SPIRV_WARNINGS ${SPIRV_WARNINGS} /WX)
  endif()
endif()

option(SPIRV_COLOR_TERMINAL "Enable color terminal output" ON)
if(${SPIRV_COLOR_TERMINAL})
  add_definitions(-DSPIRV_COLOR_TERMINAL)
endif()

option(SPIRV_LOG_DEBUG "Enable excessive debug output" OFF)
if(${SPIRV_LOG_DEBUG})
  add_definitions(-DSPIRV_LOG_DEBUG)
endif()

if (DEFINED SPIRV_TOOLS_EXTRA_DEFINITIONS)
  add_definitions(${SPIRV_TOOLS_EXTRA_DEFINITIONS})
endif()

function(spvtools_default_compile_options TARGET)
  target_compile_options(${TARGET} PRIVATE ${SPIRV_WARNINGS})

  if (${COMPILER_IS_LIKE_GNU})
    target_compile_options(${TARGET} PRIVATE
      -fno-exceptions -fno-rtti
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
        target_link_libraries(${TARGET} PRIVATE
          -fsanitize=${SPIRV_USE_SANITIZER})
      endif()
      target_compile_options(${TARGET} PRIVATE
         -ftemplate-depth=1024)
    else()
      target_compile_options(${TARGET} PRIVATE
         -Wno-missing-field-initializers)
    endif()
  endif()

  if (MSVC)
    # Specify /EHs for exception handling. This makes using SPIRV-Tools as
    # dependencies in other projects easier.
    target_compile_options(${TARGET} PRIVATE /EHs)
  endif()

  # For MinGW cross compile, statically link to the C++ runtime.
  # But it still depends on MSVCRT.dll.
  if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
      target_link_libraries(${TARGET} PRIVATE
        -static -static-libgcc -static-libstdc++)
    endif()
  endif()
endfunction()

if(NOT COMMAND find_host_package)
  macro(find_host_package)
    find_package(${ARGN})
  endmacro()
endif()
if(NOT COMMAND find_host_program)
  macro(find_host_program)
    find_program(${ARGN})
  endmacro()
endif()

# Tests require Python3
find_host_package(PythonInterp 3 REQUIRED)
find_host_package(nosetests)

# First look for the headers in external/ using the FindSPIRV-Headers.cmake
# then look for installed headers.
find_package(SPIRV-Headers MODULE)
if(NOT SPIRV-Headers_FOUND)
  find_package(SPIRV-Headers CONFIG)
endif()

# Check for symbol exports on Linux.
# At the moment, this check will fail on the OSX build machines for the Android NDK.
# It appears they don't have objdump.
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  macro(spvtools_check_symbol_exports TARGET)
    if (NOT "${SPIRV_SKIP_TESTS}")
      add_test(NAME spirv-tools-symbol-exports-${TARGET}
               COMMAND ${PYTHON_EXECUTABLE}
               ${spirv-tools_SOURCE_DIR}/utils/check_symbol_exports.py "$<TARGET_FILE:${TARGET}>")
    endif()
  endmacro()
else()
  macro(spvtools_check_symbol_exports TARGET)
    if (NOT "${SPIRV_SKIP_TESTS}")
      message("Skipping symbol exports test for ${TARGET}")
    endif()
  endmacro()
endif()

# Defaults to OFF if the user didn't set it.
option(SPIRV_SKIP_EXECUTABLES
  "Skip building the executable and tests along with the library"
  ${SPIRV_SKIP_EXECUTABLES})
option(SPIRV_SKIP_TESTS
  "Skip building tests along with the library" ${SPIRV_SKIP_TESTS})
if ("${SPIRV_SKIP_EXECUTABLES}")
  set(SPIRV_SKIP_TESTS ON)
endif()

# Defaults to ON.  The checks can be time consuming.
# Turn off if they take too long.
option(SPIRV_CHECK_CONTEXT "In a debug build, check if the IR context is in a valid state." ON)
if (${SPIRV_CHECK_CONTEXT})
  add_definitions(-DSPIRV_CHECK_CONTEXT)
endif()

# Precompiled header macro. Parameters are source file list and filename for pch cpp file.
macro(spvtools_pch SRCS PCHPREFIX)
  if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
    set(PCH_NAME "$(IntDir)\\${PCHPREFIX}.pch")
    # make source files use/depend on PCH_NAME
    set_source_files_properties(${${SRCS}} PROPERTIES COMPILE_FLAGS "/Yu${PCHPREFIX}.h /FI${PCHPREFIX}.h /Fp${PCH_NAME} /Zm300" OBJECT_DEPENDS "${PCH_NAME}")
    # make PCHPREFIX.cpp file compile and generate PCH_NAME
    set_source_files_properties("${PCHPREFIX}.cpp" PROPERTIES COMPILE_FLAGS "/Yc${PCHPREFIX}.h /Fp${PCH_NAME} /Zm300" OBJECT_OUTPUTS "${PCH_NAME}")
    list(APPEND ${SRCS} "${PCHPREFIX}.cpp")
  endif()
endmacro(spvtools_pch)

add_library(SPIRV-Tools-Headers INTERFACE)
target_include_directories(SPIRV-Tools-Headers
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
            $<INSTALL_INTERFACE:include>)
target_link_libraries(SPIRV-Tools-Headers
  INTERFACE $<BUILD_INTERFACE:SPIRV-Headers::SPIRV-Headers>)
install(TARGETS SPIRV-Tools-Headers EXPORT SPIRV-Tools)

add_subdirectory(external)

add_subdirectory(source)
add_subdirectory(tools)

add_subdirectory(test)
add_subdirectory(examples)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (NOT "${SPIRV_SKIP_TESTS}")
  add_test(NAME spirv-tools-copyrights
           COMMAND ${PYTHON_EXECUTABLE} utils/check_copyright.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(SPIRV_LIBRARIES "-lSPIRV-Tools -lSPIRV-Tools-link -lSPIRV-Tools-opt")
set(SPIRV_SHARED_LIBRARIES "-lSPIRV-Tools-shared")

# Build pkg-config file
# Use a first-class target so it's regenerated when relevant files are updated.
add_custom_command(OUTPUT SPIRV-Tools.pc
  COMMAND ${CMAKE_COMMAND}
    -DCHANGES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
    -DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools.pc.in
    -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools.pc
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DSPIRV_LIBRARIES=${SPIRV_LIBRARIES}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
  MAIN_DEPENDENCY
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools.pc.in
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake)

add_custom_command(OUTPUT SPIRV-Tools-shared.pc
  COMMAND ${CMAKE_COMMAND}
    -DCHANGES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
    -DTEMPLATE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools-shared.pc.in
    -DOUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools-shared.pc
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DSPIRV_SHARED_LIBRARIES=${SPIRV_SHARED_LIBRARIES}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
  MAIN_DEPENDENCY
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SPIRV-Tools-shared.pc.in
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/CHANGES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake)

add_custom_target(spirv-tools-pkg-config ALL DEPENDS SPIRV-Tools.pc)
add_custom_target(spirv-tools-shared-pkg-config ALL DEPENDS SPIRV-Tools-shared.pc)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools.pc
    ${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Tools-shared.pc
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-ToolsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "cmake/SPIRV-ToolsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-ToolsConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SPIRV-Tools"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-ToolsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-ToolsConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SPIRV-Tools"
)
install(
    EXPORT SPIRV-Tools
    NAMESPACE SPIRV-Tools::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SPIRV-Tools"
)
