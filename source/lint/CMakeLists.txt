# Copyright (c) 2021 Google LLC.

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
set(SPIRV_TOOLS_LINT_SOURCES
  divergence_analysis.h
  lints.h

  linter.cpp
  divergence_analysis.cpp
  lint_divergent_derivatives.cpp
)

if(MSVC AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
  # Enable parallel builds across four cores for this lib.
  add_definitions(/MP4)
endif()

add_library(SPIRV-Tools-lint STATIC ${SPIRV_TOOLS_LINT_SOURCES})
if(SPIRV_INSTALL_STATIC_LIBS)
  set(SPIRV_TOOLS_LINT_STATIC SPIRV-Tools-lint)
endif()

if(BUILD_SHARED_LIBS)
  add_library(SPIRV-Tools-lint-shared SHARED ${SPIRV_TOOLS_LINT_SOURCES})
  set(SPIRV_TOOLS_LINT_SHARED SPIRV-Tools-lint-shared)
endif()

spvtools_default_compile_options(SPIRV-Tools-lint ${SPIRV_TOOLS_LINT_SHARED})

set(SPIRV_TOOLS_LINT_TARGET SPIRV-Tools-lint ${SPIRV_TOOLS_LINT_SHARED})

foreach(target ${SPIRV_TOOLS_LINT_TARGET})
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${spirv-tools_BINARY_DIR}
  )
endforeach()

# We need the assembling and disassembling functionalities in the main library.
target_link_libraries(SPIRV-Tools-lint
  PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY}
  PUBLIC SPIRV-Tools-opt)

if(BUILD_SHARED_LIBS)
  target_link_libraries(SPIRV-Tools-lint-shared
    PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY}-shared
    PUBLIC SPIRV-Tools-opt-shared)
endif()

set_property(TARGET SPIRV-Tools-lint ${SPIRV_TOOLS_LINT_SHARED} PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-lint ${SPIRV_TOOLS_LINT_SHARED})

if(ENABLE_SPIRV_TOOLS_INSTALL)
  if(BUILD_SHARED_LIBS OR SPIRV_INSTALL_STATIC_LIBS)
    if(BUILD_SHARED_LIBS)
      set_target_properties(SPIRV-Tools-lint-shared PROPERTIES OUTPUT_NAME SPIRV-Tools-lint)
    endif()
    if(NOT SPIRV_INSTALL_STATIC_LIBS)
        set_target_properties(SPIRV-Tools-lint-shared PROPERTIES EXPORT_NAME SPIRV-Tools-lint)
    endif()
    install(TARGETS ${SPIRV_TOOLS_LINT_STATIC} ${SPIRV_TOOLS_LINT_SHARED} EXPORT SPIRV-Tools-lintTarget
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    export(EXPORT SPIRV-Tools-lintTarget FILE SPIRV-Tools-lintTargets.cmake)

    spvtools_config_package_dir(SPIRV-Tools-lint PACKAGE_DIR)
    install(EXPORT SPIRV-Tools-lintTarget FILE SPIRV-Tools-lintTargets.cmake
      DESTINATION ${PACKAGE_DIR})

    spvtools_generate_config_file(SPIRV-Tools-lint)
    install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-lintConfig.cmake DESTINATION ${PACKAGE_DIR})
  endif()
endif(ENABLE_SPIRV_TOOLS_INSTALL)
