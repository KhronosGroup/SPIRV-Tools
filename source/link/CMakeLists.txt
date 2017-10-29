# Copyright (c) 2017 Pierre Moreau

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

add_library(SPIRV-Tools-link "")

include(GenerateExportHeader)
generate_export_header(SPIRV-Tools-link)

target_sources(SPIRV-Tools-link
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/spirv-tools-link_export.h"
    "linker.cpp"
)

spvtools_default_compile_options(SPIRV-Tools-link)
target_include_directories(SPIRV-Tools-link
  PUBLIC
    ${spirv-tools_SOURCE_DIR}/include
    ${SPIRV_HEADER_INCLUDE_DIR}
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  PRIVATE
    ${spirv-tools_BINARY_DIR}
)
# We need the IR functionnalities from the optimizer
target_link_libraries(SPIRV-Tools-link
  PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-link PROPERTY FOLDER "SPIRV-Tools libraries")

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-link
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/spirv-tools-link_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spirv-tools
  )
endif(ENABLE_SPIRV_TOOLS_INSTALL)
