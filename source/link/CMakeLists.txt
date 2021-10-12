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

add_library(SPIRV-Tools-link STATIC linker.cpp)
if(SPIRV_INSTALL_STATIC_LIBS)
  set(SPIRV_TOOLS_LINK_STATIC SPIRV-Tools-link)
endif()

if(BUILD_SHARED_LIBS)
  add_library(SPIRV-Tools-link-shared SHARED linker.cpp)
  set(SPIRV_TOOLS_LINK_SHARED SPIRV-Tools-link-shared)
endif()

spvtools_default_compile_options(SPIRV-Tools-link ${SPIRV_TOOLS_LINK_SHARED})

set(SPIRV_TOOLS_LINK_TARGET SPIRV-Tools-link ${SPIRV_TOOLS_LINK_SHARED})

foreach(target ${SPIRV_TOOLS_LINK_TARGET})
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${spirv-tools_BINARY_DIR}
  )
endforeach()

# We need the IR functionnalities from the optimizer
target_link_libraries(SPIRV-Tools-link
  PUBLIC SPIRV-Tools-opt)

if(SPIRV_TOOLS_BUILD_SHARED_LIBS)
  target_link_libraries(SPIRV-Tools-link-shared
    PUBLIC SPIRV-Tools-opt-shared)
endif()

set_property(TARGET SPIRV-Tools-link ${SPIRV_TOOLS_LINK_SHARED} PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-link ${SPIRV_TOOLS_LINK_SHARED})

if(ENABLE_SPIRV_TOOLS_INSTALL)
  if(BUILD_SHARED_LIBS OR SPIRV_INSTALL_STATIC_LIBS)
    if(BUILD_SHARED_LIBS)
      set_target_properties(SPIRV-Tools-link-shared PROPERTIES OUTPUT_NAME SPIRV-Tools-link)
    endif()
    if(NOT SPIRV_INSTALL_STATIC_LIBS)
        set_target_properties(SPIRV-Tools-link-shared PROPERTIES EXPORT_NAME SPIRV-Tools-link)
    endif()
    install(TARGETS ${SPIRV_TOOLS_LINK_STATIC} ${SPIRV_TOOLS_LINK_SHARED} EXPORT SPIRV-Tools-linkTarget
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    export(EXPORT SPIRV-Tools-linkTarget FILE SPIRV-Tools-linkTargets.cmake)

    spvtools_config_package_dir(SPIRV-Tools-link PACKAGE_DIR)
    install(EXPORT SPIRV-Tools-linkTarget FILE SPIRV-Tools-linkTargets.cmake
      DESTINATION ${PACKAGE_DIR})

    spvtools_generate_config_file(SPIRV-Tools-link)
    install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-linkConfig.cmake DESTINATION ${PACKAGE_DIR})
  endif()
endif(ENABLE_SPIRV_TOOLS_INSTALL)
