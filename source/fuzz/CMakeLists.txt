# Copyright (c) 2019 Google LLC

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
set(SPIRV_TOOLS_FUZZ_SOURCES
        fuzzer.h
        fuzzer_context.h
        fuzzer_pass.h
        fuzzer_pass_add_dead_breaks.h
        fuzzer_pass_add_useful_constructs.h
        fuzzer_pass_permute_blocks.h
        fuzzer_pass_split_blocks.h
        pseudo_random_generator.h
        random_generator.h
        transformation.h
        transformation_add_boolean_constant.h
        transformation_add_dead_break.h
        transformation_move_block_down.h
        transformation_split_block.h

        fuzzer.cpp
        fuzzer_context.cpp
        fuzzer_pass_add_dead_breaks.cpp
        fuzzer_pass_add_useful_constructs.cpp
        fuzzer_pass_permute_blocks.cpp
        fuzzer_pass_split_blocks.cpp
        pseudo_random_generator.cpp
        transformation.cpp
        transformation_add_boolean_constant.cpp
        transformation_add_dead_break.cpp
        transformation_move_block_down.cpp
        transformation_split_block.cpp
        )

if(MSVC)
    # Enable parallel builds across four cores for this lib
    add_definitions(/MP4)
endif()

spvtools_pch(SPIRV_TOOLS_FUZZ_SOURCES pch_source_fuzz)

add_library(SPIRV-Tools-fuzz ${SPIRV_TOOLS_FUZZ_SOURCES})

spvtools_default_compile_options(SPIRV-Tools-fuzz)
target_include_directories(SPIRV-Tools-fuzz
        PUBLIC ${spirv-tools_SOURCE_DIR}/include
        PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
        PRIVATE ${spirv-tools_BINARY_DIR}
        )
# The fuzzer reuses a lot of functionality from the SPIRV-Tools library.
target_link_libraries(SPIRV-Tools-fuzz
        PUBLIC ${SPIRV_TOOLS}
        PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-fuzz PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-fuzz)

if(ENABLE_SPIRV_TOOLS_INSTALL)
    install(TARGETS SPIRV-Tools-fuzz
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
