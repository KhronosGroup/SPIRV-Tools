# Copyright (c) 2023 Google LLC.
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

if (NOT "${SPIRV_SKIP_TESTS}" AND TARGET gmock_main)
  add_executable(
    spirv-tools-dump
    extract_source_test.cpp
    ${spirv-tools_SOURCE_DIR}/tools/util/cli_consumer.cpp
    extract_source.cpp)
  target_link_libraries(spirv-tools-dump PRIVATE ${SPIRV_TOOLS_FULL_VISIBILITY} SPIRV-Tools-opt)

  target_include_directories(spirv-tools-dump PRIVATE
    ${SPIRV_HEADER_INCLUDE_DIR}
    ${spirv-tools_SOURCE_DIR}
    ${spirv-tools_SOURCE_DIR}/include
    ${spirv-tools_SOURCE_DIR}/test
    ${spirv-tools_BINARY_DIR}
    ${gtest_SOURCE_DIR}/include
    ${gmock_SOURCE_DIR}/include
  )

  target_link_libraries(spirv-tools-dump PRIVATE gmock_main)
  add_test(NAME spirv-tools-dump COMMAND spirv-tools-dump)
  set_property(TARGET spirv-tools-dump PROPERTY FOLDER "SPIRV-Tools tests")
endif ()
