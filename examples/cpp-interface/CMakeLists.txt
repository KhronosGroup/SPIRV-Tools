# Copyright (c) 2016 Google Inc.
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

if (NOT ${SPIRV_SKIP_EXECUTABLES})
  add_executable(spirv-tools-cpp-example main.cpp)
  spvtools_default_compile_options(spirv-tools-cpp-example)
  target_link_libraries(spirv-tools-cpp-example PRIVATE SPIRV-Tools-opt)
endif()
