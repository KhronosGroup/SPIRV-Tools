# Copyright (c) 2018 Google LLC.
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

if(CMAKE_VERSION VERSION_LESS "3.12")
  add_test(NAME spirv-tools_expect_unittests
           COMMAND ${PYTHON_EXECUTABLE} -m unittest expect_unittest.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME spirv-tools_spirv_test_framework_unittests
           COMMAND ${PYTHON_EXECUTABLE} -m unittest spirv_test_framework_unittest.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
  add_test(NAME spirv-tools_expect_unittests
           COMMAND ${Python3_EXECUTABLE} -m unittest expect_unittest.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME spirv-tools_spirv_test_framework_unittests
           COMMAND ${Python3_EXECUTABLE} -m unittest spirv_test_framework_unittest.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
add_subdirectory(opt)
