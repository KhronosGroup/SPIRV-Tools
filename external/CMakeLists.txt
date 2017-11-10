# Copyright (c) 2015-2016 The Khronos Group Inc.
#
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

if (DEFINED SPIRV-Headers_SOURCE_DIR)
  # This allows flexible position of the SPIRV-Headers repo.
  set(SPIRV_HEADER_DIR ${SPIRV-Headers_SOURCE_DIR})
else()
  set(SPIRV_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers)
endif()

if (IS_DIRECTORY ${SPIRV_HEADER_DIR})
  set(SPIRV_HEADER_INCLUDE_DIR ${SPIRV_HEADER_DIR}/include PARENT_SCOPE)
else()
  message(FATAL_ERROR
    "SPIRV-Headers was not found - please checkout a copy under external/.")
endif()

if (NOT ${SPIRV_SKIP_EXECUTABLES})
  # Find gmock if we can. If it's not already configured, then try finding
  # it in external/googletest.
  if (TARGET gmock)
    message(STATUS "Google Mock already configured")
  else()
    set(GMOCK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock)
    if(EXISTS ${GMOCK_DIR})
      if(WIN32)
        option(gtest_force_shared_crt
          "Use shared (DLL) run-time lib even when Google Test is built as static lib."
          ON)
      endif()
      add_subdirectory(${GMOCK_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()
  if (TARGET gmock)
    set(GTEST_TARGETS
      gtest
      gtest_main
      gmock
      gmock_main
    )
    foreach(target ${GTEST_TARGETS})
      set_property(TARGET ${target} PROPERTY FOLDER GoogleTest)
    endforeach()
  endif()

  # Find Effcee and RE2, for testing.
  # Optional for now, but eventually we'll make this required.

  # First find RE2, since Effcee depends on it.
  # TODO(dneto): I intend to require this eventually.
  if (NOT TARGET re2)
    # If we're configuring RE2, then turn off RE2 testing.  It takes a
    # very long time.  If an enclosing project has configured RE2, then
    # it has already chosen whether to enable testing.
    set(RE2_BUILD_TESTING OFF)
    # Expect to find re2 in this directory.  If not, then let Effcee
    # find it for us, possibly under its own third_party directory.
    if (EXISTS re2)
      add_subdirectory(re2)
      set_property(TARGET re2 PROPERTY FOLDER RE2)
    endif()
  endif()

  # TODO(dneto): I intend to require this eventually.
  if (NOT TARGET effcee)
    # Expect to find effcee in this directory.
    if (EXISTS effcee)
      # If we're configuring RE2 (via Effcee), then turn off RE2 testing.
      if (NOT TARGET re2)
        set(RE2_BUILD_TESTING OFF)
      endif()
      add_subdirectory(effcee)
      set_property(TARGET effcee PROPERTY FOLDER Effcee)
    endif()
  endif()
  if (TARGET effcee)
    message(STATUS "SPIRV-Tools: Effcee is configured")
  else()
    message(STATUS "SPIRV-Tools: Effcee is not configured.  Skipping Effcee-based tests.")
  endif()

endif()
