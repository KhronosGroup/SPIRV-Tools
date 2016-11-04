# Copyright (c) 2016 The Khronos Group Inc.
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

set(VAL_TEST_COMMON_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/../test_fixture.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_spirv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_spirv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ValidateFixtures.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ValidateFixtures.cpp
)


add_spvtools_unittest(TARGET val_capability
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.Capability.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_cfg
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.CFG.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_id
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ValidateID.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_layout
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.Layout.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_ssa
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.SSA.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_storage
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.Storage.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_state
  SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ValidationState.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

add_spvtools_unittest(TARGET val_data
	SRCS ${CMAKE_CURRENT_SOURCE_DIR}/Validate.Data.cpp
       ${VAL_TEST_COMMON_SRCS}
  LIBS ${SPIRV_TOOLS}
)

