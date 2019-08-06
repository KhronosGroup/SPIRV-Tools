find_path(SPIRV_INCLUDE_DIRS "spirv/unified1/spirv.h"
  HINTS
    ${SPIRV-Headers_SOURCE_DIR}
    "${PROJECT_SOURCE_DIR}/external/SPIRV-Headers/include"
    "${PROJECT_SOURCE_DIR}/external/spirv-headers/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SPIRV-Headers
  FOUND_VAR SPIRV-Headers_FOUND
  REQUIRED_VARS
    SPIRV_INCLUDE_DIRS
  FAIL_MESSAGE "SPIRV-Headers was not found - please set CMAKE_PREFIX_PATH to the installation prefix of the headers \
  or SPIRV-Headers_DIR to the path of the SPIRV-HeadersConfig.cmake or \
  checkout a copy under external/."
)

if(SPIRV-Headers_FOUND AND NOT TARGET SPIRV-Headers::SPIRV-Headers)
  add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
  set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${SPIRV_INCLUDE_DIRS}
  )

  mark_as_advanced(SPIRV_INCLUDE_DIRS)
endif()
