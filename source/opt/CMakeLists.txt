add_library(SPIRV-Tools-opt
  basic_block.h
  function.h
  instruction.h
  module.h
  reflect.h

  function.cpp
  instruction.cpp
  module.cpp
)

spvtools_default_compile_options(SPIRV-Tools-opt)
target_include_directories(SPIRV-Tools-opt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
  PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
)
