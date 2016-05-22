add_library(SpvToolsOpt
  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.h

  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.cpp
)

spvtools_default_compile_options(SpvToolsOpt)
target_include_directories(SpvToolsOpt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
)
