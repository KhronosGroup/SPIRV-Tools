add_library(SpvToolsOpt
  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spv_builder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/types.h

  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spv_builder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
)

spvtools_default_compile_options(SpvToolsOpt)
target_include_directories(SpvToolsOpt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
)
