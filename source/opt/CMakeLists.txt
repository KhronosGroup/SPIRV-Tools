add_library(SpvToolsOpt
  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/def_use_manager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reflect.h
  ${CMAKE_CURRENT_SOURCE_DIR}/passes.h
  ${CMAKE_CURRENT_SOURCE_DIR}/pass_manager.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spv_builder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/type_builder.h

  ${CMAKE_CURRENT_SOURCE_DIR}/constructs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/def_use_manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spv_builder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type_builder.cpp
)

spvtools_default_compile_options(SpvToolsOpt)
target_include_directories(SpvToolsOpt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
)
