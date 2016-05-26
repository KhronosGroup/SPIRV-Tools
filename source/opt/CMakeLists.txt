add_library(SpvToolsOpt
  constructs.h
  def_use_manager.h
  reflect.h
  passes.h
  pass_manager.h
  spv_builder.h
  types.h
  type_builder.h

  constructs.cpp
  def_use_manager.cpp
  passes.cpp
  spv_builder.cpp
  types.cpp
  type_builder.cpp
)

spvtools_default_compile_options(SpvToolsOpt)
target_include_directories(SpvToolsOpt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
)
