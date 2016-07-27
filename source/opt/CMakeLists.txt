add_library(SPIRV-Tools-opt
  basic_block.h
  def_use_manager.h
  function.h
  instruction.h
  ir_loader.h
  libspirv.hpp
  module.h
  reflect.h
  passes.h
  pass_manager.h

  def_use_manager.cpp
  function.cpp
  instruction.cpp
  ir_loader.cpp
  libspirv.cpp
  module.cpp
  passes.cpp
)

spvtools_default_compile_options(SPIRV-Tools-opt)
target_include_directories(SPIRV-Tools-opt
  PRIVATE ${spirv-tools_SOURCE_DIR}/include
  PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
)
