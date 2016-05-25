# Copyright (c) 2016 Google Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
# KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
# SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
#    https://www.khronos.org/registry/
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

add_spvtools_unittest(types
  ${CMAKE_CURRENT_SOURCE_DIR}/test_types.cpp
  SpvToolsOpt
)

add_spvtools_unittest(type_builder
  ${CMAKE_CURRENT_SOURCE_DIR}/test_type_builder.cpp
  "SpvToolsOpt;${SPIRV_TOOLS}"
)

add_spvtools_unittest(spv_builder
  ${CMAKE_CURRENT_SOURCE_DIR}/test_spv_builder.cpp
  "SpvToolsOpt;${SPIRV_TOOLS}"
)

add_spvtools_unittest(opt_passes
  ${CMAKE_CURRENT_SOURCE_DIR}/test_passes.cpp
  "SpvToolsOpt;${SPIRV_TOOLS}"
)
