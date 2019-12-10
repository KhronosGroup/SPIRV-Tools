#!/usr/bin/env python
# Copyright (c) 2019 Google LLC
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

import os
import re
import sys

HEADER_FILE_TEMPLATE = """// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_FUZZ_TRANSFORMATION_%UPPER%_H_
#define SOURCE_FUZZ_TRANSFORMATION_%UPPER%_H_

#include "source/fuzz/fact_manager.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class Transformation%CAMEL% : public Transformation {
 public:
  explicit Transformation%CAMEL%(
      const protobufs::Transformation%CAMEL%& message);

  Transformation%CAMEL%(/* TODO */);

  // TODO comment
  bool IsApplicable(opt::IRContext* context,
                    const FactManager& fact_manager) const override;

  // TODO comment
  void Apply(opt::IRContext* context, FactManager* fact_manager) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::Transformation%CAMEL% message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_%UPPER%_H_
"""

CPP_FILE_TEMPLATE = """// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/fuzz/transformation_%LOWER%.h"

namespace spvtools {
namespace fuzz {

Transformation%CAMEL%::Transformation%CAMEL%(
    const spvtools::fuzz::protobufs::Transformation%CAMEL%& message)
    : message_(message) {}

Transformation%CAMEL%::Transformation%CAMEL%(/* TODO */) {
  assert(false && "Not implemented yet");
}

bool Transformation%CAMEL%::IsApplicable(
    opt::IRContext* /*context*/,
    const spvtools::fuzz::FactManager& /*unused*/) const {
  assert(false && "Not implemented yet");
  return false;
}

void Transformation%CAMEL%::Apply(
    opt::IRContext* /*context*/, spvtools::fuzz::FactManager* /*unused*/) const {
  assert(false && "Not implemented yet");
}

protobufs::Transformation Transformation%CAMEL%::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_%LOWER%() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
"""

TEST_FILE_TEMPLATE = """// Copyright (c) 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/fuzz/transformation_%LOWER%.h"
#include "test/fuzz/fuzz_test_util.h"

namespace spvtools {
namespace fuzz {
namespace {

TEST(Transformation%CAMEL%Test, BasicTest) {
  std::string shader = R"(
  )";

  const auto env = SPV_ENV_UNIVERSAL_1_4;
  const auto consumer = nullptr;
  const auto context = BuildModule(env, consumer, shader, kFuzzAssembleOption);
  ASSERT_TRUE(IsValid(env, context.get()));

  FactManager fact_manager;

  // TODO - add test content

  std::string after_transformation = R"(
  )";
  ASSERT_TRUE(IsEqual(env, after_transformation, context.get()));
  FAIL(); // Remove once test is implemented
}

}  // namespace
}  // namespace fuzz
}  // namespace spvtools
"""

lower_case_with_underscores = sys.argv[1]
camel_case = ""
start_of_word = True
for c in lower_case_with_underscores:
  if c == '_':
    start_of_word = True
  else:
    camel_case += c.upper() if start_of_word else c
    start_of_word = False

upper_case_with_underscores = lower_case_with_underscores.upper()

source_fuzz_dir = 'source' + os.sep + 'fuzz' + os.sep
test_fuzz_dir = 'test' + os.sep + 'fuzz' + os.sep

open(source_fuzz_dir + 'transformation_' + lower_case_with_underscores + '.h', 'w').write(HEADER_FILE_TEMPLATE.replace('%CAMEL%', camel_case).replace('%UPPER%', upper_case_with_underscores))

open(source_fuzz_dir + 'transformation_' + lower_case_with_underscores + '.cpp', 'w').write(CPP_FILE_TEMPLATE.replace('%CAMEL%', camel_case).replace('%LOWER%', lower_case_with_underscores))

open(test_fuzz_dir + 'transformation_' + lower_case_with_underscores + '_test.cpp', 'w').write(TEST_FILE_TEMPLATE.replace('%CAMEL%', camel_case).replace('%LOWER%', lower_case_with_underscores))

protobufs_file = source_fuzz_dir + 'protobufs' + os.sep + 'spvtoolsfuzz.proto'
protobufs_lines = open(protobufs_file, 'r').readlines()
with open(protobufs_file, 'w') as outfile:
  prev_line = None
  looking_for_transformation_message_slot = False
  for line in protobufs_lines:
    if '// Add additional option using the next available number.' in line:
      next_number = int(re.findall(r'\d+', prev_line)[0]) + 1
      outfile.write('    Transformation' + camel_case + ' ' + lower_case_with_underscores + ' = ' + str(next_number) + ';\n')

    if '// Keep transformation message types in alphabetical order:' in line:
      looking_for_transformation_message_slot = True

    if looking_for_transformation_message_slot and 'message Transformation' in line:
      new_message_start_line = 'message Transformation' + camel_case + ' {\n'
      if new_message_start_line < line:
        outfile.write(new_message_start_line + '\n')
        outfile.write('  // TODO comment and populate\n\n')
        outfile.write('}\n\n')
        looking_for_transformation_message_slot = False

    outfile.write(line)
    prev_line = line

cmake_source_file = source_fuzz_dir + 'CMakeLists.txt'
cmake_source_lines = open(cmake_source_file, 'r').readlines()
with open(cmake_source_file, 'w') as outfile:
  header_line = '        transformation_' + lower_case_with_underscores + '.h\n'
  cpp_line = '        transformation_' + lower_case_with_underscores + '.cpp\n'

  state = 0 # 0: not ready; 1: looking for header slow; 2: looking for cpp slot; 3. done
  for line in cmake_source_lines:
    if 'set(SPIRV_TOOLS_FUZZ_SOURCES' in line:
      state = 1
    elif state == 1 and '.h' in line and header_line < line:
      outfile.write(header_line)
      state = 2
    elif state == 2 and '.cpp' in line and cpp_line < line:
      outfile.write(cpp_line)
      state = 3
    outfile.write(line)

cmake_test_file = test_fuzz_dir + 'CMakeLists.txt'
cmake_test_lines = open(cmake_test_file, 'r').readlines()
with open(cmake_test_file, 'w') as outfile:
  handled = False
  test_file = '          transformation_' + lower_case_with_underscores + '_test.cpp\n'
  for line in cmake_test_lines:
    if not handled and '.cpp' in line and test_file < line:
      outfile.write(test_file)
      handled = True
    outfile.write(line)

transformation_cpp_file = source_fuzz_dir + 'transformation.cpp'
transformation_cpp_lines = open(transformation_cpp_file, 'r').readlines()
with open(transformation_cpp_file, 'w') as outfile:
  handled_case = False
  handled_include = False
  for i in range(0, len(transformation_cpp_lines)):
    if not handled_include and transformation_cpp_lines[i].startswith('#include "source/fuzz/transformation_'):
      include_line = '#include "source/fuzz/transformation_' + lower_case_with_underscores + '.h"\n'
      if include_line < transformation_cpp_lines[i]:
        handled_include = True
        outfile.write(include_line)
    if not handled_case and transformation_cpp_lines[i].strip().startswith('case'):
      case_line = transformation_cpp_lines[i].strip()
      if case_line.endswith('::') or not case_line.endswith(':'):
        case_line += transformation_cpp_lines[i + 1].strip()
      new_case_line = 'case protobufs::Transformation::TransformationCase::k' + camel_case + ':'
      if new_case_line < case_line:
        handled_case = True
        outfile.write('    ' + new_case_line + '\n')
        outfile.write('      return MakeUnique<Transformation' + camel_case + '>(\n')
        outfile.write('          message.' + lower_case_with_underscores + '());\n')
    outfile.write(transformation_cpp_lines[i])
