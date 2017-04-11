#!/usr/bin/env python
# Copyright (c) 2017 Google Inc.

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
"""Tests correctness of opt pass tools/opt --compact-ids."""

from __future__ import print_function

import os.path
import sys

TEST_HOME_PATH = '/tmp/test_compact_ids'
OPTIMIZED_SPV_PATH = TEST_HOME_PATH + '/optimized.spv'
OPTIMIZED_DIS_PATH = TEST_HOME_PATH + '/optimized.dis'
CONVERTED_SPV_PATH = TEST_HOME_PATH + '/converted.spv'
CONVERTED_DIS_PATH = TEST_HOME_PATH + '/converted.dis'

def test_spirv_file(path):
  os.system('tools/spirv-opt ' + path + ' -o ' + OPTIMIZED_SPV_PATH +
            ' --compact-ids')
  os.system('tools/spirv-dis ' + OPTIMIZED_SPV_PATH + ' -o ' +
            OPTIMIZED_DIS_PATH)

  os.system('tools/spirv-dis ' + path + ' -o ' + CONVERTED_DIS_PATH)
  os.system('tools/spirv-as ' + CONVERTED_DIS_PATH + ' -o ' +
            CONVERTED_SPV_PATH)
  os.system('tools/spirv-dis ' + CONVERTED_SPV_PATH + ' -o ' +
            CONVERTED_DIS_PATH)

  #os.system('diff ' + CONVERTED_DIS_PATH + ' ' + OPTIMIZED_DIS_PATH)
  with open(CONVERTED_DIS_PATH, 'r') as f:
    converted_dis = f.readlines()[3:]

  with open(OPTIMIZED_DIS_PATH, 'r') as f:
    optimized_dis = f.readlines()[3:]

  return converted_dis == optimized_dis

def print_usage():
  template= \
"""{script} tests correctness of opt pass tools/opt --compact-ids

USAGE: python {script} [<spirv_files>]

Requires tools/spirv-dis, tools/spirv-as and tools/spirv-opt to be in path
(call the script from the SPIRV-Tools build output directory).

TIP: In order to test all .spv files under current dir use
find <path> -name "*.spv" -print0 | xargs -0 -s 2000000 python {script}
"""
  print(template.format(script=sys.argv[0]));

def main():
  if not os.path.isfile('tools/spirv-dis'):
      print('error: tools/spirv-dis not found')
      print_usage()
      exit(1)

  if not os.path.isfile('tools/spirv-as'):
      print('error: tools/spirv-as not found')
      print_usage()
      exit(1)

  if not os.path.isfile('tools/spirv-opt'):
      print('error: tools/spirv-opt not found')
      print_usage()
      exit(1)

  paths = sys.argv[1:]
  if not paths:
      print_usage()

  os.system('rm -rf ' + TEST_HOME_PATH)
  os.system('mkdir ' + TEST_HOME_PATH)

  num_failed = 0

  for path in paths:
    success = test_spirv_file(path)
    if not success:
      print('Test failed for ' + path)
      num_failed += 1

  print('Tested ' + str(len(paths)) + ' files')

  if num_failed:
    print(str(num_failed) + ' tests failed')
    exit(1)
  else:
    print('All tests successful')
    exit(0)

if __name__ == '__main__':
  main()
