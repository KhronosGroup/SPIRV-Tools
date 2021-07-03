#!/usr/bin/env python
# Copyright (c) 2021 Google LLC

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

"""Script for generating protobuf C++ files from a protobuf description,
for the spirv-fuzz library.
"""

import os
import platform
import subprocess
import sys


def main():
  protoc = 'protoc'
  if platform.system() == 'Windows':
    protoc += '.exe'
  protobufs_dir_gen = os.path.abspath(sys.argv[1])
  protobufs_dir_source = os.path.abspath(sys.argv[2])
  cmd = [os.path.abspath(protoc), "-I=" + protobufs_dir_source, "--cpp_out=" + protobufs_dir_gen, os.sep.join([protobufs_dir_source, "spvtoolsfuzz.proto"])]
  ret = subprocess.call(cmd)
  return ret


if __name__ == '__main__':
  sys.exit(main())
