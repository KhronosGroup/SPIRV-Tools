#!/bin/bash
# Copyright (c) 2018 Google LLC.
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
#
# Android Build Script.

# Fail on any error.
set -e
# Display commands being run.
set -x

. /bin/using.sh

BUILD_ROOT=$PWD
SRC=$PWD/github/SPIRV-Tools

# This is required to run any git command in the docker since owner will
# have changed between the clone environment, and the docker container.
# Marking the root of the repo as safe for ownership changes.
git config --global --add safe.directory $SRC

using clang-13.0.1

cd $SRC
git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers external/spirv-headers
git clone https://github.com/google/googletest          external/googletest
cd external && cd googletest && git reset --hard 1fb1bb23bb8418dc73a5a9a82bbed31dc610fec7 && cd .. && cd ..
git clone --depth=1 https://github.com/google/effcee              external/effcee
git clone --depth=1 https://github.com/google/re2                 external/re2
# The --fail flag causes the command to fail on HTTP error response codes, like 404.
curl -L --fail https://raw.githubusercontent.com/llvm/llvm-project/main/clang/tools/clang-format/clang-format-diff.py -o utils/clang-format-diff.py

echo $(date): Check formatting...
./utils/check_code_format.sh;
echo $(date): check completed.
