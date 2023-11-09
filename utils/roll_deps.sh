#!/usr/bin/env bash
# Copyright (c) 2021 Google LLC
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

# Attempts to roll submodules to tip-of-tree and create a commit.

set -eo pipefail

function ExitIfIsInterestingError() {
  local return_code=$1
  if [[ ${return_code} -ne 0 && ${return_code} -ne 2 ]]; then
    exit ${return_code}
  fi
  return 0
}


dependencies=("external/effcee/"
              "external/googletest/"
              "external/re2/"
              "external/spirv-headers/")

# This script assumes it's parent directory is the repo root.
repo_path=$(dirname "$0")/..

cd "$repo_path"

if [[ $(git diff --stat) != '' ]]; then
    echo "Working tree is dirty, commit changes before attempting to roll dependencies"
    exit 1
fi

set +e

for dep in ${dependencies[@]}; do
  echo "Rolling $dep"
  git submodule update --init --remote -- ${dep}
  ExitIfIsInterestingError $?
done

if [[ $(git diff --stat) != '' ]]; then
  echo "Committing updates"
  git commit -am "Roll dependencies"
fi
