#!/bin/bash
# Copyright (c) 2025 Google LLC
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

# Fail on any error.
set -e

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )"
SRC_ROOT="$( cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd )"
TARGET_BRANCH="${KOKORO_GITHUB_PULL_REQUEST_TARGET_BRANCH-main}"

# Translate a path under KOKORO_ROOT_DIR to the corresponding host VM path under
# KOKORO_HOST_ROOT_DIR when running Docker sibling containers on Kokoro Instances.
function to_host_path() {
  local path="$1"
  if [ -n "${KOKORO_HOST_ROOT_DIR:-}" ] && [ -n "${KOKORO_ROOT_DIR:-}" ]; then
    local container_root="${KOKORO_ROOT_DIR%/}"
    local host_root="${KOKORO_HOST_ROOT_DIR%/}"
    if [ "${path}" = "${container_root}" ]; then
      echo "${host_root}"
    elif [[ "${path}" == "${container_root}/"* ]]; then
      echo "${host_root}${path#${container_root}}"
    else
      echo "${path}"
    fi
  else
    echo "${path}"
  fi
}

docker run --rm -i \
  --volume "$(to_host_path "${SRC_ROOT}"):${SRC_ROOT}" \
  --workdir "${SRC_ROOT}" \
  "us-east4-docker.pkg.dev/shaderc-build/radial-docker/ubuntu-24.04-amd64/formatter" \
  "${SCRIPT_DIR}/build-docker.sh" "${TARGET_BRANCH}"
