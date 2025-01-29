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
BUILD_SHA=${KOKORO_GIT_COMMIT}
TARGET_BRANCH="${KOKORO_GITHUB_PULL_REQUEST_TARGET_BRANCH-main}"

set -x
exec docker run --rm -i \
  --privileged \
  --volume "${SRC_ROOT}:${SRC_ROOT}" \
  --workdir "${SRC_ROOT}" \
  --env SCRIPT_DIR=${SCRIPT_DIR} \
  --env SRC_ROOT=${SRC_ROOT} \
  --env BUILD_SHA="${BUILD_SHA}" \
  --entrypoint "${SCRIPT_DIR}/build-docker.sh" \
  "us-east4-docker.pkg.dev/shaderc-build/radial-docker/radial-ubuntu-2404-amd64-clang-format" \
  "${TARGET_BRANCH}" \
  /opt/clang-format-diff.py
