// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_SPIRV_TARGET_ENV_H_
#define SOURCE_SPIRV_TARGET_ENV_H_

#include <string>
#include <utility>
#include <vector>

#include "spirv-tools/libspirv.h"

// Returns true if |env| is a VULKAN environment, false otherwise.
bool spvIsVulkanEnv(spv_target_env env);

// Returns true if |env| is an OPENCL environment, false otherwise.
bool spvIsOpenCLEnv(spv_target_env env);

// Returns true if |env| is an OPENGL environment, false otherwise.
bool spvIsOpenGLEnv(spv_target_env env);

// Returns true if |env| is an implemented/valid environment, false otherwise.
bool spvIsValidEnv(spv_target_env env);

// Returns the version number for the given SPIR-V target environment.
uint32_t spvVersionForTargetEnv(spv_target_env env);

// Returns a string to use in logging messages that indicates the class of
// environment, i.e. "Vulkan", "OpenCL", etc.
std::string spvLogStringForEnv(spv_target_env env);

// Returns a formatted list of all SPIR-V target environment names that
// can be parsed by spvParseTargetEnv.
// |pad| is the number of space characters that the beginning of each line
//       except the first one will be padded with.
// |wrap| is the max length of lines the user desires. Word-wrapping will
//        occur to satisfy this limit.
std::string spvTargetEnvList(const int pad, const int wrap);

// Reads the target environment from the header comments of disassembly. Returns
// true if valid name found, false otherwise.
bool spvReadEnvironmentFromText(std::vector<char>& text, spv_target_env* env);

static constexpr std::pair<const char*, spv_target_env> spvTargetEnvNameMap[] =
    {
        {"vulkan1.1spv1.4", SPV_ENV_VULKAN_1_1_SPIRV_1_4},
        {"vulkan1.0", SPV_ENV_VULKAN_1_0},
        {"vulkan1.1", SPV_ENV_VULKAN_1_1},
        {"vulkan1.2", SPV_ENV_VULKAN_1_2},
        {"vulkan1.3", SPV_ENV_VULKAN_1_3},
        {"spv1.0", SPV_ENV_UNIVERSAL_1_0},
        {"spv1.1", SPV_ENV_UNIVERSAL_1_1},
        {"spv1.2", SPV_ENV_UNIVERSAL_1_2},
        {"spv1.3", SPV_ENV_UNIVERSAL_1_3},
        {"spv1.4", SPV_ENV_UNIVERSAL_1_4},
        {"spv1.5", SPV_ENV_UNIVERSAL_1_5},
        {"spv1.6", SPV_ENV_UNIVERSAL_1_6},
        {"opencl1.2embedded", SPV_ENV_OPENCL_EMBEDDED_1_2},
        {"opencl1.2", SPV_ENV_OPENCL_1_2},
        {"opencl2.0embedded", SPV_ENV_OPENCL_EMBEDDED_2_0},
        {"opencl2.0", SPV_ENV_OPENCL_2_0},
        {"opencl2.1embedded", SPV_ENV_OPENCL_EMBEDDED_2_1},
        {"opencl2.1", SPV_ENV_OPENCL_2_1},
        {"opencl2.2embedded", SPV_ENV_OPENCL_EMBEDDED_2_2},
        {"opencl2.2", SPV_ENV_OPENCL_2_2},
        {"opengl4.0", SPV_ENV_OPENGL_4_0},
        {"opengl4.1", SPV_ENV_OPENGL_4_1},
        {"opengl4.2", SPV_ENV_OPENGL_4_2},
        {"opengl4.3", SPV_ENV_OPENGL_4_3},
        {"opengl4.5", SPV_ENV_OPENGL_4_5},
        {"assume", SPV_ENV_MAX},
};
static constexpr auto kAssumeIndex = 25;
// The index in the map of the first SPV_ENV_UNIVERSAL version, assuming that
// versions are enumerated in order from 1.0
static constexpr auto kSpvEnvUniversalStart = 5;

#endif  // SOURCE_SPIRV_TARGET_ENV_H_
