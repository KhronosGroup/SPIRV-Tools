// Copyright (c) 2020 The Khronos Group Inc.
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

const spirvTools = require("../../out/web/spirv-tools/spirv-tools");
const spirvToolsOpt = require("../../out/web/spirv-tools-opt/spirv-tools-opt");

const test = async () => {
  const spv = await spirvTools();

  const source = `
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
               OpName %i "i"
               OpName %i2 "i2"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_2 = OpConstant %int 2
       %main = OpFunction %void None %3
          %5 = OpLabel
          %i = OpVariable %_ptr_Function_int Function
         %i2 = OpVariable %_ptr_Function_int Function
               OpStore %i %int_2
         %11 = OpLoad %int %i
         %12 = OpLoad %int %i
         %13 = OpIAdd %int %11 %12
               OpStore %i2 %13
               OpReturn
               OpFunctionEnd `;


  const asResult = spv.as(
    source,
    spv.SPV_ENV_UNIVERSAL_1_0,
    spv.SPV_TEXT_TO_BINARY_OPTION_NONE
  );
  console.log(`as returned ${asResult.length} bytes`);

  console.log(`\nNow optimize for performance ... `);
  // Optimize for performance.
  const opt = await spirvToolsOpt();
  const optResult = opt.optimizePerformance(opt.SPV_ENV_VULKAN_1_0,asResult);


  console.log(`optimizePerformance returned ${optResult.length} bytes`);

  // disassemble
  const disResult = spv.dis(
    optResult,
    spv.SPV_ENV_UNIVERSAL_1_0,
    spv.SPV_BINARY_TO_TEXT_OPTION_INDENT |
      spv.SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
      spv.SPV_BINARY_TO_TEXT_OPTION_COLOR
  );
  console.log("dis:\n", disResult);

};

test();
