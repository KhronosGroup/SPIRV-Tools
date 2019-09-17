// Copyright 2019 The Khronos Group Inc.
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

import { assert } from "chai";
import Lexer from "./lexer";
import Parser from "./parser";
import grammar from "./spirv.data.js";
import Assembler from "./assembler";

describe("assembler", () => {
  it("generates SPIR-V magic number", () => {
    let input = `; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 7
; Bound: 6
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 440
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd`;
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast, p.error);

    let a = new Assembler(ast);
    let res = a.assemble();
    assert.equal(0x07230203, res[0]);
  });

  it("assembles enumerant params", () => {
    let input = "OpExecutionMode %main LocalSize 2 3 4";

    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast, p.error);

    let a = new Assembler(ast);
    let res = a.assemble();

    assert.lengthOf(res, 11);
    assert.equal((6 /* word count */ << 16) | 16 /* opcode */, res[5]);
    assert.equal(1 /* %main */, res[6]);
    assert.equal(17 /* LocalSize */, res[7]);
    assert.equal(2, res[8]);
    assert.equal(3, res[9]);
    assert.equal(4, res[10]);
  });

  it("assembles float 32 values", () => {
    let input = `%float = OpTypeFloat 32
                 %float1 = OpConstant %float 0.400000006`;
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast, p.error);

    let a = new Assembler(ast);
    let res = a.assemble();

    assert.lengthOf(res, 12);
    assert.equal((4 /* word count */ << 16) | 43 /* opcode */, res[8]);
    assert.equal(1 /* %float */, res[9]);
    assert.equal(2 /* %float */, res[10]);
    assert.equal(0x3ecccccd /* 0.400000006 */, res[11]);
  });
});
