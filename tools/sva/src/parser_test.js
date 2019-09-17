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

describe("parser", () => {
  it("parses an opcode", () => {
    let input = "OpKill";
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.lengthOf(ast.instructions(), 1);

    let inst = ast.instruction(0);
    assert.equal("OpKill", inst.name());
    assert.equal(252, inst.opcode());
    assert.lengthOf(inst.operands, 0);
  });

  it("parses an opcode with an identifier", () => {
    let input = "OpCapability Shader";
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast, p.error);
    assert.lengthOf(ast.instructions(), 1);

    let inst = ast.instruction(0);
    assert.equal("OpCapability", inst.name());
    assert.equal(17, inst.opcode());
    assert.lengthOf(inst.operands(), 1);

    let op = inst.operand(0);
    assert.equal("Shader", op.name());
    assert.equal("ValueEnum", op.type());
    assert.equal(1, op.value());
  });

  it("parses an opcode with a result", () => {
    let input = "%void = OpTypeVoid";
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.lengthOf(ast.instructions(), 1);

    let inst = ast.instruction(0);
    assert.equal("OpTypeVoid", inst.name());
    assert.equal(19, inst.opcode());
    assert.lengthOf(inst.operands(), 1);

    let op = inst.operand(0);
    assert.equal("void", op.name());
    assert.equal(1, op.value());
  });

  it("sets module bounds based on numeric result", () => {
    let input = "%3 = OpTypeVoid";

    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.equal(4, ast.getId("next"));
  });

  it("returns the same value for a named result_id", () => {
    let input = "%3 = OpTypeFunction %int %int";

    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.lengthOf(ast.instructions(), 1);

    let inst = ast.instruction(0);
    let op1 = inst.operand(1);
    assert.equal("int", op1.name());
    assert.equal(4, op1.value());

    let op2 = inst.operand(2);
    assert.equal("int", op2.name());
    assert.equal(4, op2.value());
  });

  it("parses an opcode with a string", () => {
    let input = "OpEntryPoint Fragment %main \"main\"";

    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.lengthOf(ast.instructions(), 1);

    let inst = ast.instruction(0);
    let op = inst.operand(2);
    assert.equal("main", op.name());
    assert.equal("main", op.value());
  });

  describe("numerics", () => {
    describe("integers", () => {
      it("parses an opcode with an integer", () => {
        let input = "OpSource GLSL 440";

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 1);

        let inst = ast.instruction(0);
        let op0 = inst.operand(0);
        assert.equal("GLSL", op0.name());
        assert.equal("ValueEnum", op0.type());
        assert.equal(2, op0.value());

        let op1 = inst.operand(1);
        assert.equal("440", op1.name());
        assert.equal(440, op1.value());
      });

      it("parses an opcode with a hex integer", () => {
        let input = "OpSource GLSL 0x440";

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 1);

        let inst = ast.instruction(0);
        let op0 = inst.operand(0);
        assert.equal("GLSL", op0.name());
        assert.equal("ValueEnum", op0.type());
        assert.equal(2, op0.value());

        let op1 = inst.operand(1);
        assert.equal("1088", op1.name());
        assert.equal(0x440, op1.value());
      });

      it.skip("parses immediate integers", () => {
        // TODO(dsinclair): Support or skip?
      });
    });

    describe("floats", () => {
      it("parses floats", () => {
        let input = `%float = OpTypeFloat 32
                     %float1 = OpConstant %float 0.400000006`;

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast, p.error);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(1);
        let op2 = inst.operand(2);
        assert.equal(0.400000006, op2.value());
      });

      // TODO(dsinclair): Make hex encoded floats parse ...
      it.skip("parses hex floats", () => {
        let input = `%float = OpTypeFloat 32
                     %nfloat = OpConstant %float -0.4p+2
                     %pfloat = OpConstant %float 0.4p-2
                     %inf = OpConstant %float32 0x1p+128
                     %neginf = OpConstant %float32 -0x1p+128
                     %aNaN = OpConstant %float32 0x1.8p+128
                     %moreNaN = OpConstant %float32 -0x1.0002p+128`;

        let results = [-40.0, .004, 0x00000, 0x00000, 0x7fc00000, 0xff800100];
        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast, p.error);
        assert.lengthOf(ast.instructions(), 7);

        for (const idx in results) {
          let inst = ast.instruction(idx);
          let op2 = inst.operand(2);
          assert.equal(results[idx], op2.value());
        }
      });

      it("parses a float that looks like an int", () => {
        let input = `%float = OpTypeFloat 32
                     %float1 = OpConstant %float 1`;

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast, p.error);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(1);
        let op2 = inst.operand(2);
        assert.equal(1, op2.value());
        assert.equal("float", op2.type());
      });
    });
  });

  describe("enums", () => {
    it("parses enum values", () => {
      let input = `%1 = OpTypeFloat 32
  %30 = OpImageSampleExplicitLod %1 %20 %18 Grad|ConstOffset %22 %24 %29`;

      let vals = [{val: 1, name: "1"},
        {val: 30, name: "30"},
        {val: 20, name: "20"},
        {val: 18, name: "18"},
        {val: 12, name: "Grad|ConstOffset"}];

      let l = new Lexer(input);
      let p = new Parser(grammar, l);

      let ast = p.parse();
      assert.exists(ast, p.error);
      assert.lengthOf(ast.instructions(), 2);

      let inst = ast.instruction(1);
      for (let idx in vals) {
        let op = inst.operand(idx);
        assert.equal(vals[idx].name, op.name());
        assert.equal(vals[idx].val, op.value());
      }

      // BitEnum
      let params = inst.operand(4).params();
      assert.lengthOf(params, 3);
      assert.equal("22", params[0].name());
      assert.equal(22, params[0].value());
      assert.equal("24", params[1].name());
      assert.equal(24, params[1].value());
      assert.equal("29", params[2].name());
      assert.equal(29, params[2].value());
    });

    it("parses enumerants with parameters", () => {
      let input ="OpExecutionMode %main LocalSize 2 3 4";

      let l = new Lexer(input);
      let p = new Parser(grammar, l);

      let ast = p.parse();
      assert.exists(ast, p.error);
      assert.lengthOf(ast.instructions(), 1);

      let inst = ast.instruction(0);
      assert.equal("OpExecutionMode", inst.name());
      assert.lengthOf(inst.operands(), 2);
      assert.equal("main", inst.operand(0).name());
      assert.equal("LocalSize", inst.operand(1).name());

      let params = inst.operand(1).params();
      assert.lengthOf(params, 3);
      assert.equal("2", params[0].name());
      assert.equal("3", params[1].name());
      assert.equal("4", params[2].name());
    });
  });

  it("parses result into second operand if needed", () => {
    let input = `%int = OpTypeInt 32 1
                 %int_3 = OpConstant %int 3`;
    let l = new Lexer(input);
    let p = new Parser(grammar, l);

    let ast = p.parse();
    assert.exists(ast);
    assert.lengthOf(ast.instructions(), 2);

    let inst = ast.instruction(1);
    assert.equal("OpConstant", inst.name());
    assert.equal(43, inst.opcode());
    assert.lengthOf(inst.operands(), 3);

    let op0 = inst.operand(0);
    assert.equal("int", op0.name());
    assert.equal(1, op0.value());

    let op1 = inst.operand(1);
    assert.equal("int_3", op1.name());
    assert.equal(2, op1.value());

    let op2 = inst.operand(2);
    assert.equal("3", op2.name());
    assert.equal(3, op2.value());
  });

  describe("quantifiers", () => {
    describe("?", () => {
      it("skips if missing", () => {
        let input = `OpImageWrite %1 %2 %3
OpKill`;
        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(0);
        assert.equal("OpImageWrite", inst.name());
        assert.lengthOf(inst.operands(), 3);
      });

      it("skips if missing at EOF", () => {
        let input = "OpImageWrite %1 %2 %3";
        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 1);

        let inst = ast.instruction(0);
        assert.equal("OpImageWrite", inst.name());
        assert.lengthOf(inst.operands(), 3);
      });

      it("extracts if available", () => {
        let input = `OpImageWrite %1 %2 %3 ConstOffset %2
OpKill`;
        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(0);
        assert.equal("OpImageWrite", inst.name());
        assert.lengthOf(inst.operands(), 4);
        assert.equal("ConstOffset", inst.operand(3).name());
      });
    });

    describe("*", () => {
      it("skips if missing", () => {
        let input = `OpEntryPoint Fragment %main "main"
OpKill`;

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(0);
        assert.equal("OpEntryPoint", inst.name());
        assert.lengthOf(inst.operands(), 3);
        assert.equal("main", inst.operand(2).name());
      });

      it("extracts one if available", () => {
        let input = `OpEntryPoint Fragment %main "main" %2
OpKill`;

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(0);
        assert.equal("OpEntryPoint", inst.name());
        assert.lengthOf(inst.operands(), 4);
        assert.equal("2", inst.operand(3).name());
      });

      it("extracts multiple if available", () => {
        let input = `OpEntryPoint Fragment %main "main" %2 %3 %4 %5
OpKill`;

        let l = new Lexer(input);
        let p = new Parser(grammar, l);

        let ast = p.parse();
        assert.exists(ast);
        assert.lengthOf(ast.instructions(), 2);

        let inst = ast.instruction(0);
        assert.equal("OpEntryPoint", inst.name());
        assert.lengthOf(inst.operands(), 7);
        assert.equal("2", inst.operand(3).name());
        assert.equal("3", inst.operand(4).name());
        assert.equal("4", inst.operand(5).name());
        assert.equal("5", inst.operand(6).name());
      });
    });
  });

  describe("extended instructions", () => {
    it("errors on non-glsl extensions", () => {
      let input = "%1 = OpExtInstImport \"OpenCL.std.100\"";

      let l = new Lexer(input);
      let p = new Parser(grammar, l);

      assert.isUndefined(p.parse());
    });

    it("handles extended instructions", () => {
      let input = `%1 = OpExtInstImport "GLSL.std.450"
  %44 = OpExtInst %7 %1 Sqrt %43`;

      let l = new Lexer(input);
      let p = new Parser(grammar, l);

      let ast = p.parse();
      assert.exists(ast, p.error);
      assert.lengthOf(ast.instructions(), 2);

      let inst = ast.instruction(1);
      assert.lengthOf(inst.operands(), 5);
      assert.equal(31, inst.operand(3).value());
      assert.equal("Sqrt", inst.operand(3).name());
      assert.equal(43, inst.operand(4).value());
      assert.equal("43", inst.operand(4).name());
    });
  });

  it.skip("handles spec constant ops", () => {
    // let input = "%sum = OpSpecConstantOp %i32 IAdd %a %b";
  });
});
