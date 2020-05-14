// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using InlineTest = PassTest<::testing::Test>;

TEST_F(InlineTest, Simple) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     return bar.x + bar.y;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 140",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %color \"color\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
       "%void = OpTypeVoid",
         "%10 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%14 = OpTypeFunction %float %_ptr_Function_v4float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
"%_ptr_Function_float = OpTypePointer Function %float",
     "%uint_1 = OpConstant %uint 1",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %float None %14",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%26 = OpLabel",
         "%27 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%28 = OpLoad %float %27",
         "%29 = OpAccessChain %_ptr_Function_float %bar %uint_1",
         "%30 = OpLoad %float %29",
         "%31 = OpFAdd %float %28 %30",
               "OpReturnValue %31",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %10",
         "%21 = OpLabel",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
         "%22 = OpLoad %v4float %BaseColor",
               "OpStore %param %22",
         "%23 = OpFunctionCall %float %foo_vf4_ %param",
         "%24 = OpCompositeConstruct %v4float %23 %23 %23 %23",
               "OpStore %color %24",
         "%25 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %25",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %10",
         "%21 = OpLabel",
         "%32 = OpVariable %_ptr_Function_float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
         "%22 = OpLoad %v4float %BaseColor",
               "OpStore %param %22",
         "%34 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%35 = OpLoad %float %34",
         "%36 = OpAccessChain %_ptr_Function_float %param %uint_1",
         "%37 = OpLoad %float %36",
         "%38 = OpFAdd %float %35 %37",
               "OpStore %32 %38",
         "%23 = OpLoad %float %32",
         "%24 = OpCompositeConstruct %v4float %23 %23 %23 %23",
               "OpStore %color %24",
         "%25 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %25",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, Nested) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo2(float f, float f2)
  // {
  //     return f * f2;
  // }
  //
  // float foo(vec4 bar)
  // {
  //     return foo2(bar.x + bar.y, bar.z);
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 140",
               "OpName %main \"main\"",
               "OpName %foo2_f1_f1_ \"foo2(f1;f1;\"",
               "OpName %f \"f\"",
               "OpName %f2 \"f2\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %param \"param\"",
               "OpName %param_0 \"param\"",
               "OpName %color \"color\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param_1 \"param\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
       "%void = OpTypeVoid",
         "%15 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
"%_ptr_Function_float = OpTypePointer Function %float",
         "%18 = OpTypeFunction %float %_ptr_Function_float %_ptr_Function_float",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%21 = OpTypeFunction %float %_ptr_Function_v4float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
     "%uint_1 = OpConstant %uint 1",
     "%uint_2 = OpConstant %uint 2",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
"%foo2_f1_f1_ = OpFunction %float None %18",
          "%f = OpFunctionParameter %_ptr_Function_float",
         "%f2 = OpFunctionParameter %_ptr_Function_float",
         "%33 = OpLabel",
         "%34 = OpLoad %float %f",
         "%35 = OpLoad %float %f2",
         "%36 = OpFMul %float %34 %35",
               "OpReturnValue %36",
               "OpFunctionEnd",
   "%foo_vf4_ = OpFunction %float None %21",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%37 = OpLabel",
      "%param = OpVariable %_ptr_Function_float Function",
    "%param_0 = OpVariable %_ptr_Function_float Function",
         "%38 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%39 = OpLoad %float %38",
         "%40 = OpAccessChain %_ptr_Function_float %bar %uint_1",
         "%41 = OpLoad %float %40",
         "%42 = OpFAdd %float %39 %41",
               "OpStore %param %42",
         "%43 = OpAccessChain %_ptr_Function_float %bar %uint_2",
         "%44 = OpLoad %float %43",
               "OpStore %param_0 %44",
         "%45 = OpFunctionCall %float %foo2_f1_f1_ %param %param_0",
               "OpReturnValue %45",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %15",
         "%28 = OpLabel",
      "%color = OpVariable %_ptr_Function_v4float Function",
    "%param_1 = OpVariable %_ptr_Function_v4float Function",
         "%29 = OpLoad %v4float %BaseColor",
               "OpStore %param_1 %29",
         "%30 = OpFunctionCall %float %foo_vf4_ %param_1",
         "%31 = OpCompositeConstruct %v4float %30 %30 %30 %30",
               "OpStore %color %31",
         "%32 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %32",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %15",
         "%28 = OpLabel",
         "%58 = OpVariable %_ptr_Function_float Function",
         "%46 = OpVariable %_ptr_Function_float Function",
         "%47 = OpVariable %_ptr_Function_float Function",
         "%48 = OpVariable %_ptr_Function_float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
    "%param_1 = OpVariable %_ptr_Function_v4float Function",
         "%29 = OpLoad %v4float %BaseColor",
               "OpStore %param_1 %29",
         "%50 = OpAccessChain %_ptr_Function_float %param_1 %uint_0",
         "%51 = OpLoad %float %50",
         "%52 = OpAccessChain %_ptr_Function_float %param_1 %uint_1",
         "%53 = OpLoad %float %52",
         "%54 = OpFAdd %float %51 %53",
               "OpStore %46 %54",
         "%55 = OpAccessChain %_ptr_Function_float %param_1 %uint_2",
         "%56 = OpLoad %float %55",
               "OpStore %47 %56",
         "%60 = OpLoad %float %46",
         "%61 = OpLoad %float %47",
         "%62 = OpFMul %float %60 %61",
               "OpStore %58 %62",
         "%57 = OpLoad %float %58",
               "OpStore %48 %57",
         "%30 = OpLoad %float %48",
         "%31 = OpCompositeConstruct %v4float %30 %30 %30 %30",
               "OpStore %color %31",
         "%32 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %32",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, InOutParameter) {
  // #version 400
  //
  // in vec4 Basecolor;
  //
  // void foo(inout vec4 bar)
  // {
  //     bar.z = bar.x + bar.y;
  // }
  //
  // void main()
  // {
  //     vec4 b = Basecolor;
  //     foo(b);
  //     vec4 color = vec4(b.z);
  //     gl_FragColor = color;
  // }
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %Basecolor %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 400",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %b \"b\"",
               "OpName %Basecolor \"Basecolor\"",
               "OpName %param \"param\"",
               "OpName %color \"color\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
       "%void = OpTypeVoid",
         "%11 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%15 = OpTypeFunction %void %_ptr_Function_v4float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
"%_ptr_Function_float = OpTypePointer Function %float",
     "%uint_1 = OpConstant %uint 1",
     "%uint_2 = OpConstant %uint 2",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%Basecolor = OpVariable %_ptr_Input_v4float Input",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %void None %15",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%32 = OpLabel",
         "%33 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%34 = OpLoad %float %33",
         "%35 = OpAccessChain %_ptr_Function_float %bar %uint_1",
         "%36 = OpLoad %float %35",
         "%37 = OpFAdd %float %34 %36",
         "%38 = OpAccessChain %_ptr_Function_float %bar %uint_2",
               "OpStore %38 %37",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %11",
         "%23 = OpLabel",
          "%b = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
         "%24 = OpLoad %v4float %Basecolor",
               "OpStore %b %24",
         "%25 = OpLoad %v4float %b",
               "OpStore %param %25",
         "%26 = OpFunctionCall %void %foo_vf4_ %param",
         "%27 = OpLoad %v4float %param",
               "OpStore %b %27",
         "%28 = OpAccessChain %_ptr_Function_float %b %uint_2",
         "%29 = OpLoad %float %28",
         "%30 = OpCompositeConstruct %v4float %29 %29 %29 %29",
               "OpStore %color %30",
         "%31 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %31",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %11",
         "%23 = OpLabel",
          "%b = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
         "%24 = OpLoad %v4float %Basecolor",
               "OpStore %b %24",
         "%25 = OpLoad %v4float %b",
               "OpStore %param %25",
         "%40 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%41 = OpLoad %float %40",
         "%42 = OpAccessChain %_ptr_Function_float %param %uint_1",
         "%43 = OpLoad %float %42",
         "%44 = OpFAdd %float %41 %43",
         "%45 = OpAccessChain %_ptr_Function_float %param %uint_2",
               "OpStore %45 %44",
         "%27 = OpLoad %v4float %param",
               "OpStore %b %27",
         "%28 = OpAccessChain %_ptr_Function_float %b %uint_2",
         "%29 = OpLoad %float %28",
         "%30 = OpCompositeConstruct %v4float %29 %29 %29 %29",
               "OpStore %color %30",
         "%31 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %31",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, BranchInCallee) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     float r = bar.x;
  //     if (r < 0.0)
  //         r = -r;
  //     return r;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //
  //     gl_FragColor = color;
  // }
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 140",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %r \"r\"",
               "OpName %color \"color\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
       "%void = OpTypeVoid",
         "%11 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%15 = OpTypeFunction %float %_ptr_Function_v4float",
"%_ptr_Function_float = OpTypePointer Function %float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
    "%float_0 = OpConstant %float 0",
       "%bool = OpTypeBool",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %float None %15",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%28 = OpLabel",
          "%r = OpVariable %_ptr_Function_float Function",
         "%29 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%30 = OpLoad %float %29",
               "OpStore %r %30",
         "%31 = OpLoad %float %r",
         "%32 = OpFOrdLessThan %bool %31 %float_0",
               "OpSelectionMerge %33 None",
               "OpBranchConditional %32 %34 %33",
         "%34 = OpLabel",
         "%35 = OpLoad %float %r",
         "%36 = OpFNegate %float %35",
               "OpStore %r %36",
               "OpBranch %33",
         "%33 = OpLabel",
         "%37 = OpLoad %float %r",
               "OpReturnValue %37",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %11",
         "%23 = OpLabel",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
         "%24 = OpLoad %v4float %BaseColor",
               "OpStore %param %24",
         "%25 = OpFunctionCall %float %foo_vf4_ %param",
         "%26 = OpCompositeConstruct %v4float %25 %25 %25 %25",
               "OpStore %color %26",
         "%27 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %27",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %11",
         "%23 = OpLabel",
         "%38 = OpVariable %_ptr_Function_float Function",
         "%39 = OpVariable %_ptr_Function_float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
         "%24 = OpLoad %v4float %BaseColor",
               "OpStore %param %24",
         "%41 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%42 = OpLoad %float %41",
               "OpStore %38 %42",
         "%43 = OpLoad %float %38",
         "%44 = OpFOrdLessThan %bool %43 %float_0",
               "OpSelectionMerge %48 None",
               "OpBranchConditional %44 %45 %48",
         "%45 = OpLabel",
         "%46 = OpLoad %float %38",
         "%47 = OpFNegate %float %46",
               "OpStore %38 %47",
               "OpBranch %48",
         "%48 = OpLabel",
         "%49 = OpLoad %float %38",
               "OpStore %39 %49",
         "%25 = OpLoad %float %39",
         "%26 = OpCompositeConstruct %v4float %25 %25 %25 %25",
               "OpStore %color %26",
         "%27 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %27",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, PhiAfterCall) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(float bar)
  // {
  //     float r = bar;
  //     if (r < 0.0)
  //         r = -r;
  //     return r;
  // }
  //
  // void main()
  // {
  //     vec4 color = BaseColor;
  //     if (foo(color.x) > 2.0 && foo(color.y) > 2.0)
  //         color = vec4(0.0);
  //     gl_FragColor = color;
  // }
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %gl_FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 140",
               "OpName %main \"main\"",
               "OpName %foo_f1_ \"foo(f1;\"",
               "OpName %bar \"bar\"",
               "OpName %r \"r\"",
               "OpName %color \"color\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %param_0 \"param\"",
               "OpName %gl_FragColor \"gl_FragColor\"",
       "%void = OpTypeVoid",
         "%12 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
"%_ptr_Function_float = OpTypePointer Function %float",
         "%15 = OpTypeFunction %float %_ptr_Function_float",
    "%float_0 = OpConstant %float 0",
       "%bool = OpTypeBool",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
    "%float_2 = OpConstant %float 2",
     "%uint_1 = OpConstant %uint 1",
         "%25 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
"%gl_FragColor = OpVariable %_ptr_Output_v4float Output",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
    "%foo_f1_ = OpFunction %float None %15",
        "%bar = OpFunctionParameter %_ptr_Function_float",
         "%43 = OpLabel",
          "%r = OpVariable %_ptr_Function_float Function",
         "%44 = OpLoad %float %bar",
               "OpStore %r %44",
         "%45 = OpLoad %float %r",
         "%46 = OpFOrdLessThan %bool %45 %float_0",
               "OpSelectionMerge %47 None",
               "OpBranchConditional %46 %48 %47",
         "%48 = OpLabel",
         "%49 = OpLoad %float %r",
         "%50 = OpFNegate %float %49",
               "OpStore %r %50",
               "OpBranch %47",
         "%47 = OpLabel",
         "%51 = OpLoad %float %r",
               "OpReturnValue %51",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %12",
         "%27 = OpLabel",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_float Function",
    "%param_0 = OpVariable %_ptr_Function_float Function",
         "%28 = OpLoad %v4float %BaseColor",
               "OpStore %color %28",
         "%29 = OpAccessChain %_ptr_Function_float %color %uint_0",
         "%30 = OpLoad %float %29",
               "OpStore %param %30",
         "%31 = OpFunctionCall %float %foo_f1_ %param",
         "%32 = OpFOrdGreaterThan %bool %31 %float_2",
               "OpSelectionMerge %33 None",
               "OpBranchConditional %32 %34 %33",
         "%34 = OpLabel",
         "%35 = OpAccessChain %_ptr_Function_float %color %uint_1",
         "%36 = OpLoad %float %35",
               "OpStore %param_0 %36",
         "%37 = OpFunctionCall %float %foo_f1_ %param_0",
         "%38 = OpFOrdGreaterThan %bool %37 %float_2",
               "OpBranch %33",
         "%33 = OpLabel",
         "%39 = OpPhi %bool %32 %27 %38 %34",
               "OpSelectionMerge %40 None",
               "OpBranchConditional %39 %41 %40",
         "%41 = OpLabel",
               "OpStore %color %25",
               "OpBranch %40",
         "%40 = OpLabel",
         "%42 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %42",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %12",
         "%27 = OpLabel",
         "%63 = OpVariable %_ptr_Function_float Function",
         "%64 = OpVariable %_ptr_Function_float Function",
         "%52 = OpVariable %_ptr_Function_float Function",
         "%53 = OpVariable %_ptr_Function_float Function",
      "%color = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_float Function",
    "%param_0 = OpVariable %_ptr_Function_float Function",
         "%28 = OpLoad %v4float %BaseColor",
               "OpStore %color %28",
         "%29 = OpAccessChain %_ptr_Function_float %color %uint_0",
         "%30 = OpLoad %float %29",
               "OpStore %param %30",
         "%55 = OpLoad %float %param",
               "OpStore %52 %55",
         "%56 = OpLoad %float %52",
         "%57 = OpFOrdLessThan %bool %56 %float_0",
               "OpSelectionMerge %61 None",
               "OpBranchConditional %57 %58 %61",
         "%58 = OpLabel",
         "%59 = OpLoad %float %52",
         "%60 = OpFNegate %float %59",
               "OpStore %52 %60",
               "OpBranch %61",
         "%61 = OpLabel",
         "%62 = OpLoad %float %52",
               "OpStore %53 %62",
         "%31 = OpLoad %float %53",
         "%32 = OpFOrdGreaterThan %bool %31 %float_2",
               "OpSelectionMerge %33 None",
               "OpBranchConditional %32 %34 %33",
         "%34 = OpLabel",
         "%35 = OpAccessChain %_ptr_Function_float %color %uint_1",
         "%36 = OpLoad %float %35",
               "OpStore %param_0 %36",
         "%66 = OpLoad %float %param_0",
               "OpStore %63 %66",
         "%67 = OpLoad %float %63",
         "%68 = OpFOrdLessThan %bool %67 %float_0",
               "OpSelectionMerge %72 None",
               "OpBranchConditional %68 %69 %72",
         "%69 = OpLabel",
         "%70 = OpLoad %float %63",
         "%71 = OpFNegate %float %70",
               "OpStore %63 %71",
               "OpBranch %72",
         "%72 = OpLabel",
         "%73 = OpLoad %float %63",
               "OpStore %64 %73",
         "%37 = OpLoad %float %64",
         "%38 = OpFOrdGreaterThan %bool %37 %float_2",
               "OpBranch %33",
         "%33 = OpLabel",
         "%39 = OpPhi %bool %32 %61 %38 %72",
               "OpSelectionMerge %40 None",
               "OpBranchConditional %39 %41 %40",
         "%41 = OpLabel",
               "OpStore %color %25",
               "OpBranch %40",
         "%40 = OpLabel",
         "%42 = OpLoad %v4float %color",
               "OpStore %gl_FragColor %42",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, OpSampledImageOutOfBlock) {
  // #version 450
  //
  // uniform texture2D t2D;
  // uniform sampler samp;
  // out vec4 FragColor;
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     float r = bar.x;
  //     if (r < 0.0)
  //         r = -r;
  //     return r;
  // }
  //
  // void main()
  // {
  //     vec4 color1 = texture(sampler2D(t2D, samp), vec2(1.0));
  //     vec4 color2 = vec4(foo(BaseColor));
  //     vec4 color3 = texture(sampler2D(t2D, samp), vec2(0.5));
  //     FragColor = (color1 + color2 + color3)/3;
  // }
  //
  // Note: the before SPIR-V will need to be edited to create a use of
  // the OpSampledImage across the function call.
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 450",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %r \"r\"",
               "OpName %color1 \"color1\"",
               "OpName %t2D \"t2D\"",
               "OpName %samp \"samp\"",
               "OpName %color2 \"color2\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %color3 \"color3\"",
               "OpName %FragColor \"FragColor\"",
               "OpDecorate %t2D DescriptorSet 0",
               "OpDecorate %samp DescriptorSet 0",
       "%void = OpTypeVoid",
         "%15 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%19 = OpTypeFunction %float %_ptr_Function_v4float",
"%_ptr_Function_float = OpTypePointer Function %float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
    "%float_0 = OpConstant %float 0",
       "%bool = OpTypeBool",
         "%25 = OpTypeImage %float 2D 0 0 0 1 Unknown",
"%_ptr_UniformConstant_25 = OpTypePointer UniformConstant %25",
        "%t2D = OpVariable %_ptr_UniformConstant_25 UniformConstant",
         "%27 = OpTypeSampler",
"%_ptr_UniformConstant_27 = OpTypePointer UniformConstant %27",
       "%samp = OpVariable %_ptr_UniformConstant_27 UniformConstant",
         "%29 = OpTypeSampledImage %25",
    "%v2float = OpTypeVector %float 2",
    "%float_1 = OpConstant %float 1",
         "%32 = OpConstantComposite %v2float %float_1 %float_1",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
  "%float_0_5 = OpConstant %float 0.5",
         "%35 = OpConstantComposite %v2float %float_0_5 %float_0_5",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
  "%FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_3 = OpConstant %float 3",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %float None %19",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%56 = OpLabel",
          "%r = OpVariable %_ptr_Function_float Function",
         "%57 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%58 = OpLoad %float %57",
               "OpStore %r %58",
         "%59 = OpLoad %float %r",
         "%60 = OpFOrdLessThan %bool %59 %float_0",
               "OpSelectionMerge %61 None",
               "OpBranchConditional %60 %62 %61",
         "%62 = OpLabel",
         "%63 = OpLoad %float %r",
         "%64 = OpFNegate %float %63",
               "OpStore %r %64",
               "OpBranch %61",
         "%61 = OpLabel",
         "%65 = OpLoad %float %r",
               "OpReturnValue %65",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %15",
         "%38 = OpLabel",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%39 = OpLoad %25 %t2D",
         "%40 = OpLoad %27 %samp",
         "%41 = OpSampledImage %29 %39 %40",
         "%42 = OpImageSampleImplicitLod %v4float %41 %32",
               "OpStore %color1 %42",
         "%43 = OpLoad %v4float %BaseColor",
               "OpStore %param %43",
         "%44 = OpFunctionCall %float %foo_vf4_ %param",
         "%45 = OpCompositeConstruct %v4float %44 %44 %44 %44",
               "OpStore %color2 %45",
         "%46 = OpLoad %25 %t2D",
         "%47 = OpLoad %27 %samp",
         "%48 = OpImageSampleImplicitLod %v4float %41 %35",
               "OpStore %color3 %48",
         "%49 = OpLoad %v4float %color1",
         "%50 = OpLoad %v4float %color2",
         "%51 = OpFAdd %v4float %49 %50",
         "%52 = OpLoad %v4float %color3",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%55 = OpFDiv %v4float %53 %54",
               "OpStore %FragColor %55",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %15",
         "%38 = OpLabel",
         "%66 = OpVariable %_ptr_Function_float Function",
         "%67 = OpVariable %_ptr_Function_float Function",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%39 = OpLoad %25 %t2D",
         "%40 = OpLoad %27 %samp",
         "%41 = OpSampledImage %29 %39 %40",
         "%42 = OpImageSampleImplicitLod %v4float %41 %32",
               "OpStore %color1 %42",
         "%43 = OpLoad %v4float %BaseColor",
               "OpStore %param %43",
         "%69 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%70 = OpLoad %float %69",
               "OpStore %66 %70",
         "%71 = OpLoad %float %66",
         "%72 = OpFOrdLessThan %bool %71 %float_0",
               "OpSelectionMerge %76 None",
               "OpBranchConditional %72 %73 %76",
         "%73 = OpLabel",
         "%74 = OpLoad %float %66",
         "%75 = OpFNegate %float %74",
               "OpStore %66 %75",
               "OpBranch %76",
         "%76 = OpLabel",
         "%77 = OpLoad %float %66",
               "OpStore %67 %77",
         "%44 = OpLoad %float %67",
         "%45 = OpCompositeConstruct %v4float %44 %44 %44 %44",
               "OpStore %color2 %45",
         "%46 = OpLoad %25 %t2D",
         "%47 = OpLoad %27 %samp",
         "%78 = OpSampledImage %29 %39 %40",
         "%48 = OpImageSampleImplicitLod %v4float %78 %35",
               "OpStore %color3 %48",
         "%49 = OpLoad %v4float %color1",
         "%50 = OpLoad %v4float %color2",
         "%51 = OpFAdd %v4float %49 %50",
         "%52 = OpLoad %v4float %color3",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%55 = OpFDiv %v4float %53 %54",
               "OpStore %FragColor %55",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, OpImageOutOfBlock) {
  // #version 450
  //
  // uniform texture2D t2D;
  // uniform sampler samp;
  // uniform sampler samp2;
  //
  // out vec4 FragColor;
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     float r = bar.x;
  //     if (r < 0.0)
  //         r = -r;
  //     return r;
  // }
  //
  // void main()
  // {
  //     vec4 color1 = texture(sampler2D(t2D, samp), vec2(1.0));
  //     vec4 color2 = vec4(foo(BaseColor));
  //     vec4 color3 = texture(sampler2D(t2D, samp2), vec2(0.5));
  //     FragColor = (color1 + color2 + color3)/3;
  // }
  // Note: the before SPIR-V will need to be edited to create an OpImage
  // from the first OpSampledImage, place it before the call and use it
  // in the second OpSampledImage following the call.
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 450",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %r \"r\"",
               "OpName %color1 \"color1\"",
               "OpName %t2D \"t2D\"",
               "OpName %samp \"samp\"",
               "OpName %color2 \"color2\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %color3 \"color3\"",
               "OpName %samp2 \"samp2\"",
               "OpName %FragColor \"FragColor\"",
               "OpDecorate %t2D DescriptorSet 0",
               "OpDecorate %samp DescriptorSet 0",
               "OpDecorate %samp2 DescriptorSet 0",
       "%void = OpTypeVoid",
         "%16 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%20 = OpTypeFunction %float %_ptr_Function_v4float",
"%_ptr_Function_float = OpTypePointer Function %float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
    "%float_0 = OpConstant %float 0",
       "%bool = OpTypeBool",
         "%26 = OpTypeImage %float 2D 0 0 0 1 Unknown",
"%_ptr_UniformConstant_26 = OpTypePointer UniformConstant %26",
        "%t2D = OpVariable %_ptr_UniformConstant_26 UniformConstant",
         "%28 = OpTypeSampler",
"%_ptr_UniformConstant_28 = OpTypePointer UniformConstant %28",
       "%samp = OpVariable %_ptr_UniformConstant_28 UniformConstant",
         "%30 = OpTypeSampledImage %26",
    "%v2float = OpTypeVector %float 2",
    "%float_1 = OpConstant %float 1",
         "%33 = OpConstantComposite %v2float %float_1 %float_1",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
      "%samp2 = OpVariable %_ptr_UniformConstant_28 UniformConstant",
  "%float_0_5 = OpConstant %float 0.5",
         "%36 = OpConstantComposite %v2float %float_0_5 %float_0_5",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
  "%FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_3 = OpConstant %float 3",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %float None %20",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%58 = OpLabel",
          "%r = OpVariable %_ptr_Function_float Function",
         "%59 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%60 = OpLoad %float %59",
               "OpStore %r %60",
         "%61 = OpLoad %float %r",
         "%62 = OpFOrdLessThan %bool %61 %float_0",
               "OpSelectionMerge %63 None",
               "OpBranchConditional %62 %64 %63",
         "%64 = OpLabel",
         "%65 = OpLoad %float %r",
         "%66 = OpFNegate %float %65",
               "OpStore %r %66",
               "OpBranch %63",
         "%63 = OpLabel",
         "%67 = OpLoad %float %r",
               "OpReturnValue %67",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %16",
         "%39 = OpLabel",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%40 = OpLoad %26 %t2D",
         "%41 = OpLoad %28 %samp",
         "%42 = OpSampledImage %30 %40 %41",
         "%43 = OpImageSampleImplicitLod %v4float %42 %33",
         "%44 = OpImage %26 %42",
         "%45 = OpLoad %28 %samp2",
               "OpStore %color1 %43",
         "%46 = OpLoad %v4float %BaseColor",
               "OpStore %param %46",
         "%47 = OpFunctionCall %float %foo_vf4_ %param",
         "%48 = OpCompositeConstruct %v4float %47 %47 %47 %47",
               "OpStore %color2 %48",
         "%49 = OpSampledImage %30 %44 %45",
         "%50 = OpImageSampleImplicitLod %v4float %49 %36",
               "OpStore %color3 %50",
         "%51 = OpLoad %v4float %color1",
         "%52 = OpLoad %v4float %color2",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpLoad %v4float %color3",
         "%55 = OpFAdd %v4float %53 %54",
         "%56 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%57 = OpFDiv %v4float %55 %56",
               "OpStore %FragColor %57",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %16",
         "%39 = OpLabel",
         "%68 = OpVariable %_ptr_Function_float Function",
         "%69 = OpVariable %_ptr_Function_float Function",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%40 = OpLoad %26 %t2D",
         "%41 = OpLoad %28 %samp",
         "%42 = OpSampledImage %30 %40 %41",
         "%43 = OpImageSampleImplicitLod %v4float %42 %33",
         "%44 = OpImage %26 %42",
         "%45 = OpLoad %28 %samp2",
               "OpStore %color1 %43",
         "%46 = OpLoad %v4float %BaseColor",
               "OpStore %param %46",
         "%71 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%72 = OpLoad %float %71",
               "OpStore %68 %72",
         "%73 = OpLoad %float %68",
         "%74 = OpFOrdLessThan %bool %73 %float_0",
               "OpSelectionMerge %78 None",
               "OpBranchConditional %74 %75 %78",
         "%75 = OpLabel",
         "%76 = OpLoad %float %68",
         "%77 = OpFNegate %float %76",
               "OpStore %68 %77",
               "OpBranch %78",
         "%78 = OpLabel",
         "%79 = OpLoad %float %68",
               "OpStore %69 %79",
         "%47 = OpLoad %float %69",
         "%48 = OpCompositeConstruct %v4float %47 %47 %47 %47",
               "OpStore %color2 %48",
         "%80 = OpSampledImage %30 %40 %41",
         "%81 = OpImage %26 %80",
         "%49 = OpSampledImage %30 %81 %45",
         "%50 = OpImageSampleImplicitLod %v4float %49 %36",
               "OpStore %color3 %50",
         "%51 = OpLoad %v4float %color1",
         "%52 = OpLoad %v4float %color2",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpLoad %v4float %color3",
         "%55 = OpFAdd %v4float %53 %54",
         "%56 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%57 = OpFDiv %v4float %55 %56",
               "OpStore %FragColor %57",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, OpImageAndOpSampledImageOutOfBlock) {
  // #version 450
  //
  // uniform texture2D t2D;
  // uniform sampler samp;
  // uniform sampler samp2;
  //
  // out vec4 FragColor;
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     float r = bar.x;
  //     if (r < 0.0)
  //         r = -r;
  //     return r;
  // }
  //
  // void main()
  // {
  //     vec4 color1 = texture(sampler2D(t2D, samp), vec2(1.0));
  //     vec4 color2 = vec4(foo(BaseColor));
  //     vec4 color3 = texture(sampler2D(t2D, samp2), vec2(0.5));
  //     FragColor = (color1 + color2 + color3)/3;
  // }
  // Note: the before SPIR-V will need to be edited to create an OpImage
  // and subsequent OpSampledImage that is used across the function call.
  const std::vector<const char*> predefs = {
      // clang-format off
               "OpCapability Shader",
          "%1 = OpExtInstImport \"GLSL.std.450\"",
               "OpMemoryModel Logical GLSL450",
               "OpEntryPoint Fragment %main \"main\" %BaseColor %FragColor",
               "OpExecutionMode %main OriginUpperLeft",
               "OpSource GLSL 450",
               "OpName %main \"main\"",
               "OpName %foo_vf4_ \"foo(vf4;\"",
               "OpName %bar \"bar\"",
               "OpName %r \"r\"",
               "OpName %color1 \"color1\"",
               "OpName %t2D \"t2D\"",
               "OpName %samp \"samp\"",
               "OpName %color2 \"color2\"",
               "OpName %BaseColor \"BaseColor\"",
               "OpName %param \"param\"",
               "OpName %color3 \"color3\"",
               "OpName %samp2 \"samp2\"",
               "OpName %FragColor \"FragColor\"",
               "OpDecorate %t2D DescriptorSet 0",
               "OpDecorate %samp DescriptorSet 0",
               "OpDecorate %samp2 DescriptorSet 0",
       "%void = OpTypeVoid",
         "%16 = OpTypeFunction %void",
      "%float = OpTypeFloat 32",
    "%v4float = OpTypeVector %float 4",
"%_ptr_Function_v4float = OpTypePointer Function %v4float",
         "%20 = OpTypeFunction %float %_ptr_Function_v4float",
"%_ptr_Function_float = OpTypePointer Function %float",
       "%uint = OpTypeInt 32 0",
     "%uint_0 = OpConstant %uint 0",
    "%float_0 = OpConstant %float 0",
       "%bool = OpTypeBool",
         "%26 = OpTypeImage %float 2D 0 0 0 1 Unknown",
"%_ptr_UniformConstant_26 = OpTypePointer UniformConstant %26",
        "%t2D = OpVariable %_ptr_UniformConstant_26 UniformConstant",
         "%28 = OpTypeSampler",
"%_ptr_UniformConstant_28 = OpTypePointer UniformConstant %28",
       "%samp = OpVariable %_ptr_UniformConstant_28 UniformConstant",
         "%30 = OpTypeSampledImage %26",
    "%v2float = OpTypeVector %float 2",
    "%float_1 = OpConstant %float 1",
         "%33 = OpConstantComposite %v2float %float_1 %float_1",
"%_ptr_Input_v4float = OpTypePointer Input %v4float",
  "%BaseColor = OpVariable %_ptr_Input_v4float Input",
      "%samp2 = OpVariable %_ptr_UniformConstant_28 UniformConstant",
  "%float_0_5 = OpConstant %float 0.5",
         "%36 = OpConstantComposite %v2float %float_0_5 %float_0_5",
"%_ptr_Output_v4float = OpTypePointer Output %v4float",
  "%FragColor = OpVariable %_ptr_Output_v4float Output",
    "%float_3 = OpConstant %float 3",
      // clang-format on
  };

  const std::vector<const char*> nonEntryFuncs = {
      // clang-format off
   "%foo_vf4_ = OpFunction %float None %20",
        "%bar = OpFunctionParameter %_ptr_Function_v4float",
         "%58 = OpLabel",
          "%r = OpVariable %_ptr_Function_float Function",
         "%59 = OpAccessChain %_ptr_Function_float %bar %uint_0",
         "%60 = OpLoad %float %59",
               "OpStore %r %60",
         "%61 = OpLoad %float %r",
         "%62 = OpFOrdLessThan %bool %61 %float_0",
               "OpSelectionMerge %63 None",
               "OpBranchConditional %62 %64 %63",
         "%64 = OpLabel",
         "%65 = OpLoad %float %r",
         "%66 = OpFNegate %float %65",
               "OpStore %r %66",
               "OpBranch %63",
         "%63 = OpLabel",
         "%67 = OpLoad %float %r",
               "OpReturnValue %67",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> before = {
      // clang-format off
       "%main = OpFunction %void None %16",
         "%39 = OpLabel",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%40 = OpLoad %26 %t2D",
         "%41 = OpLoad %28 %samp",
         "%42 = OpSampledImage %30 %40 %41",
         "%43 = OpImageSampleImplicitLod %v4float %42 %33",
         "%44 = OpImage %26 %42",
         "%45 = OpLoad %28 %samp2",
         "%46 = OpSampledImage %30 %44 %45",
               "OpStore %color1 %43",
         "%47 = OpLoad %v4float %BaseColor",
               "OpStore %param %47",
         "%48 = OpFunctionCall %float %foo_vf4_ %param",
         "%49 = OpCompositeConstruct %v4float %48 %48 %48 %48",
               "OpStore %color2 %49",
         "%50 = OpImageSampleImplicitLod %v4float %46 %36",
               "OpStore %color3 %50",
         "%51 = OpLoad %v4float %color1",
         "%52 = OpLoad %v4float %color2",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpLoad %v4float %color3",
         "%55 = OpFAdd %v4float %53 %54",
         "%56 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%57 = OpFDiv %v4float %55 %56",
               "OpStore %FragColor %57",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };

  const std::vector<const char*> after = {
      // clang-format off
       "%main = OpFunction %void None %16",
         "%39 = OpLabel",
         "%68 = OpVariable %_ptr_Function_float Function",
         "%69 = OpVariable %_ptr_Function_float Function",
     "%color1 = OpVariable %_ptr_Function_v4float Function",
     "%color2 = OpVariable %_ptr_Function_v4float Function",
      "%param = OpVariable %_ptr_Function_v4float Function",
     "%color3 = OpVariable %_ptr_Function_v4float Function",
         "%40 = OpLoad %26 %t2D",
         "%41 = OpLoad %28 %samp",
         "%42 = OpSampledImage %30 %40 %41",
         "%43 = OpImageSampleImplicitLod %v4float %42 %33",
         "%44 = OpImage %26 %42",
         "%45 = OpLoad %28 %samp2",
         "%46 = OpSampledImage %30 %44 %45",
               "OpStore %color1 %43",
         "%47 = OpLoad %v4float %BaseColor",
               "OpStore %param %47",
         "%71 = OpAccessChain %_ptr_Function_float %param %uint_0",
         "%72 = OpLoad %float %71",
               "OpStore %68 %72",
         "%73 = OpLoad %float %68",
         "%74 = OpFOrdLessThan %bool %73 %float_0",
               "OpSelectionMerge %78 None",
               "OpBranchConditional %74 %75 %78",
         "%75 = OpLabel",
         "%76 = OpLoad %float %68",
         "%77 = OpFNegate %float %76",
               "OpStore %68 %77",
               "OpBranch %78",
         "%78 = OpLabel",
         "%79 = OpLoad %float %68",
               "OpStore %69 %79",
         "%48 = OpLoad %float %69",
         "%49 = OpCompositeConstruct %v4float %48 %48 %48 %48",
               "OpStore %color2 %49",
         "%80 = OpSampledImage %30 %40 %41",
         "%81 = OpImage %26 %80",
         "%82 = OpSampledImage %30 %81 %45",
         "%50 = OpImageSampleImplicitLod %v4float %82 %36",
               "OpStore %color3 %50",
         "%51 = OpLoad %v4float %color1",
         "%52 = OpLoad %v4float %color2",
         "%53 = OpFAdd %v4float %51 %52",
         "%54 = OpLoad %v4float %color3",
         "%55 = OpFAdd %v4float %53 %54",
         "%56 = OpCompositeConstruct %v4float %float_3 %float_3 %float_3 %float_3",
         "%57 = OpFDiv %v4float %55 %56",
               "OpStore %FragColor %57",
               "OpReturn",
               "OpFunctionEnd",
      // clang-format on
  };
  SinglePassRunAndCheck<InlineExhaustivePass>(
      JoinAllInsts(Concat(Concat(predefs, before), nonEntryFuncs)),
      JoinAllInsts(Concat(Concat(predefs, after), nonEntryFuncs)),
      /* skip_nop = */ false, /* do_validate = */ true);
}

TEST_F(InlineTest, EarlyReturnInLoopIsNotInlined) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     while (true) {
  //         if (bar.x < 0.0)
  //             return 0.0;
  //         return bar.x;
  //     }
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }

  const std::string assembly =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%23 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%24 = OpLoad %v4float %BaseColor
OpStore %param %24
%25 = OpFunctionCall %float %foo_vf4_ %param
%26 = OpCompositeConstruct %v4float %25 %25 %25 %25
OpStore %color %26
%27 = OpLoad %v4float %color
OpStore %gl_FragColor %27
OpReturn
OpFunctionEnd
%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%28 = OpLabel
OpBranch %29
%29 = OpLabel
OpLoopMerge %30 %31 None
OpBranch %32
%32 = OpLabel
OpBranchConditional %true %33 %30
%33 = OpLabel
%34 = OpAccessChain %_ptr_Function_float %bar %uint_0
%35 = OpLoad %float %34
%36 = OpFOrdLessThan %bool %35 %float_0
OpSelectionMerge %37 None
OpBranchConditional %36 %38 %37
%38 = OpLabel
OpReturnValue %float_0
%37 = OpLabel
%39 = OpAccessChain %_ptr_Function_float %bar %uint_0
%40 = OpLoad %float %39
OpReturnValue %40
%31 = OpLabel
OpBranch %29
%30 = OpLabel
%41 = OpUndef %float
OpReturnValue %41
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(assembly, assembly, false, true);
}

TEST_F(InlineTest, ExternalFunctionIsNotInlined) {
  // In particular, don't crash.
  // See report https://github.com/KhronosGroup/SPIRV-Tools/issues/605
  const std::string assembly =
      R"(OpCapability Addresses
OpCapability Kernel
OpCapability Linkage
OpMemoryModel Physical32 OpenCL
OpEntryPoint Kernel %1 "entry_pt"
OpDecorate %2 LinkageAttributes "external" Import
%void = OpTypeVoid
%4 = OpTypeFunction %void
%2 = OpFunction %void None %4
OpFunctionEnd
%1 = OpFunction %void None %4
%5 = OpLabel
%6 = OpFunctionCall %void %2
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(assembly, assembly, false, true);
}

TEST_F(InlineTest, SingleBlockLoopCallsMultiBlockCallee) {
  // Example from https://github.com/KhronosGroup/SPIRV-Tools/issues/787
  //
  // CFG structure is:
  //    foo:
  //       fooentry -> fooexit
  //
  //    main:
  //       entry -> loop
  //       loop -> loop, merge
  //         loop calls foo()
  //       merge
  //
  // Since the callee has multiple blocks, it will split the calling block
  // into at least two, resulting in a new "back-half" block that contains
  // the instructions after the inlined function call.  If the calling block
  // has an OpLoopMerge that points back to the calling block itself, then
  // the OpLoopMerge can't remain in the back-half block, but must be
  // moved to the end of the original calling block, and it continue target
  // operand updated to point to the back-half block.

  const std::string predefs =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpSource OpenCL_C 120
%bool = OpTypeBool
%true = OpConstantTrue %bool
%void = OpTypeVoid
)";

  const std::string nonEntryFuncs =
      R"(%5 = OpTypeFunction %void
%6 = OpFunction %void None %5
%7 = OpLabel
OpBranch %8
%8 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%1 = OpFunction %void None %5
%9 = OpLabel
OpBranch %10
%10 = OpLabel
%11 = OpFunctionCall %void %6
OpLoopMerge %12 %10 None
OpBranchConditional %true %10 %12
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%1 = OpFunction %void None %5
%9 = OpLabel
OpBranch %10
%10 = OpLabel
OpLoopMerge %12 %10 None
OpBranch %14
%14 = OpLabel
OpBranchConditional %true %10 %12
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + nonEntryFuncs + before,
                                              predefs + nonEntryFuncs + after,
                                              false, true);
}

TEST_F(InlineTest, MultiBlockLoopHeaderCallsMultiBlockCallee) {
  // Like SingleBlockLoopCallsMultiBlockCallee but the loop has several
  // blocks, but the function call still occurs in the loop header.
  // Example from https://github.com/KhronosGroup/SPIRV-Tools/issues/800

  const std::string predefs =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpSource OpenCL_C 120
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%int_4 = OpConstant %int 4
%int_5 = OpConstant %int 5
%void = OpTypeVoid
%11 = OpTypeFunction %void
)";

  const std::string nonEntryFuncs =
      R"(%12 = OpFunction %void None %11
%13 = OpLabel
%14 = OpCopyObject %int %int_1
OpBranch %15
%15 = OpLabel
%16 = OpCopyObject %int %int_2
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%1 = OpFunction %void None %11
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpCopyObject %int %int_3
%20 = OpFunctionCall %void %12
%21 = OpCopyObject %int %int_4
OpLoopMerge %22 %23 None
OpBranchConditional %true %23 %22
%23 = OpLabel
%24 = OpCopyObject %int %int_5
OpBranchConditional %true %18 %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%1 = OpFunction %void None %11
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpCopyObject %int %int_3
%26 = OpCopyObject %int %int_1
OpLoopMerge %22 %23 None
OpBranch %27
%27 = OpLabel
%28 = OpCopyObject %int %int_2
%21 = OpCopyObject %int %int_4
OpBranchConditional %true %23 %22
%23 = OpLabel
%24 = OpCopyObject %int %int_5
OpBranchConditional %true %18 %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + nonEntryFuncs + before,
                                              predefs + nonEntryFuncs + after,
                                              false, true);
}

TEST_F(InlineTest, SingleBlockLoopCallsMultiBlockCalleeHavingSelectionMerge) {
  // This is similar to SingleBlockLoopCallsMultiBlockCallee except
  // that calleee block also has a merge instruction in its first block.
  // That merge instruction must be an OpSelectionMerge (because the entry
  // block of a function can't be the header of a loop since the entry
  // block can't be the target of a branch).
  //
  // In this case the OpLoopMerge can't be placed in the same block as
  // the OpSelectionMerge, so inlining must create a new block to contain
  // the callee contents.
  //
  // Additionally, we have two dummy OpCopyObject instructions to prove that
  // the OpLoopMerge is moved to the right location.
  //
  // Also ensure that OpPhis within the cloned callee code are valid.
  // We need to test that the predecessor blocks are remapped correctly so that
  // dominance rules are satisfied

  const std::string predefs =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpSource OpenCL_C 120
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse %bool
%void = OpTypeVoid
%6 = OpTypeFunction %void
)";

  // This callee has multiple blocks, and an OpPhi in the last block
  // that references a value from the first block.  This tests that
  // cloned block IDs are remapped appropriately.  The OpPhi dominance
  // requires that the remapped %9 must be in a block that dominates
  // the remapped %8.
  const std::string nonEntryFuncs =
      R"(%7 = OpFunction %void None %6
%8 = OpLabel
%9 = OpCopyObject %bool %true
OpSelectionMerge %10 None
OpBranchConditional %true %10 %10
%10 = OpLabel
%11 = OpPhi %bool %9 %8
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%1 = OpFunction %void None %6
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%14 = OpCopyObject %bool %false
%15 = OpFunctionCall %void %7
OpLoopMerge %16 %13 None
OpBranchConditional %true %13 %16
%16 = OpLabel
OpReturn
OpFunctionEnd
)";

  // Note the remapped Phi uses %17 as the parent instead
  // of %13, demonstrating that the parent block has been remapped
  // correctly.
  const std::string after =
      R"(%1 = OpFunction %void None %6
%12 = OpLabel
OpBranch %13
%13 = OpLabel
%14 = OpCopyObject %bool %false
OpLoopMerge %16 %13 None
OpBranch %17
%17 = OpLabel
%19 = OpCopyObject %bool %true
OpSelectionMerge %20 None
OpBranchConditional %true %20 %20
%20 = OpLabel
%21 = OpPhi %bool %19 %17
OpBranchConditional %true %13 %16
%16 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + nonEntryFuncs + before,
                                              predefs + nonEntryFuncs + after,
                                              false, true);
}

TEST_F(InlineTest,
       MultiBlockLoopHeaderCallsFromToMultiBlockCalleeHavingSelectionMerge) {
  // This is similar to SingleBlockLoopCallsMultiBlockCalleeHavingSelectionMerge
  // but the call is in the header block of a multi block loop.

  const std::string predefs =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main"
OpSource OpenCL_C 120
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%int_2 = OpConstant %int 2
%int_3 = OpConstant %int 3
%int_4 = OpConstant %int 4
%int_5 = OpConstant %int 5
%void = OpTypeVoid
%11 = OpTypeFunction %void
)";

  const std::string nonEntryFuncs =
      R"(%12 = OpFunction %void None %11
%13 = OpLabel
%14 = OpCopyObject %int %int_1
OpSelectionMerge %15 None
OpBranchConditional %true %15 %15
%15 = OpLabel
%16 = OpCopyObject %int %int_2
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%1 = OpFunction %void None %11
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpCopyObject %int %int_3
%20 = OpFunctionCall %void %12
%21 = OpCopyObject %int %int_4
OpLoopMerge %22 %23 None
OpBranchConditional %true %23 %22
%23 = OpLabel
%24 = OpCopyObject %int %int_5
OpBranchConditional %true %18 %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%1 = OpFunction %void None %11
%17 = OpLabel
OpBranch %18
%18 = OpLabel
%19 = OpCopyObject %int %int_3
OpLoopMerge %22 %23 None
OpBranch %25
%25 = OpLabel
%27 = OpCopyObject %int %int_1
OpSelectionMerge %28 None
OpBranchConditional %true %28 %28
%28 = OpLabel
%29 = OpCopyObject %int %int_2
%21 = OpCopyObject %int %int_4
OpBranchConditional %true %23 %22
%23 = OpLabel
%24 = OpCopyObject %int %int_5
OpBranchConditional %true %18 %22
%22 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + nonEntryFuncs + before,
                                              predefs + nonEntryFuncs + after,
                                              false, true);
}

TEST_F(InlineTest, NonInlinableCalleeWithSingleReturn) {
  // The case from https://github.com/KhronosGroup/SPIRV-Tools/issues/2018
  //
  // The callee has a single return, but cannot be inlined because the
  // return is inside a loop.

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %_GLF_color
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %f_ "f("
OpName %i "i"
OpName %_GLF_color "_GLF_color"
OpDecorate %_GLF_color Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%9 = OpTypeFunction %float
%float_1 = OpConstant %float 1
%bool = OpTypeBool
%false = OpConstantFalse %bool
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_GLF_color = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%21 = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
)";

  const std::string caller =
      R"(%main = OpFunction %void None %7
%22 = OpLabel
%i = OpVariable %_ptr_Function_int Function
OpStore %i %int_0
OpBranch %23
%23 = OpLabel
OpLoopMerge %24 %25 None
OpBranch %26
%26 = OpLabel
%27 = OpLoad %int %i
%28 = OpSLessThan %bool %27 %int_1
OpBranchConditional %28 %29 %24
%29 = OpLabel
OpStore %_GLF_color %20
%30 = OpFunctionCall %float %f_
OpBranch %25
%25 = OpLabel
%31 = OpLoad %int %i
%32 = OpIAdd %int %31 %int_1
OpStore %i %32
OpBranch %23
%24 = OpLabel
OpStore %_GLF_color %21
OpReturn
OpFunctionEnd
)";

  const std::string callee =
      R"(%f_ = OpFunction %float None %9
%33 = OpLabel
OpBranch %34
%34 = OpLabel
OpLoopMerge %35 %36 None
OpBranch %37
%37 = OpLabel
OpReturnValue %float_1
%36 = OpLabel
OpBranch %34
%35 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(
      predefs + caller + callee, predefs + caller + callee, false, true);
}

TEST_F(InlineTest, Decorated1) {
  // Same test as Simple with the difference
  // that OpFAdd in the outlined function is
  // decorated with RelaxedPrecision
  // Expected result is an equal decoration
  // of the corresponding inlined instruction
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     return bar.x + bar.y;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %9 RelaxedPrecision
)";

  const std::string before =
      R"(%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%15 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %11
%22 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
%24 = OpFunctionCall %float %foo_vf4_ %param
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpDecorate %38 RelaxedPrecision
%void = OpTypeVoid
%11 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%15 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %11
%22 = OpLabel
%32 = OpVariable %_ptr_Function_float Function
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
%34 = OpAccessChain %_ptr_Function_float %param %uint_0
%35 = OpLoad %float %34
%36 = OpAccessChain %_ptr_Function_float %param %uint_1
%37 = OpLoad %float %36
%38 = OpFAdd %float %35 %37
OpStore %32 %38
%24 = OpLoad %float %32
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string nonEntryFuncs =
      R"(%foo_vf4_ = OpFunction %float None %15
%bar = OpFunctionParameter %_ptr_Function_v4float
%27 = OpLabel
%28 = OpAccessChain %_ptr_Function_float %bar %uint_0
%29 = OpLoad %float %28
%30 = OpAccessChain %_ptr_Function_float %bar %uint_1
%31 = OpLoad %float %30
%9 = OpFAdd %float %29 %31
OpReturnValue %9
OpFunctionEnd
)";
  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + before + nonEntryFuncs,
                                              predefs + after + nonEntryFuncs,
                                              false, true);
}

TEST_F(InlineTest, Decorated2) {
  // Same test as Simple with the difference
  // that the Result <id> of the outlined OpFunction
  // is decorated with RelaxedPrecision
  // Expected result is an equal decoration
  // of the created return variable
  //
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     return bar.x + bar.y;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
OpDecorate %foo_vf4_ RelaxedPrecision
)";

  const std::string before =
      R"(%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%21 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %BaseColor
OpStore %param %22
%23 = OpFunctionCall %float %foo_vf4_ %param
%24 = OpCompositeConstruct %v4float %23 %23 %23 %23
OpStore %color %24
%25 = OpLoad %v4float %color
OpStore %gl_FragColor %25
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(OpDecorate %32 RelaxedPrecision
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%uint_1 = OpConstant %uint 1
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %10
%21 = OpLabel
%32 = OpVariable %_ptr_Function_float Function
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%22 = OpLoad %v4float %BaseColor
OpStore %param %22
%34 = OpAccessChain %_ptr_Function_float %param %uint_0
%35 = OpLoad %float %34
%36 = OpAccessChain %_ptr_Function_float %param %uint_1
%37 = OpLoad %float %36
%38 = OpFAdd %float %35 %37
OpStore %32 %38
%23 = OpLoad %float %32
%24 = OpCompositeConstruct %v4float %23 %23 %23 %23
OpStore %color %24
%25 = OpLoad %v4float %color
OpStore %gl_FragColor %25
OpReturn
OpFunctionEnd
)";

  const std::string nonEntryFuncs =
      R"(%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%26 = OpLabel
%27 = OpAccessChain %_ptr_Function_float %bar %uint_0
%28 = OpLoad %float %27
%29 = OpAccessChain %_ptr_Function_float %bar %uint_1
%30 = OpLoad %float %29
%31 = OpFAdd %float %28 %30
OpReturnValue %31
OpFunctionEnd
)";
  SinglePassRunAndCheck<InlineExhaustivePass>(predefs + before + nonEntryFuncs,
                                              predefs + after + nonEntryFuncs,
                                              false, true);
}

TEST_F(InlineTest, DeleteName) {
  // Test that the name of the result id of the call is deleted.
  const std::string before =
      R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
               OpName %main_entry "main_entry"
               OpName %foo_result "foo_result"
               OpName %void_fn "void_fn"
               OpName %foo "foo"
               OpName %foo_entry "foo_entry"
       %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
        %foo = OpFunction %void None %void_fn
  %foo_entry = OpLabel
               OpReturn
               OpFunctionEnd
       %main = OpFunction %void None %void_fn
 %main_entry = OpLabel
 %foo_result = OpFunctionCall %void %foo
               OpReturn
               OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpName %main "main"
OpName %main_entry "main_entry"
OpName %void_fn "void_fn"
OpName %foo "foo"
OpName %foo_entry "foo_entry"
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%foo = OpFunction %void None %void_fn
%foo_entry = OpLabel
OpReturn
OpFunctionEnd
%main = OpFunction %void None %void_fn
%main_entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<InlineExhaustivePass>(before, after, false, true);
}

TEST_F(InlineTest, SetParent) {
  // Test that after inlining all basic blocks have the correct parent.
  const std::string text =
      R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main"
               OpName %main "main"
               OpName %main_entry "main_entry"
               OpName %foo_result "foo_result"
               OpName %void_fn "void_fn"
               OpName %foo "foo"
               OpName %foo_entry "foo_entry"
       %void = OpTypeVoid
    %void_fn = OpTypeFunction %void
        %foo = OpFunction %void None %void_fn
  %foo_entry = OpLabel
               OpReturn
               OpFunctionEnd
       %main = OpFunction %void None %void_fn
 %main_entry = OpLabel
 %foo_result = OpFunctionCall %void %foo
               OpReturn
               OpFunctionEnd
)";

  std::unique_ptr<IRContext> context =
      BuildModule(SPV_ENV_UNIVERSAL_1_2, nullptr, text);
  InlineExhaustivePass pass;
  pass.Run(context.get());

  for (Function& func : *context->module()) {
    for (BasicBlock& bb : func) {
      EXPECT_TRUE(bb.GetParent() == &func);
    }
  }
}

TEST_F(InlineTest, OpVariableWithInit) {
  // Check that there is a store that corresponds to the initializer.  This
  // test makes sure that is a store to the variable in the loop and before any
  // load.
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NOT: OpFunctionEnd
; CHECK: [[var:%\w+]] = OpVariable %_ptr_Function_float Function %float_0
; CHECK: OpLoopMerge [[outer_merge:%\w+]]
; CHECK-NOT: OpLoad %float [[var]]
; CHECK: OpStore [[var]] %float_0
; CHECK: OpFunctionEnd
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %o
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpDecorate %o Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %7 = OpTypeFunction %float
%_ptr_Function_float = OpTypePointer Function %float
    %float_0 = OpConstant %float 0
       %bool = OpTypeBool
    %float_1 = OpConstant %float 1
%_ptr_Output_float = OpTypePointer Output %float
          %o = OpVariable %_ptr_Output_float Output
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%_ptr_Input_int = OpTypePointer Input %int
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpStore %o %float_0
               OpBranch %34
         %34 = OpLabel
         %39 = OpPhi %int %int_0 %5 %47 %37
               OpLoopMerge %36 %37 None
               OpBranch %38
         %38 = OpLabel
         %41 = OpSLessThan %bool %39 %int_2
               OpBranchConditional %41 %35 %36
         %35 = OpLabel
         %42 = OpFunctionCall %float %foo_
         %43 = OpLoad %float %o
         %44 = OpFAdd %float %43 %42
               OpStore %o %44
               OpBranch %37
         %37 = OpLabel
         %47 = OpIAdd %int %39 %int_1
               OpBranch %34
         %36 = OpLabel
               OpReturn
               OpFunctionEnd
       %foo_ = OpFunction %float None %7
          %9 = OpLabel
          %n = OpVariable %_ptr_Function_float Function %float_0
         %13 = OpLoad %float %n
         %15 = OpFOrdEqual %bool %13 %float_0
               OpSelectionMerge %17 None
               OpBranchConditional %15 %16 %17
         %16 = OpLabel
         %19 = OpLoad %float %n
         %20 = OpFAdd %float %19 %float_1
               OpStore %n %20
               OpBranch %17
         %17 = OpLabel
         %21 = OpLoad %float %n
               OpReturnValue %21
               OpFunctionEnd
)";

  SinglePassRunAndMatch<InlineExhaustivePass>(text, true);
}

TEST_F(InlineTest, DontInlineDirectlyRecursiveFunc) {
  // Test that the name of the result id of the call is deleted.
  const std::string test =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%15 = OpConstantNull %_struct_6
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %9
OpReturnValue %15
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InlineExhaustivePass>(test, test, false, true);
}

TEST_F(InlineTest, DontInlineInDirectlyRecursiveFunc) {
  // Test that the name of the result id of the call is deleted.
  const std::string test =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %1 "main"
OpExecutionMode %1 OriginUpperLeft
OpDecorate %2 DescriptorSet 439418829
%void = OpTypeVoid
%4 = OpTypeFunction %void
%float = OpTypeFloat 32
%_struct_6 = OpTypeStruct %float %float
%15 = OpConstantNull %_struct_6
%7 = OpTypeFunction %_struct_6
%1 = OpFunction %void Pure|Const %4
%8 = OpLabel
%2 = OpFunctionCall %_struct_6 %9
OpKill
OpFunctionEnd
%9 = OpFunction %_struct_6 None %7
%10 = OpLabel
%11 = OpFunctionCall %_struct_6 %12
OpReturnValue %15
OpFunctionEnd
%12 = OpFunction %_struct_6 None %7
%13 = OpLabel
%14 = OpFunctionCall %_struct_6 %9
OpReturnValue %15
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InlineExhaustivePass>(test, test, false, true);
}

TEST_F(InlineTest, DontInlineFuncWithOpKillInContinue) {
  const std::string test =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 330
OpName %main "main"
OpName %kill_ "kill("
%void = OpTypeVoid
%3 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %3
%5 = OpLabel
OpBranch %9
%9 = OpLabel
OpLoopMerge %11 %12 None
OpBranch %13
%13 = OpLabel
OpBranchConditional %true %10 %11
%10 = OpLabel
OpBranch %12
%12 = OpLabel
%16 = OpFunctionCall %void %kill_
OpBranch %9
%11 = OpLabel
OpReturn
OpFunctionEnd
%kill_ = OpFunction %void None %3
%7 = OpLabel
OpKill
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InlineExhaustivePass>(test, test, false, true);
}

TEST_F(InlineTest, InlineFuncWithOpKillNotInContinue) {
  const std::string before =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 330
OpName %main "main"
OpName %kill_ "kill("
%void = OpTypeVoid
%3 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %3
%5 = OpLabel
%16 = OpFunctionCall %void %kill_
OpReturn
OpFunctionEnd
%kill_ = OpFunction %void None %3
%7 = OpLabel
OpKill
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 330
OpName %main "main"
OpName %kill_ "kill("
%void = OpTypeVoid
%3 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %3
%5 = OpLabel
OpKill
%18 = OpLabel
OpReturn
OpFunctionEnd
%kill_ = OpFunction %void None %3
%7 = OpLabel
OpKill
OpFunctionEnd
)";

  SetAssembleOptions(SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
  SinglePassRunAndCheck<InlineExhaustivePass>(before, after, false, true);
}

TEST_F(InlineTest, EarlyReturnFunctionInlined) {
  // #version 140
  //
  // in vec4 BaseColor;
  //
  // float foo(vec4 bar)
  // {
  //     if (bar.x < 0.0)
  //         return 0.0;
  //     return bar.x;
  // }
  //
  // void main()
  // {
  //     vec4 color = vec4(foo(BaseColor));
  //     gl_FragColor = color;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %foo_vf4_ "foo(vf4;"
OpName %bar "bar"
OpName %color "color"
OpName %BaseColor "BaseColor"
OpName %param "param"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%14 = OpTypeFunction %float %_ptr_Function_v4float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%bool = OpTypeBool
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string foo =
      R"(%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%27 = OpLabel
%28 = OpAccessChain %_ptr_Function_float %bar %uint_0
%29 = OpLoad %float %28
%30 = OpFOrdLessThan %bool %29 %float_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
OpReturnValue %float_0
%31 = OpLabel
%33 = OpAccessChain %_ptr_Function_float %bar %uint_0
%34 = OpLoad %float %33
OpReturnValue %34
OpFunctionEnd
)";

  const std::string fooMergeReturn =
      R"(%foo_vf4_ = OpFunction %float None %14
%bar = OpFunctionParameter %_ptr_Function_v4float
%27 = OpLabel
%41 = OpVariable %_ptr_Function_bool Function %false
%36 = OpVariable %_ptr_Function_float Function
OpSelectionMerge %35 None
OpSwitch %uint_0 %38
%38 = OpLabel
%28 = OpAccessChain %_ptr_Function_float %bar %uint_0
%29 = OpLoad %float %28
%30 = OpFOrdLessThan %bool %29 %float_0
OpSelectionMerge %31 None
OpBranchConditional %30 %32 %31
%32 = OpLabel
OpStore %41 %true
OpStore %36 %float_0
OpBranch %35
%31 = OpLabel
%33 = OpAccessChain %_ptr_Function_float %bar %uint_0
%34 = OpLoad %float %33
OpStore %41 %true
OpStore %36 %34
OpBranch %35
%35 = OpLabel
%37 = OpLoad %float %36
OpReturnValue %37
OpFunctionEnd
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%22 = OpLabel
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
%24 = OpFunctionCall %float %foo_vf4_ %param
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%false = OpConstantFalse %bool
%_ptr_Function_bool = OpTypePointer Function %bool
%true = OpConstantTrue %bool
%main = OpFunction %void None %10
%22 = OpLabel
%43 = OpVariable %_ptr_Function_bool Function %false
%44 = OpVariable %_ptr_Function_float Function
%45 = OpVariable %_ptr_Function_float Function
%color = OpVariable %_ptr_Function_v4float Function
%param = OpVariable %_ptr_Function_v4float Function
%23 = OpLoad %v4float %BaseColor
OpStore %param %23
OpStore %43 %false
OpSelectionMerge %55 None
OpSwitch %uint_0 %47
%47 = OpLabel
%48 = OpAccessChain %_ptr_Function_float %param %uint_0
%49 = OpLoad %float %48
%50 = OpFOrdLessThan %bool %49 %float_0
OpSelectionMerge %52 None
OpBranchConditional %50 %51 %52
%51 = OpLabel
OpStore %43 %true
OpStore %44 %float_0
OpBranch %55
%52 = OpLabel
%53 = OpAccessChain %_ptr_Function_float %param %uint_0
%54 = OpLoad %float %53
OpStore %43 %true
OpStore %44 %54
OpBranch %55
%55 = OpLabel
%56 = OpLoad %float %44
OpStore %45 %56
%24 = OpLoad %float %45
%25 = OpCompositeConstruct %v4float %24 %24 %24 %24
OpStore %color %25
%26 = OpLoad %v4float %color
OpStore %gl_FragColor %26
OpReturn
OpFunctionEnd
)";

  // The early return case must be handled by merge-return first.
  AddPass<MergeReturnPass>();
  AddPass<InlineExhaustivePass>();
  RunAndCheck(predefs + before + foo, predefs + after + fooMergeReturn);
}

TEST_F(InlineTest, EarlyReturnNotAppearingLastInFunctionInlined) {
  // Example from https://github.com/KhronosGroup/SPIRV-Tools/issues/755
  //
  // Original example is derived from:
  //
  // #version 450
  //
  // float foo() {
  //     if (true) {
  //     }
  // }
  //
  // void main() { foo(); }
  //
  // But the order of basic blocks in foo is changed so that the return
  // block is listed second-last.  There is only one return in the callee
  // but it does not appear last.

  const std::string predefs =
      R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
%void = OpTypeVoid
%4 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
)";

  const std::string foo =
      R"(%foo_ = OpFunction %void None %4
%7 = OpLabel
OpSelectionMerge %8 None
OpBranchConditional %true %9 %8
%8 = OpLabel
OpReturn
%9 = OpLabel
OpBranch %8
OpFunctionEnd
)";

  const std::string fooMergeReturn =
      R"(%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%_ptr_Function_bool = OpTypePointer Function %bool
%foo_ = OpFunction %void None %4
%7 = OpLabel
%18 = OpVariable %_ptr_Function_bool Function %false
OpSelectionMerge %12 None
OpSwitch %uint_0 %13
%13 = OpLabel
OpSelectionMerge %8 None
OpBranchConditional %true %9 %8
%8 = OpLabel
OpStore %18 %true
OpBranch %12
%9 = OpLabel
OpBranch %8
%12 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string before =
      R"(%main = OpFunction %void None %4
%10 = OpLabel
%11 = OpFunctionCall %void %foo_
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %4
%10 = OpLabel
%19 = OpVariable %_ptr_Function_bool Function %false
OpStore %19 %false
OpSelectionMerge %24 None
OpSwitch %uint_0 %21
%21 = OpLabel
OpSelectionMerge %22 None
OpBranchConditional %true %23 %22
%22 = OpLabel
OpStore %19 %true
OpBranch %24
%23 = OpLabel
OpBranch %22
%24 = OpLabel
OpReturn
OpFunctionEnd
)";

  // The early return case must be handled by merge-return first.
  AddPass<MergeReturnPass>();
  AddPass<InlineExhaustivePass>();
  RunAndCheck(predefs + foo + before, predefs + fooMergeReturn + after);
}

TEST_F(InlineTest, CalleeWithSingleReturnNeedsSingleTripLoopWrapper) {
  // The case from https://github.com/KhronosGroup/SPIRV-Tools/issues/2018
  //
  // The callee has a single return, but needs single-trip loop wrapper
  // to be inlined because the return is in a selection structure.

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %_GLF_color
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %f_ "f("
OpName %i "i"
OpName %_GLF_color "_GLF_color"
OpDecorate %_GLF_color Location 0
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%9 = OpTypeFunction %float
%float_1 = OpConstant %float 1
%bool = OpTypeBool
%false = OpConstantFalse %bool
%true = OpConstantTrue %bool
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_GLF_color = OpVariable %_ptr_Output_v4float Output
%float_0 = OpConstant %float 0
%21 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%22 = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
)";

  const std::string new_predefs =
      R"(%_ptr_Function_float = OpTypePointer Function %float
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Function_bool = OpTypePointer Function %bool
)";

  const std::string main_before =
      R"(%main = OpFunction %void None %7
%23 = OpLabel
%i = OpVariable %_ptr_Function_int Function
OpStore %i %int_0
OpBranch %24
%24 = OpLabel
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
%28 = OpLoad %int %i
%29 = OpSLessThan %bool %28 %int_1
OpBranchConditional %29 %30 %25
%30 = OpLabel
OpStore %_GLF_color %21
%31 = OpFunctionCall %float %f_
OpBranch %26
%26 = OpLabel
%32 = OpLoad %int %i
%33 = OpIAdd %int %32 %int_1
OpStore %i %33
OpBranch %24
%25 = OpLabel
OpStore %_GLF_color %22
OpReturn
OpFunctionEnd
)";

  const std::string main_after =
      R"(%main = OpFunction %void None %7
%23 = OpLabel
%46 = OpVariable %_ptr_Function_bool Function %false
%47 = OpVariable %_ptr_Function_float Function
%48 = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %i %int_0
OpBranch %24
%24 = OpLabel
OpLoopMerge %25 %26 None
OpBranch %27
%27 = OpLabel
%28 = OpLoad %int %i
%29 = OpSLessThan %bool %28 %int_1
OpBranchConditional %29 %30 %25
%30 = OpLabel
OpStore %_GLF_color %21
OpStore %46 %false
OpSelectionMerge %53 None
OpSwitch %uint_0 %50
%50 = OpLabel
OpSelectionMerge %52 None
OpBranchConditional %true %51 %52
%51 = OpLabel
OpStore %46 %true
OpStore %47 %float_1
OpBranch %53
%52 = OpLabel
OpStore %46 %true
OpStore %47 %float_1
OpBranch %53
%53 = OpLabel
%54 = OpLoad %float %47
OpStore %48 %54
%31 = OpLoad %float %48
OpBranch %26
%26 = OpLabel
%32 = OpLoad %int %i
%33 = OpIAdd %int %32 %int_1
OpStore %i %33
OpBranch %24
%25 = OpLabel
OpStore %_GLF_color %22
OpReturn
OpFunctionEnd
)";

  const std::string callee =
      R"(%f_ = OpFunction %float None %9
%34 = OpLabel
OpSelectionMerge %35 None
OpBranchConditional %true %36 %35
%36 = OpLabel
OpReturnValue %float_1
%35 = OpLabel
OpReturnValue %float_1
OpFunctionEnd
)";

  const std::string calleeMergeReturn =
      R"(%f_ = OpFunction %float None %9
%34 = OpLabel
%45 = OpVariable %_ptr_Function_bool Function %false
%39 = OpVariable %_ptr_Function_float Function
OpSelectionMerge %37 None
OpSwitch %uint_0 %41
%41 = OpLabel
OpSelectionMerge %35 None
OpBranchConditional %true %36 %35
%36 = OpLabel
OpStore %45 %true
OpStore %39 %float_1
OpBranch %37
%35 = OpLabel
OpStore %45 %true
OpStore %39 %float_1
OpBranch %37
%37 = OpLabel
%40 = OpLoad %float %39
OpReturnValue %40
OpFunctionEnd
)";

  // The early return case must be handled by merge-return first.
  AddPass<MergeReturnPass>();
  AddPass<InlineExhaustivePass>();
  RunAndCheck(predefs + main_before + callee,
              predefs + new_predefs + main_after + calleeMergeReturn);
}

TEST_F(InlineTest, ForwardReferencesInPhiInlined) {
  // The basic structure of the test case is like this:
  //
  // int foo() {
  //   int result = 1;
  //   if (true) {
  //      result = 1;
  //   }
  //   return result;
  // }
  //
  // void main() {
  //  int x = foo();
  // }
  //
  // but with modifications: Using Phi instead of load/store, and the
  // return block in foo appears before the "then" block.

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main"
OpSource GLSL 450
OpName %main "main"
OpName %foo_ "foo("
OpName %x "x"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%8 = OpTypeFunction %int
%bool = OpTypeBool
%true = OpConstantTrue %bool
%int_0 = OpConstant %int 0
%_ptr_Function_int = OpTypePointer Function %int
)";

  const std::string callee =
      R"(%foo_ = OpFunction %int None %8
%13 = OpLabel
%14 = OpCopyObject %int %int_0
OpSelectionMerge %15 None
OpBranchConditional %true %16 %15
%15 = OpLabel
%17 = OpPhi %int %14 %13 %18 %16
OpReturnValue %17
%16 = OpLabel
%18 = OpCopyObject %int %int_0
OpBranch %15
OpFunctionEnd
)";

  const std::string calleeMergeReturn =
      R"(%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%false = OpConstantFalse %bool
%_ptr_Function_bool = OpTypePointer Function %bool
%foo_ = OpFunction %int None %8
%13 = OpLabel
%29 = OpVariable %_ptr_Function_bool Function %false
%22 = OpVariable %_ptr_Function_int Function
OpSelectionMerge %21 None
OpSwitch %uint_0 %24
%24 = OpLabel
%14 = OpCopyObject %int %int_0
OpSelectionMerge %15 None
OpBranchConditional %true %16 %15
%15 = OpLabel
%17 = OpPhi %int %14 %24 %18 %16
OpStore %29 %true
OpStore %22 %17
OpBranch %21
%16 = OpLabel
%18 = OpCopyObject %int %int_0
OpBranch %15
%21 = OpLabel
%23 = OpLoad %int %22
OpReturnValue %23
OpFunctionEnd
)";

  const std::string before =
      R"(%main = OpFunction %void None %6
%19 = OpLabel
%x = OpVariable %_ptr_Function_int Function
%20 = OpFunctionCall %int %foo_
OpStore %x %20
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %6
%19 = OpLabel
%30 = OpVariable %_ptr_Function_bool Function %false
%31 = OpVariable %_ptr_Function_int Function
%32 = OpVariable %_ptr_Function_int Function
%x = OpVariable %_ptr_Function_int Function
OpStore %30 %false
OpSelectionMerge %40 None
OpSwitch %uint_0 %34
%34 = OpLabel
%35 = OpCopyObject %int %int_0
OpSelectionMerge %36 None
OpBranchConditional %true %38 %36
%36 = OpLabel
%37 = OpPhi %int %35 %34 %39 %38
OpStore %30 %true
OpStore %31 %37
OpBranch %40
%38 = OpLabel
%39 = OpCopyObject %int %int_0
OpBranch %36
%40 = OpLabel
%41 = OpLoad %int %31
OpStore %32 %41
%20 = OpLoad %int %32
OpStore %x %20
OpReturn
OpFunctionEnd
)";

  AddPass<MergeReturnPass>();
  AddPass<InlineExhaustivePass>();
  RunAndCheck(predefs + callee + before, predefs + calleeMergeReturn + after);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    Empty modules
//    Modules without function definitions
//    Modules in which all functions do not call other functions
//    Caller and callee both accessing the same global variable
//    Functions with OpLine & OpNoLine
//    Others?

// TODO(dneto): Test suggestions from code review
// https://github.com/KhronosGroup/SPIRV-Tools/pull/534
//
//    Callee function returns a value generated outside the callee,
//      e.g. a constant value. This might exercise some logic not yet
//      exercised by the current tests: the false branch in the "if"
//      inside the SpvOpReturnValue case in InlinePass::GenInlineCode?
//    SampledImage before function call, but callee is only single block.
//      Then the SampledImage instruction is not cloned. Documents existing
//      behaviour.
//    SampledImage after function call. It is not cloned or changed.

}  // namespace
}  // namespace opt
}  // namespace spvtools
