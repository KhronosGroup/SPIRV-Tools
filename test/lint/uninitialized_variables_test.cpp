#include "source/lint/uninitialized_analysis.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.h"

#include <string>
#include <vector>

namespace spvtools {
namespace lint {
namespace {

void CLIMessageConsumer(spv_message_level_t level, const char*,
                        const spv_position_t& position, const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: line " << position.index << ": " << message << "\n";
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: line " << position.index << ": " << message
                << "\n";
      break;
    case SPV_MSG_INFO:
      std::cout << "info: line " << position.index << ": " << message << "\n";
      break;
    default:
      break;
  }
}

uninitialized_variables::State SNo() {
  return uninitialized_variables::State::NewLeaf(
      uninitialized_variables::Initialized::No);
}

uninitialized_variables::State SYes() {
  return uninitialized_variables::State::NewLeaf(
      uninitialized_variables::Initialized::Yes);
}

uninitialized_variables::State SUnk() {
  return
  uninitialized_variables::State::NewLeaf(uninitialized_variables::Initialized::Unknown);
}

uninitialized_variables::State SC(
    std::initializer_list<uninitialized_variables::State> components) {
  return uninitialized_variables::State::NewComposite(components);
}

std::string Preamble() {
  return R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main"
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
  )";
}

} // namespace

class UninitializedVariablesTest : public ::testing::Test {
 protected:
  struct BadLocalLoad {
    uint32_t id;
    uninitialized_variables::State state_have = SNo();
    uninitialized_variables::State state_missing = SYes();

    friend bool operator==(const BadLocalLoad& lhs, const BadLocalLoad& rhs) {
      return lhs.id == rhs.id &&
             lhs.state_have.TryEquals(rhs.state_have).value_or(false) &&
             lhs.state_missing.TryEquals(rhs.state_missing).value_or(false);
    }

    friend std::ostream& operator<<(std::ostream& stream,
                                    const BadLocalLoad& l) {
      stream << "{ load_id: " << l.id << ", have: " << l.state_have
             << ", missing: " << l.state_missing << " }";
      return stream;
    }
  };
  std::unique_ptr<opt::IRContext> context_;
  std::vector<BadLocalLoad> bad_loads_;

  void Build(const std::string& text) {
    context_ = BuildModule(SPV_ENV_UNIVERSAL_1_0, CLIMessageConsumer, text,
                           SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ASSERT_NE(nullptr, context_.get());
    opt::Module* module = context_->module();
    ASSERT_NE(nullptr, module);
    // First function should have the given ID.
    ASSERT_NE(module->begin(), module->end());
    ASSERT_EQ(1, module->end() - module->begin());
    opt::Function& function = *module->begin();
    uninitialized_variables::ModuleAnalysis analysis(*context_);
    uninitialized_variables::CheckUninitializedResult result = analysis.Run();
    ASSERT_EQ(result.globals.size(), 0) << "Test is only for function local variables";
    ASSERT_LE(result.locals.size(), 1);
    if (!result.locals.empty()) {
        ASSERT_EQ(result.locals[0].function_id, function.result_id());
        for (auto& bl : result.locals[0].bad_accesses) {
        bad_loads_.push_back(BadLocalLoad{
          .id = bl.load->result_id(),
          .state_have = bl.state_have,
          .state_missing = bl.state_missing,
        });
      }
    }
  }
};

namespace {

class UninitializedGlobalsTest : public ::testing::Test {
 protected:
  struct BadGlobalLoad {
    uint32_t load_result_id;
    uninitialized_variables::State state_have;
    uninitialized_variables::State state_missing;
    std::vector<uint32_t> call_trace;

    friend bool operator==(const BadGlobalLoad& lhs, const BadGlobalLoad& rhs) {
      return lhs.load_result_id == rhs.load_result_id &&
             lhs.call_trace == rhs.call_trace &&
             lhs.state_have.TryEquals(rhs.state_have).value_or(false) &&
             lhs.state_missing.TryEquals(rhs.state_missing).value_or(false);
    }

    friend std::ostream& operator<<(std::ostream& stream,
                                    const BadGlobalLoad& l) {
      stream << "{ load_id: " << l.load_result_id << ", have: " << l.state_have
             << ", missing: " << l.state_missing << ", trace: [ ";
      for (uint32_t id : l.call_trace) {
        stream << id << " ";
      }
      stream << "] }";
      return stream;
    }
  };
  std::unique_ptr<opt::IRContext> context_;
  std::unordered_map<uint32_t, std::vector<BadGlobalLoad>> ep_results;

  void Build(const std::string& text) {
    context_ = BuildModule(SPV_ENV_UNIVERSAL_1_0, CLIMessageConsumer, text,
                           SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS);
    ASSERT_NE(nullptr, context_.get());
    opt::Module* module = context_->module();
    ASSERT_NE(nullptr, module);
    ASSERT_NE(module->begin(), module->end());
    uninitialized_variables::ModuleAnalysis analysis(*context_);
    uninitialized_variables::CheckUninitializedResult result = analysis.Run();
    ASSERT_EQ(result.locals.size(), 0)
        << "Test is only for global (module private) variables";
    if (!result.globals.empty()) {
      for (auto& ep_result : result.globals) {
        std::vector<BadGlobalLoad> bad_loads;
        for (auto& bl : ep_result.bad_accesses) {
          std::vector<uint32_t> call_trace;
          call_trace.reserve(bl.call_sequence.size());
          for (auto op_call : bl.call_sequence) {
            call_trace.push_back(op_call->result_id());
          }
          bad_loads.push_back(BadGlobalLoad{
              .load_result_id = bl.op_load->result_id(),
              .state_have = bl.state_have,
              .state_missing = bl.state_missing,
              .call_trace = call_trace,
          });
        }
        ep_results.insert(
            {ep_result.op_entrypoint->GetSingleWordInOperand(1), bad_loads});
      }
    }
  }
};

} // namespace

// ====================
// LOCAL VARIABLE TESTS
// ====================

TEST_F(UninitializedVariablesTest, AllOkTest) {
   // int a;
   // int b = a;
   // int five = 5;
   // int d;
   // if (five == 5) {
   //     int c = 0;
   //     for (int i; i < five; ++i) {
   //         c += 2;
   //     }
   // } else {
   //     if (five < 10) {
   //     } else if (five < 20) {
   //         d = 1;
   //     } else {
   //         d = 2 * d;
   //     }
   // }
   ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
         %void = OpTypeVoid
            %3 = OpTypeFunction %void
          %int = OpTypeInt 32 1
  %_ptr_Function_int = OpTypePointer Function %int
        %int_2 = OpConstant %int 2
        %int_4 = OpConstant %int 4
         %main = OpFunction %void None %3
            %5 = OpLabel
            %a = OpVariable %_ptr_Function_int Function
            %b = OpVariable %_ptr_Function_int Function
            %c = OpVariable %_ptr_Function_int Function
                 OpStore %a %int_2
           %10 = OpLoad %int %a
                 OpSelectionMerge %13 None
                 OpSwitch %10 %12 1 %11 2 %12 3 %12
           %12 = OpLabel
                 OpStore %b %int_4
                 OpBranch %13
           %11 = OpLabel
                 OpStore %b %int_2
                 OpBranch %13
           %13 = OpLabel
           %19 = OpLoad %int %b
           %20 = OpIMul %int %int_2 %19
                 OpStore %c %20
                 OpReturn
                 OpFunctionEnd
  )"));
  EXPECT_EQ(0, bad_loads_.size());
}

TEST_F(UninitializedVariablesTest, SimpleTest) {
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
  %void = OpTypeVoid
            %3 = OpTypeFunction %void
          %int = OpTypeInt 32 1
  %_ptr_Function_int = OpTypePointer Function %int
        %int_5 = OpConstant %int 5
         %bool = OpTypeBool
        %int_0 = OpConstant %int 0
        %int_2 = OpConstant %int 2
        %int_1 = OpConstant %int 1
       %int_10 = OpConstant %int 10
       %int_20 = OpConstant %int 20
         %main = OpFunction %void None %3
            %5 = OpLabel
            %b = OpVariable %_ptr_Function_int Function
            %a = OpVariable %_ptr_Function_int Function
         %five = OpVariable %_ptr_Function_int Function
            %c = OpVariable %_ptr_Function_int Function
            %i = OpVariable %_ptr_Function_int Function
            %d = OpVariable %_ptr_Function_int Function
           %10 = OpLoad %int %a
                 OpStore %b %10
                 OpStore %five %int_5
           %13 = OpLoad %int %five
           %15 = OpIEqual %bool %13 %int_5
                 OpSelectionMerge %17 None
                 OpBranchConditional %15 %16 %35
           %16 = OpLabel
                 OpStore %c %int_0
                 OpBranch %20
           %20 = OpLabel
                 OpLoopMerge %22 %23 None
                 OpBranch %24
           %24 = OpLabel
           %26 = OpLoad %int %i
           %27 = OpLoad %int %five
           %28 = OpSLessThan %bool %26 %27
                 OpBranchConditional %28 %21 %22
           %21 = OpLabel
           %30 = OpLoad %int %c
           %31 = OpIAdd %int %30 %int_2
                 OpStore %c %31
                 OpBranch %23
           %23 = OpLabel
           %32 = OpLoad %int %i
           %34 = OpIAdd %int %32 %int_1
                 OpStore %i %34
                 OpBranch %20
           %22 = OpLabel
                 OpBranch %17
           %35 = OpLabel
           %36 = OpLoad %int %five
           %38 = OpSLessThan %bool %36 %int_10
                 OpSelectionMerge %40 None
                 OpBranchConditional %38 %39 %41
           %39 = OpLabel
                 OpBranch %40
           %41 = OpLabel
           %42 = OpLoad %int %five
           %44 = OpSLessThan %bool %42 %int_20
                 OpSelectionMerge %46 None
                 OpBranchConditional %44 %45 %48
           %45 = OpLabel
                 OpStore %d %int_1
                 OpBranch %46
           %48 = OpLabel
           %49 = OpLoad %int %d
           %50 = OpIMul %int %int_2 %49
                 OpStore %d %50
                 OpBranch %46
           %46 = OpLabel
                 OpBranch %40
           %40 = OpLabel
                 OpBranch %17
           %17 = OpLabel
                 OpReturn
                 OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 10 },
    BadLocalLoad { .id = 26 },
    BadLocalLoad { .id = 32 },
    BadLocalLoad { .id = 49 }
  ));
}

TEST_F(UninitializedVariablesTest, UninitInFirstIterTest) {
  // int a;
  // for (int i = 0; i < 5; i++) {
  //     if (a < 5) {
  //         a = 1;
  //     } else {
  //         a = 2;
  //     }
  //     int b = 2 * a;
  // }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
         %void = OpTypeVoid
            %3 = OpTypeFunction %void
          %int = OpTypeInt 32 1
  %_ptr_Function_int = OpTypePointer Function %int
        %int_0 = OpConstant %int 0
        %int_5 = OpConstant %int 5
         %bool = OpTypeBool
        %int_1 = OpConstant %int 1
        %int_2 = OpConstant %int 2
         %main = OpFunction %void None %3
            %5 = OpLabel
            %i = OpVariable %_ptr_Function_int Function
            %a = OpVariable %_ptr_Function_int Function
            %b = OpVariable %_ptr_Function_int Function
                 OpStore %i %int_0
                 OpBranch %10
           %10 = OpLabel
                 OpLoopMerge %12 %13 None
                 OpBranch %14
           %14 = OpLabel
           %15 = OpLoad %int %i
           %18 = OpSLessThan %bool %15 %int_5
                 OpBranchConditional %18 %11 %12
           %11 = OpLabel
           %20 = OpLoad %int %a
           %21 = OpSLessThan %bool %20 %int_5
                 OpSelectionMerge %23 None
                 OpBranchConditional %21 %22 %25
           %22 = OpLabel
                 OpStore %a %int_1
                 OpBranch %23
           %25 = OpLabel
                 OpStore %a %int_2
                 OpBranch %23
           %23 = OpLabel
           %28 = OpLoad %int %a
           %29 = OpIMul %int %int_2 %28
                 OpStore %b %29
                 OpBranch %13
           %13 = OpLabel
           %30 = OpLoad %int %i
           %31 = OpIAdd %int %30 %int_1
                 OpStore %i %31
                 OpBranch %10
           %12 = OpLabel
                 OpReturn
                 OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 20 }
  ));
}

TEST_F(UninitializedVariablesTest, VecArrayTest) {
  // vec2 vecs[4];
  // vec2 v = vecs[0];
  // float f = vecs[1].x;
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
         %void = OpTypeVoid
            %3 = OpTypeFunction %void
        %float = OpTypeFloat 32
      %v2float = OpTypeVector %float 2
  %_ptr_Function_v2float = OpTypePointer Function %v2float
         %uint = OpTypeInt 32 0
       %uint_4 = OpConstant %uint 4
  %_arr_v2float_uint_4 = OpTypeArray %v2float %uint_4
  %_ptr_Function__arr_v2float_uint_4 = OpTypePointer Function %_arr_v2float_uint_4
          %int = OpTypeInt 32 1
        %int_0 = OpConstant %int 0
  %_ptr_Function_float = OpTypePointer Function %float
        %int_1 = OpConstant %int 1
       %uint_0 = OpConstant %uint 0
         %main = OpFunction %void None %3
            %5 = OpLabel
            %v = OpVariable %_ptr_Function_v2float Function
         %vecs = OpVariable %_ptr_Function__arr_v2float_uint_4 Function
            %f = OpVariable %_ptr_Function_float Function
           %17 = OpAccessChain %_ptr_Function_v2float %vecs %int_0
           %18 = OpLoad %v2float %17
                 OpStore %v %18
           %23 = OpAccessChain %_ptr_Function_float %vecs %int_1 %uint_0
           %24 = OpLoad %float %23
                 OpStore %f %24
                 OpReturn
                 OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 18, .state_have = SNo(), .state_missing = SC({SYes(), SNo(), SNo(), SNo()}) },
    BadLocalLoad { .id = 24, .state_have = SNo(), .state_missing = SC({SNo(), SC({SYes(), SNo()}), SNo(), SNo()}) }
  ));
}

TEST_F(UninitializedVariablesTest, SimpleStructTest) {
  // struct Foo {
  //     int one;
  //     float two;
  // };
  //
  // void main() {
  //     Foo foo_good = Foo( 1, 2.0);
  //     int a = foo_good.one;
  //     Foo foo_bad;
  //     foo_bad.two = 3.0;
  //     a = foo_bad.one;
  // }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
%void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
        %Foo = OpTypeStruct %int %float
%_ptr_Function_Foo = OpTypePointer Function %Foo
      %int_1 = OpConstant %int 1
    %float_2 = OpConstant %float 2
         %13 = OpConstantComposite %Foo %int_1 %float_2
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
    %float_3 = OpConstant %float 3
%_ptr_Function_float = OpTypePointer Function %float
       %main = OpFunction %void None %3
          %5 = OpLabel
   %foo_good = OpVariable %_ptr_Function_Foo Function
          %a = OpVariable %_ptr_Function_int Function
    %foo_bad = OpVariable %_ptr_Function_Foo Function
               OpStore %foo_good %13
         %17 = OpAccessChain %_ptr_Function_int %foo_good %int_0
         %18 = OpLoad %int %17
               OpStore %a %18
         %22 = OpAccessChain %_ptr_Function_float %foo_bad %int_1
               OpStore %22 %float_3
         %23 = OpAccessChain %_ptr_Function_int %foo_bad %int_0
         %24 = OpLoad %int %23
               OpStore %a %24
               OpReturn
               OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 24, .state_have = SC({SNo(), SYes()}), .state_missing = SC({SYes(), SNo()}) }
  ));
}

TEST_F(UninitializedVariablesTest, VecMatTest) {
// void main() {
//     mat2 matrix;
//     matrix[0] = vec2(5);
//     float ip = dot(matrix[0], matrix[0]);
//     float ip_bad = dot(matrix[1], matrix[0]);
//     matrix[1] = vec2(6);
//     float ip_ok = dot(matrix[1], matrix[1]);
//
//     mat2 mm;
//     if (ip_ok > 0.1) {
//         mm[0][0] = 1.0;
//         mm[1][0] = 2.0;
//     } else {
//         mm[0][0] = -1.0;
//         mm[1][0] = -2.0;
//     }
//     mat2 prod_bad = mm * matrix;
//     mm[0][1] = 0;
//     mm[1][1] = 0;
//     mat2 prod = mm * matrix;
// }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%mat2v2float = OpTypeMatrix %v2float 2
%_ptr_Function_mat2v2float = OpTypePointer Function %mat2v2float
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_5 = OpConstant %float 5
         %14 = OpConstantComposite %v2float %float_5 %float_5
%_ptr_Function_v2float = OpTypePointer Function %v2float
%_ptr_Function_float = OpTypePointer Function %float
      %int_1 = OpConstant %int 1
    %float_6 = OpConstant %float 6
         %32 = OpConstantComposite %v2float %float_6 %float_6
%float_0_100000001 = OpConstant %float 0.100000001
       %bool = OpTypeBool
    %float_1 = OpConstant %float 1
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
    %float_2 = OpConstant %float 2
   %float_n1 = OpConstant %float -1
   %float_n2 = OpConstant %float -2
    %float_0 = OpConstant %float 0
     %uint_1 = OpConstant %uint 1
       %main = OpFunction %void None %3
          %5 = OpLabel
     %matrix = OpVariable %_ptr_Function_mat2v2float Function
         %ip = OpVariable %_ptr_Function_float Function
     %ip_bad = OpVariable %_ptr_Function_float Function
      %ip_ok = OpVariable %_ptr_Function_float Function
         %mm = OpVariable %_ptr_Function_mat2v2float Function
   %prod_bad = OpVariable %_ptr_Function_mat2v2float Function
       %prod = OpVariable %_ptr_Function_mat2v2float Function
         %16 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
               OpStore %16 %14
         %19 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
         %20 = OpLoad %v2float %19
         %21 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
         %22 = OpLoad %v2float %21
         %23 = OpDot %float %20 %22
               OpStore %ip %23
         %26 = OpAccessChain %_ptr_Function_v2float %matrix %int_1
         %27 = OpLoad %v2float %26
         %28 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
         %29 = OpLoad %v2float %28
         %30 = OpDot %float %27 %29
               OpStore %ip_bad %30
         %33 = OpAccessChain %_ptr_Function_v2float %matrix %int_1
               OpStore %33 %32
         %35 = OpAccessChain %_ptr_Function_v2float %matrix %int_1
         %36 = OpLoad %v2float %35
         %37 = OpAccessChain %_ptr_Function_v2float %matrix %int_1
         %38 = OpLoad %v2float %37
         %39 = OpDot %float %36 %38
               OpStore %ip_ok %39
         %40 = OpLoad %float %ip_ok
         %43 = OpFOrdGreaterThan %bool %40 %float_0_100000001
               OpSelectionMerge %45 None
               OpBranchConditional %43 %44 %53
         %44 = OpLabel
         %50 = OpAccessChain %_ptr_Function_float %mm %int_0 %uint_0
               OpStore %50 %float_1
         %52 = OpAccessChain %_ptr_Function_float %mm %int_1 %uint_0
               OpStore %52 %float_2
               OpBranch %45
         %53 = OpLabel
         %55 = OpAccessChain %_ptr_Function_float %mm %int_0 %uint_0
               OpStore %55 %float_n1
         %57 = OpAccessChain %_ptr_Function_float %mm %int_1 %uint_0
               OpStore %57 %float_n2
               OpBranch %45
         %45 = OpLabel
         %59 = OpLoad %mat2v2float %mm
         %60 = OpLoad %mat2v2float %matrix
         %61 = OpMatrixTimesMatrix %mat2v2float %59 %60
               OpStore %prod_bad %61
         %64 = OpAccessChain %_ptr_Function_float %mm %int_0 %uint_1
               OpStore %64 %float_0
         %65 = OpAccessChain %_ptr_Function_float %mm %int_1 %uint_1
               OpStore %65 %float_0
         %67 = OpLoad %mat2v2float %mm
         %68 = OpLoad %mat2v2float %matrix
         %69 = OpMatrixTimesMatrix %mat2v2float %67 %68
               OpStore %prod %69
               OpReturn
               OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 27, .state_have = SC({SYes(), SNo()}), .state_missing = SC({SNo(), SYes()}) },
    BadLocalLoad { .id = 59, .state_have = SC({SC({SYes(), SNo()}), SC({SYes(), SNo()})}), .state_missing = SC({SC({SNo(), SYes()}), SC({SNo(), SYes()})}) }
  ));
}

TEST_F(UninitializedVariablesTest, IfNestedTest) {
  // void main() {
  //     int a;
  //     int b;
  //     int c;
  //     int d;
  //     int never_touched;
  //     float five = 5.0;
  //     float six = 6.0;
  //     if (five < 10.0) {
  //         b = 4;
  //         if (five > 20.0) {
  //             d = a; // a bad
  //             b = 5;
  //             c = 3;
  //             int o = b + c;
  //         } else {
  //             int local = 0;
  //             a = 1;
  //             d = 21;
  //             c = 4;
  //         }
  //     } else {
  //         if (five == 2.0) {
  //             if (six < 3.0) {
  //                 a = 3;
  //             }
  //             int bad = 5 * a; // a bad
  //             d = 12;
  //         } else {
  //             d = 10;
  //         }
  //         b = 6;
  //     }
  //     int e = a + b; // a bad
  //     int f = c * 2 + d; // c bad
  // }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %float_5 = OpConstant %float 5
    %float_6 = OpConstant %float 6
   %float_10 = OpConstant %float 10
       %bool = OpTypeBool
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_4 = OpConstant %int 4
   %float_20 = OpConstant %float 20
      %int_5 = OpConstant %int 5
      %int_3 = OpConstant %int 3
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %int_21 = OpConstant %int 21
    %float_2 = OpConstant %float 2
    %float_3 = OpConstant %float 3
     %int_12 = OpConstant %int 12
     %int_10 = OpConstant %int 10
      %int_6 = OpConstant %int 6
      %int_2 = OpConstant %int 2
       %main = OpFunction %void None %3
          %5 = OpLabel
       %five = OpVariable %_ptr_Function_float Function
        %six = OpVariable %_ptr_Function_float Function
          %b = OpVariable %_ptr_Function_int Function
          %d = OpVariable %_ptr_Function_int Function
          %a = OpVariable %_ptr_Function_int Function
          %c = OpVariable %_ptr_Function_int Function
          %o = OpVariable %_ptr_Function_int Function
      %local = OpVariable %_ptr_Function_int Function
        %bad = OpVariable %_ptr_Function_int Function
          %e = OpVariable %_ptr_Function_int Function
          %f = OpVariable %_ptr_Function_int Function
               OpStore %five %float_5
               OpStore %six %float_6
         %12 = OpLoad %float %five
         %15 = OpFOrdLessThan %bool %12 %float_10
               OpSelectionMerge %17 None
               OpBranchConditional %15 %16 %42
         %16 = OpLabel
               OpStore %b %int_4
         %22 = OpLoad %float %five
         %24 = OpFOrdGreaterThan %bool %22 %float_20
               OpSelectionMerge %26 None
               OpBranchConditional %24 %25 %37
         %25 = OpLabel
         %29 = OpLoad %int %a
               OpStore %d %29
               OpStore %b %int_5
               OpStore %c %int_3
         %34 = OpLoad %int %b
         %35 = OpLoad %int %c
         %36 = OpIAdd %int %34 %35
               OpStore %o %36
               OpBranch %26
         %37 = OpLabel
               OpStore %local %int_0
               OpStore %a %int_1
               OpStore %d %int_21
               OpStore %c %int_4
               OpBranch %26
         %26 = OpLabel
               OpBranch %17
         %42 = OpLabel
         %43 = OpLoad %float %five
         %45 = OpFOrdEqual %bool %43 %float_2
               OpSelectionMerge %47 None
               OpBranchConditional %45 %46 %57
         %46 = OpLabel
         %48 = OpLoad %float %six
         %50 = OpFOrdLessThan %bool %48 %float_3
               OpSelectionMerge %52 None
               OpBranchConditional %50 %51 %52
         %51 = OpLabel
               OpStore %a %int_3
               OpBranch %52
         %52 = OpLabel
         %54 = OpLoad %int %a
         %55 = OpIMul %int %int_5 %54
               OpStore %bad %55
               OpStore %d %int_12
               OpBranch %47
         %57 = OpLabel
               OpStore %d %int_10
               OpBranch %47
         %47 = OpLabel
               OpStore %b %int_6
               OpBranch %17
         %17 = OpLabel
         %61 = OpLoad %int %a
         %62 = OpLoad %int %b
         %63 = OpIAdd %int %61 %62
               OpStore %e %63
         %65 = OpLoad %int %c
         %67 = OpIMul %int %65 %int_2
         %68 = OpLoad %int %d
         %69 = OpIAdd %int %67 %68
               OpStore %f %69
               OpReturn
               OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 29 },
    BadLocalLoad { .id = 54 },
    BadLocalLoad { .id = 61 },
    BadLocalLoad { .id = 65 }
  ));
}

TEST_F(UninitializedVariablesTest, AllTypesTest) {
// struct Bar {
//     vec4 aa;
//     vec4 bb;
//     uint number;
// };
//
// struct Foo {
//     vec2 foo_vecs[2];
//     Bar bar;
// };
//
// void main() {
//
//     Bar bar;
//     bar.aa = vec4(0);
//     uint n = bar.number; // bar.number bad
//
//     mat2 matrix;
//     vec2 vec_array[2];
//     float gg = vec_array[0].y; // vec_array[0] bad
//     float bad;
//     switch (n) {
//         case 1:
//             matrix[0][0] = 1;
//             matrix[0][1] = 2;
//             n = 4;
//             break;
//         case 3: n = 5; 
//         default: 
//             matrix[0] = vec2(5);
//             bad = 5.5; 
//             break;
//     }
//     vec2 col_good = matrix[0];
//     vec2 col_bad = matrix[1]; // matrix[1] bad
//     for (int i = 0; i < 2; i++) {
//         matrix[1][0] = dot(col_good, col_good);
//         vec_array[i] = vec2(i);
//         for (int j = 0; j < 3; j++) {
//             bad = j * bad; // bad bad
//             matrix[1][1] = 4;
//         }
//     }
//     float array[4] = {matrix[0][1], matrix[0][0], matrix[1][0], matrix[1][1]};
//     float unknown = vec_array[0].y;
//     Foo foo = Foo(vec_array, bar); // bar bad
//     bar.bb = foo.bar.aa;
//     do {
//         bar.number = 1;
//     } while(1 == 2);
//     Foo foo2 = Foo(vec_array, bar);
//     uint n2 = foo2.bar.number;
// }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
        %Bar = OpTypeStruct %v4float %v4float %uint
%_ptr_Function_Bar = OpTypePointer Function %Bar
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
         %15 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Function_uint = OpTypePointer Function %uint
      %int_2 = OpConstant %int 2
%_ptr_Function_float = OpTypePointer Function %float
    %v2float = OpTypeVector %float 2
     %uint_2 = OpConstant %uint 2
%_arr_v2float_uint_2 = OpTypeArray %v2float %uint_2
%_ptr_Function__arr_v2float_uint_2 = OpTypePointer Function %_arr_v2float_uint_2
     %uint_1 = OpConstant %uint 1
%mat2v2float = OpTypeMatrix %v2float 2
%_ptr_Function_mat2v2float = OpTypePointer Function %mat2v2float
    %float_1 = OpConstant %float 1
     %uint_0 = OpConstant %uint 0
    %float_2 = OpConstant %float 2
     %uint_4 = OpConstant %uint 4
     %uint_5 = OpConstant %uint 5
    %float_5 = OpConstant %float 5
         %50 = OpConstantComposite %v2float %float_5 %float_5
%_ptr_Function_v2float = OpTypePointer Function %v2float
  %float_5_5 = OpConstant %float 5.5
      %int_1 = OpConstant %int 1
%_ptr_Function_int = OpTypePointer Function %int
       %bool = OpTypeBool
      %int_3 = OpConstant %int 3
    %float_4 = OpConstant %float 4
%_arr_float_uint_4 = OpTypeArray %float %uint_4
%_ptr_Function__arr_float_uint_4 = OpTypePointer Function %_arr_float_uint_4
        %Foo = OpTypeStruct %_arr_v2float_uint_2 %Bar
%_ptr_Function_Foo = OpTypePointer Function %Foo
      %false = OpConstantFalse %bool
       %main = OpFunction %void None %3
          %5 = OpLabel
        %bar = OpVariable %_ptr_Function_Bar Function
          %n = OpVariable %_ptr_Function_uint Function
         %gg = OpVariable %_ptr_Function_float Function
  %vec_array = OpVariable %_ptr_Function__arr_v2float_uint_2 Function
     %matrix = OpVariable %_ptr_Function_mat2v2float Function
        %bad = OpVariable %_ptr_Function_float Function
   %col_good = OpVariable %_ptr_Function_v2float Function
    %col_bad = OpVariable %_ptr_Function_v2float Function
          %i = OpVariable %_ptr_Function_int Function
          %j = OpVariable %_ptr_Function_int Function
      %array = OpVariable %_ptr_Function__arr_float_uint_4 Function
    %unknown = OpVariable %_ptr_Function_float Function
        %foo = OpVariable %_ptr_Function_Foo Function
       %foo2 = OpVariable %_ptr_Function_Foo Function
         %n2 = OpVariable %_ptr_Function_uint Function
         %17 = OpAccessChain %_ptr_Function_v4float %bar %int_0
               OpStore %17 %15
         %21 = OpAccessChain %_ptr_Function_uint %bar %int_2
         %22 = OpLoad %uint %21
               OpStore %n %22
         %31 = OpAccessChain %_ptr_Function_float %vec_array %int_0 %uint_1
         %32 = OpLoad %float %31
               OpStore %gg %32
         %33 = OpLoad %uint %n
               OpSelectionMerge %37 None
               OpSwitch %33 %36 1 %34 3 %35
         %36 = OpLabel
         %52 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
               OpStore %52 %50
               OpStore %bad %float_5_5
               OpBranch %37
         %34 = OpLabel
         %43 = OpAccessChain %_ptr_Function_float %matrix %int_0 %uint_0
               OpStore %43 %float_1
         %45 = OpAccessChain %_ptr_Function_float %matrix %int_0 %uint_1
               OpStore %45 %float_2
               OpStore %n %uint_4
               OpBranch %37
         %35 = OpLabel
               OpStore %n %uint_5
               OpBranch %36
         %37 = OpLabel
         %58 = OpAccessChain %_ptr_Function_v2float %matrix %int_0
         %59 = OpLoad %v2float %58
               OpStore %col_good %59
         %62 = OpAccessChain %_ptr_Function_v2float %matrix %int_1
         %63 = OpLoad %v2float %62
               OpStore %col_bad %63
               OpStore %i %int_0
               OpBranch %66
         %66 = OpLabel
               OpLoopMerge %68 %69 None
               OpBranch %70
         %70 = OpLabel
         %71 = OpLoad %int %i
         %73 = OpSLessThan %bool %71 %int_2
               OpBranchConditional %73 %67 %68
         %67 = OpLabel
         %74 = OpLoad %v2float %col_good
         %75 = OpLoad %v2float %col_good
         %76 = OpDot %float %74 %75
         %77 = OpAccessChain %_ptr_Function_float %matrix %int_1 %uint_0
               OpStore %77 %76
         %78 = OpLoad %int %i
         %79 = OpLoad %int %i
         %80 = OpConvertSToF %float %79
         %81 = OpCompositeConstruct %v2float %80 %80
         %82 = OpAccessChain %_ptr_Function_v2float %vec_array %78
               OpStore %82 %81
               OpStore %j %int_0
               OpBranch %84
         %84 = OpLabel
               OpLoopMerge %86 %87 None
               OpBranch %88
         %88 = OpLabel
         %89 = OpLoad %int %j
         %91 = OpSLessThan %bool %89 %int_3
               OpBranchConditional %91 %85 %86
         %85 = OpLabel
         %92 = OpLoad %int %j
         %93 = OpConvertSToF %float %92
         %94 = OpLoad %float %bad
         %95 = OpFMul %float %93 %94
               OpStore %bad %95
         %97 = OpAccessChain %_ptr_Function_float %matrix %int_1 %uint_1
               OpStore %97 %float_4
               OpBranch %87
         %87 = OpLabel
         %98 = OpLoad %int %j
         %99 = OpIAdd %int %98 %int_1
               OpStore %j %99
               OpBranch %84
         %86 = OpLabel
               OpBranch %69
         %69 = OpLabel
        %100 = OpLoad %int %i
        %101 = OpIAdd %int %100 %int_1
               OpStore %i %101
               OpBranch %66
         %68 = OpLabel
        %105 = OpAccessChain %_ptr_Function_float %matrix %int_0 %uint_1
        %106 = OpLoad %float %105
        %107 = OpAccessChain %_ptr_Function_float %matrix %int_0 %uint_0
        %108 = OpLoad %float %107
        %109 = OpAccessChain %_ptr_Function_float %matrix %int_1 %uint_0
        %110 = OpLoad %float %109
        %111 = OpAccessChain %_ptr_Function_float %matrix %int_1 %uint_1
        %112 = OpLoad %float %111
        %113 = OpCompositeConstruct %_arr_float_uint_4 %106 %108 %110 %112
               OpStore %array %113
        %115 = OpAccessChain %_ptr_Function_float %vec_array %int_0 %uint_1
        %116 = OpLoad %float %115
               OpStore %unknown %116
        %120 = OpLoad %_arr_v2float_uint_2 %vec_array
        %121 = OpLoad %Bar %bar
        %122 = OpCompositeConstruct %Foo %120 %121
               OpStore %foo %122
        %123 = OpAccessChain %_ptr_Function_v4float %foo %int_1 %int_0
        %124 = OpLoad %v4float %123
        %125 = OpAccessChain %_ptr_Function_v4float %bar %int_1
               OpStore %125 %124
               OpBranch %126
        %126 = OpLabel
               OpLoopMerge %128 %129 None
               OpBranch %127
        %127 = OpLabel
        %130 = OpAccessChain %_ptr_Function_uint %bar %int_2
               OpStore %130 %uint_1
               OpBranch %129
        %129 = OpLabel
               OpBranchConditional %false %126 %128
        %128 = OpLabel
        %133 = OpLoad %_arr_v2float_uint_2 %vec_array
        %134 = OpLoad %Bar %bar
        %135 = OpCompositeConstruct %Foo %133 %134
               OpStore %foo2 %135
        %137 = OpAccessChain %_ptr_Function_uint %foo2 %int_1 %int_2
        %138 = OpLoad %uint %137
               OpStore %n2 %138
               OpReturn
               OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 22, .state_have = SC({SYes(), SNo(), SNo()}), .state_missing = SC({SNo(), SNo(), SYes()}) },
    BadLocalLoad { .id = 32, .state_have = SNo(), .state_missing = SC({SC({SNo(), SYes()}), SNo()}) },
    BadLocalLoad { .id = 63, .state_have = SC({SYes(), SNo()}), .state_missing = SC({SNo(), SYes()}) },
    BadLocalLoad { .id = 94 },
    BadLocalLoad { .id = 121, .state_have = SC({SYes(), SNo(), SNo()}), .state_missing = SC({SNo(), SYes(), SYes()}) }
  ));
}

TEST_F(UninitializedVariablesTest, ArrayDynamicTest) {
// layout(constant_id = 1) const uint SPECCONST = 3;
//
// void main() {
//     float arr[SPECCONST];
//     float a = arr[0]; // bad
//     for (int i = 0; i < 2; i++) {
//         float dd = arr[i]; // bad (first iteration)
//         arr[i] = 1.0;
//     }
//     float b = arr[2];
// }
  ASSERT_NO_FATAL_FAILURE(Build(Preamble() + R"(
               OpDecorate %SPECCONST SpecId 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
       %uint = OpTypeInt 32 0
  %SPECCONST = OpSpecConstant %uint 3
%_arr_float_SPECCONST = OpTypeArray %float %SPECCONST
%_ptr_Function__arr_float_SPECCONST = OpTypePointer Function %_arr_float_SPECCONST
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Function_int = OpTypePointer Function %int
      %int_2 = OpConstant %int 2
       %bool = OpTypeBool
    %float_1 = OpConstant %float 1
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_float Function
        %arr = OpVariable %_ptr_Function__arr_float_SPECCONST Function
          %i = OpVariable %_ptr_Function_int Function
         %dd = OpVariable %_ptr_Function_float Function
          %b = OpVariable %_ptr_Function_float Function
         %16 = OpAccessChain %_ptr_Function_float %arr %int_0
         %17 = OpLoad %float %16
               OpStore %a %17
               OpStore %i %int_0
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %int %i
         %28 = OpSLessThan %bool %25 %int_2
               OpBranchConditional %28 %21 %22
         %21 = OpLabel
         %30 = OpLoad %int %i
         %31 = OpAccessChain %_ptr_Function_float %arr %30
         %32 = OpLoad %float %31
               OpStore %dd %32
         %33 = OpLoad %int %i
         %35 = OpAccessChain %_ptr_Function_float %arr %33
               OpStore %35 %float_1
               OpBranch %23
         %23 = OpLabel
         %36 = OpLoad %int %i
         %38 = OpIAdd %int %36 %int_1
               OpStore %i %38
               OpBranch %20
         %22 = OpLabel
         %40 = OpAccessChain %_ptr_Function_float %arr %int_2
         %41 = OpLoad %float %40
               OpStore %b %41
               OpReturn
               OpFunctionEnd
  )"));
  EXPECT_THAT(bad_loads_, testing::UnorderedElementsAre(
    BadLocalLoad { .id = 17, .state_have = SNo(), .state_missing = SUnk() },
    BadLocalLoad { .id = 32, .state_have = SNo(), .state_missing = SUnk() }
  ));
}

// =====================
// GLOBAL VARIABLE TESTS
// =====================

TEST_F(UninitializedGlobalsTest, SimpleTest) {
// int g;
//
// int other() {
//     for (int i = 0; i < 10; i++) {
//         if (i == 12) {
//             return 5;
//         }
//     }
//     return 2 * g;
// }
//
// void main() {
//     float a = float(1);
//     int b = other();
//     g = 5;
//     int c = other();
// }
  ASSERT_NO_FATAL_FAILURE(Build(R"(
OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %222 "main" %g
               OpExecutionMode %222 OriginUpperLeft
               OpSource GLSL 450
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
          %7 = OpTypeFunction %int
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
     %int_10 = OpConstant %int 10
       %bool = OpTypeBool
     %int_12 = OpConstant %int 12
      %int_5 = OpConstant %int 5
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
%_ptr_Private_int = OpTypePointer Private %int
          %g = OpVariable %_ptr_Private_int Private
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %float_1 = OpConstant %float 1
       %222  = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_float Function
          %b = OpVariable %_ptr_Function_int Function
          %c = OpVariable %_ptr_Function_int Function
               OpStore %a %float_1
       %44 = OpFunctionCall %int %other_
               OpStore %b %44
               OpStore %g %int_5
       %46 = OpFunctionCall %int %other_
               OpStore %c %46
               OpReturn
               OpFunctionEnd

     %other_ = OpFunction %int None %7
          %9 = OpLabel
          %i = OpVariable %_ptr_Function_int Function
               OpStore %i %int_0
               OpBranch %13
         %13 = OpLabel
               OpLoopMerge %15 %16 None
               OpBranch %17
         %17 = OpLabel
         %18 = OpLoad %int %i
         %21 = OpSLessThan %bool %18 %int_10
               OpBranchConditional %21 %14 %15
         %14 = OpLabel
         %22 = OpLoad %int %i
         %24 = OpIEqual %bool %22 %int_12
               OpSelectionMerge %26 None
               OpBranchConditional %24 %25 %26
         %25 = OpLabel
               OpReturnValue %int_5
         %26 = OpLabel
               OpBranch %16
         %16 = OpLabel
         %29 = OpLoad %int %i
         %31 = OpIAdd %int %29 %int_1
               OpStore %i %31
               OpBranch %13
         %15 = OpLabel
         %35 = OpLoad %int %g
         %36 = OpIMul %int %int_2 %35
               OpReturnValue %36
               OpFunctionEnd
  )"));
  EXPECT_EQ(1, ep_results.size());
  EXPECT_EQ(1, ep_results.at(222).size());
  EXPECT_THAT(ep_results.at(222),
              testing::UnorderedElementsAre(
                  BadGlobalLoad{.load_result_id = 35,
                                .state_have = SNo(),
                                .state_missing = SYes(),
                                .call_trace = std::vector<uint32_t>({44})}));
}

TEST_F(UninitializedGlobalsTest, TransitiveTest) {
// struct Foo {
//     vec2 v;
//     float f;
// };
//
// Foo GLOBAL;
//
// void iSetGLOBAL() {
//     GLOBAL.f = 5.0;
// }
//
// int other() {
//     int a = 2;
//     iSetGLOBAL();
//     return a;
// }
//
// float needGLOBAL() {
//     return GLOBAL.f * 2.0;
// }
//
// float needGLOBAL2() {
//     return GLOBAL.v[0];
// }
//
// float transitivelyNeedGLOBAL(float arg) {
//     if (arg < 0.0) {
//         return needGLOBAL();
//     }
//     return needGLOBAL2();
// }
//
// void main() {
//     int a = 2;
//     if (1 == 2) {
//         needGLOBAL();
//     }
//     transitivelyNeedGLOBAL(4.0);
//     float aaa = GLOBAL.f * 2.0;
//     int c = a + other();
//     needGLOBAL();
//     float bbb = GLOBAL.v[1];
// }
  ASSERT_NO_FATAL_FAILURE(Build(R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main"
               OpExecutionMode %4 OriginUpperLeft
               OpSource GLSL 450
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %8 = OpTypeInt 32 1
          %9 = OpTypeFunction %8
         %12 = OpTypeFloat 32
         %13 = OpTypeFunction %12
         %18 = OpTypePointer Function %12
         %19 = OpTypeFunction %12 %18
         %23 = OpTypeVector %12 2
         %24 = OpTypeStruct %23 %12
         %25 = OpTypePointer Private %24
         %26 = OpVariable %25 Private
         %27 = OpConstant %8 1
         %28 = OpConstant %12 5
         %29 = OpTypePointer Private %12
         %31 = OpTypePointer Function %8
         %33 = OpConstant %8 2
         %40 = OpConstant %12 2
         %44 = OpConstant %8 0
         %45 = OpTypeInt 32 0
         %46 = OpConstant %45 0
         %52 = OpConstant %12 0
         %53 = OpTypeBool
         %63 = OpConstantFalse %53
         %67 = OpConstant %12 4
         %80 = OpConstant %45 1
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %62 = OpVariable %31 Function
         %68 = OpVariable %18 Function
         %70 = OpVariable %18 Function
         %74 = OpVariable %31 Function
         %79 = OpVariable %18 Function
               OpStore %62 %33
               OpSelectionMerge %65 None
               OpBranchConditional %63 %64 %65
         %64 = OpLabel
       %66 = OpFunctionCall %12 %14
               OpBranch %65
         %65 = OpLabel
               OpStore %68 %67
       %69 = OpFunctionCall %12 %21 %68
         %71 = OpAccessChain %29 %26 %27
         %72 = OpLoad %12 %71
         %73 = OpFMul %12 %72 %40
               OpStore %70 %73
         %75 = OpLoad %8 %62
       %76 = OpFunctionCall %8 %10
         %77 = OpIAdd %8 %75 %76
               OpStore %74 %77
       %78 = OpFunctionCall %12 %14
         %81 = OpAccessChain %29 %26 %44 %80
         %82 = OpLoad %12 %81
               OpStore %79 %82
               OpReturn
               OpFunctionEnd


          %6 = OpFunction %2 None %3
          %7 = OpLabel
         %30 = OpAccessChain %29 %26 %27
               OpStore %30 %28
               OpReturn
               OpFunctionEnd

         %10 = OpFunction %8 None %9
         %11 = OpLabel
         %32 = OpVariable %31 Function
               OpStore %32 %33
       %34 = OpFunctionCall %2 %6
         %35 = OpLoad %8 %32
               OpReturnValue %35
               OpFunctionEnd

         %14 = OpFunction %12 None %13
         %15 = OpLabel
         %38 = OpAccessChain %29 %26 %27
         %39 = OpLoad %12 %38
         %41 = OpFMul %12 %39 %40
               OpReturnValue %41
               OpFunctionEnd

         %16 = OpFunction %12 None %13
         %17 = OpLabel
         %47 = OpAccessChain %29 %26 %44 %46
         %48 = OpLoad %12 %47
               OpReturnValue %48
               OpFunctionEnd

         %21 = OpFunction %12 None %19
         %20 = OpFunctionParameter %18
         %22 = OpLabel
         %51 = OpLoad %12 %20
         %54 = OpFOrdLessThan %53 %51 %52
               OpSelectionMerge %56 None
               OpBranchConditional %54 %55 %56
         %55 = OpLabel
       %57 = OpFunctionCall %12 %14
               OpReturnValue %57
         %56 = OpLabel
       %59 = OpFunctionCall %12 %16
               OpReturnValue %59
               OpFunctionEnd
  )"));
  EXPECT_EQ(1, ep_results.size());
  EXPECT_EQ(5, ep_results.at(4).size());
  EXPECT_THAT(
      ep_results.at(4),
      testing::UnorderedElementsAre(
          BadGlobalLoad{.load_result_id = 39,
                        .state_have = SNo(),
                        .state_missing = SC({SNo(), SYes()}),
                        .call_trace = std::vector<uint32_t>({66})},
          BadGlobalLoad{.load_result_id = 39,
                        .state_have = SNo(),
                        .state_missing = SC({SNo(), SYes()}),
                        .call_trace = std::vector<uint32_t>({57, 69})},
          BadGlobalLoad{.load_result_id = 72,
                        .state_have = SNo(),
                        .state_missing = SC({SNo(), SYes()}),
                        .call_trace = std::vector<uint32_t>()},
          BadGlobalLoad{.load_result_id = 48,
                        .state_have = SNo(),
                        .state_missing = SC({SC({SYes(), SNo()}), SNo()}),
                        .call_trace = std::vector<uint32_t>({59, 69})},
          BadGlobalLoad{.load_result_id = 82,
                        .state_have = SC({SNo(), SYes()}),
                        .state_missing = SC({SC({SNo(), SYes()}), SNo()}),
                        .call_trace = std::vector<uint32_t>()}));
}


} // namespace lint
} // namespace spvtools
