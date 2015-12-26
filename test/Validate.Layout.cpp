// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
// KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
// SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
//    https://www.khronos.org/registry/
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

// Validation tests for Logical Layout

#include "gmock/gmock.h"
#include "UnitSPIRV.h"
#include "ValidateFixtures.h"

#include <functional>
#include <sstream>
#include <string>
#include <utility>

using std::function;
using std::ostream;
using std::ostream_iterator;
using std::pair;
using std::stringstream;
using std::string;
using std::tie;
using std::tuple;
using std::vector;

using ::testing::HasSubstr;

using pred_type = function<spv_result_t(int)>;
using ValidateLayout =
    spvtest::ValidateBase<tuple<int, tuple<string, pred_type, pred_type>>,
                          SPV_VALIDATE_LAYOUT_BIT>;

// TODO(umar): probably should move this to a central location
ostream& operator<<(ostream& os, const spv_result_t& err) {
  switch (err) {
    case SPV_SUCCESS:
      os << "SPV_SUCCESS";
      break;
    case SPV_UNSUPPORTED:
      os << "SPV_UNSUPPORTED";
      break;
    case SPV_END_OF_STREAM:
      os << "SPV_END_OF_STREAM";
      break;
    case SPV_WARNING:
      os << "SPV_WARNING";
      break;
    case SPV_FAILED_MATCH:
      os << "SPV_FAILED_MATCH";
      break;
    case SPV_REQUESTED_TERMINATION:
      os << "SPV_REQUESTED_TERMINATION";
      break;
    case SPV_ERROR_INTERNAL:
      os << "SPV_ERROR_INTERNAL";
      break;
    case SPV_ERROR_OUT_OF_MEMORY:
      os << "SPV_ERROR_OUT_OF_MEMORY";
      break;
    case SPV_ERROR_INVALID_POINTER:
      os << "SPV_ERROR_INVALID_POINTER";
      break;
    case SPV_ERROR_INVALID_BINARY:
      os << "SPV_ERROR_INVALID_BINARY";
      break;
    case SPV_ERROR_INVALID_TEXT:
      os << "SPV_ERROR_INVALID_TEXT";
      break;
    case SPV_ERROR_INVALID_TABLE:
      os << "SPV_ERROR_INVALID_TABLE";
      break;
    case SPV_ERROR_INVALID_VALUE:
      os << "SPV_ERROR_INVALID_VALUE";
      break;
    case SPV_ERROR_INVALID_DIAGNOSTIC:
      os << "SPV_ERROR_INVALID_DIAGNOSTIC";
      break;
    case SPV_ERROR_INVALID_LOOKUP:
      os << "SPV_ERROR_INVALID_LOOKUP";
      break;
    case SPV_ERROR_INVALID_ID:
      os << "SPV_ERROR_INVALID_ID";
      break;
    case SPV_ERROR_INVALID_CFG:
      os << "SPV_ERROR_INVALID_CFG";
      break;
    case SPV_ERROR_INVALID_LAYOUT:
      os << "SPV_ERROR_INVALID_LAYOUT";
      break;
    default:
      os << "Unknown Error";
  }
  return os;
}
namespace {

// returns true if order is equal to VAL
template <int VAL, spv_result_t RET = SPV_ERROR_INVALID_LAYOUT>
spv_result_t Equals(int order) {
  return order == VAL ? SPV_SUCCESS : RET;
}

// returns true if order is between MIN and MAX(inclusive)
template <int MIN, int MAX, spv_result_t RET = SPV_ERROR_INVALID_LAYOUT>
struct Range {
  Range(bool inverse = false) : inverse_(inverse) {}
  spv_result_t operator()(int order) {
    return (inverse_ ^ (order >= MIN && order <= MAX)) ? SPV_SUCCESS : RET;
  }

 private:
  bool inverse_;
};

template <typename... T>
spv_result_t InvalidSet(int order) {
  for (spv_result_t val : {T(true)(order)...})
    if (val != SPV_SUCCESS) return val;
  return SPV_SUCCESS;
}

// SPIRV source used to test the logical layout
const vector<string>& getInstructions() {
  // clang-format off
  static const vector<string> instructions = {
    "OpCapability Matrix",
    "OpExtension \"TestExtension\"",
    "%inst = OpExtInstImport \"GLSL.std.450\"",
    "OpMemoryModel Logical GLSL450",
    "OpEntryPoint GLCompute %func \"\"",
    "OpExecutionMode %func LocalSize 1 1 1",
    "%str = OpString \"Test String\"",
    "OpSource GLSL 450 %str \"uniform vec3 var = vec3(4.0);\"",
    "OpSourceContinued \"void main(){return;}\"",
    "OpSourceExtension \"Test extension\"",
    "OpName %id \"MyID\"",
    "OpMemberName %struct 1 \"my_member\"",
    "OpDecorate %dgrp RowMajor",
    "OpMemberDecorate %struct 1 RowMajor",
    "%dgrp   = OpDecorationGroup",
    "OpGroupDecorate %dgrp %mat33 %mat44",
    "%intt     = OpTypeInt 32 1",
    "%floatt   = OpTypeFloat 32",
    "%voidt    = OpTypeVoid",
    "%boolt    = OpTypeBool",
    "%vec4     = OpTypeVector %intt 4",
    "%vec3     = OpTypeVector %intt 3",
    "%mat33    = OpTypeMatrix %vec3 3",
    "%mat44    = OpTypeMatrix %vec4 4",
    "%struct   = OpTypeStruct %intt %mat33",
    "%vfunct   = OpTypeFunction %voidt",
    "%viifunct = OpTypeFunction %voidt %intt %intt",
    "%one      = OpConstant %intt 1",
    // TODO(umar): OpConstant fails because the type is not defined
    // TODO(umar): OpGroupMemberDecorate
    "OpLine %str 3 4",
    "%func     = OpFunction %voidt None %vfunct",
    "OpFunctionEnd",
    "%func2    = OpFunction %voidt None %viifunct",
    "%funcp1   = OpFunctionParameter %intt",
    "%funcp2   = OpFunctionParameter %intt",
    "%fLabel   = OpLabel",
    "            OpNop",
    "            OpReturn",
    "OpFunctionEnd"
  };
  return instructions;
}

static const int kRangeEnd = 1000;
pred_type All = Range<0, kRangeEnd>();

INSTANTIATE_TEST_CASE_P(InstructionsOrder,
    ValidateLayout,
    ::testing::Combine(::testing::Range((int)0, (int)getInstructions().size()),
    //                                   | Instruction              | Line(s) valid     | Lines to compile
    ::testing::Values( make_tuple(string("OpCapability")            , Equals<0>         , All)
                     , make_tuple(string("OpExtension")             , Equals<1>         , All)
                     , make_tuple(string("OpExtInstImport")         , Equals<2>         , All)
                     , make_tuple(string("OpMemoryModel")           , Equals<3>         , All)
                     , make_tuple(string("OpEntryPoint")            , Equals<4>         , All)
                     , make_tuple(string("OpExecutionMode")         , Equals<5>         , All)
                     , make_tuple(string("OpSource ")               , Range<6, 9>()     , All)
                     , make_tuple(string("OpSourceContinued ")      , Range<6, 9>()     , All)
                     , make_tuple(string("OpSourceExtension ")      , Range<6, 9>()     , All)
                     , make_tuple(string("OpString ")               , Range<6, 9>()     , All)
                     , make_tuple(string("OpName ")                 , Range<10, 11>()   , All)
                     , make_tuple(string("OpMemberName ")           , Range<10, 11>()   , All)
                     , make_tuple(string("OpDecorate ")             , Range<12, 15>()   , All)
                     , make_tuple(string("OpMemberDecorate ")       , Range<12, 15>()   , All)
                     , make_tuple(string("OpGroupDecorate ")        , Range<12, 15>()   , All)
                     , make_tuple(string("OpDecorationGroup")       , Range<12, 15>()   , All)
                     , make_tuple(string("OpTypeBool")              , Range<16, 28>()   , All)
                     , make_tuple(string("OpTypeVoid")              , Range<16, 28>()   , All)
                     , make_tuple(string("OpTypeFloat")             , Range<16, 28>()   , All)
                     , make_tuple(string("OpTypeInt")               , Range<16, 28>()   , static_cast<pred_type>(Range<0, 25>()))
                     , make_tuple(string("OpTypeVector %intt 4")    , Range<16, 28>()   , All)
                     , make_tuple(string("OpTypeMatrix %vec4 4")    , Range<16, 28>()   , All)
                     , make_tuple(string("OpTypeStruct")            , Range<16, 28>()   , All)
                     , make_tuple(string("%vfunct   = OpTypeFunction"), Range<16, 28>()   , All)
                     , make_tuple(string("OpConstant")              , Range<19, 28>()   , static_cast<pred_type>(Range<19, kRangeEnd>()))
                     , make_tuple(string("OpLabel")                 , Equals<34>        , All)
                     , make_tuple(string("OpNop")                   , Equals<35>        , All)
                     , make_tuple(string("OpReturn")                , Equals<36>        , All)
    )));
// clang-format on

// Creates a new vector which removes the string if the substr is found in the
// instructions vector and reinserts it in the location specified by order.
// NOTE: This will not work correctly if there are two instances of substr in
// instructions
vector<string> GenerateCode(string substr, int order) {
  vector<string> code(getInstructions().size());
  vector<string> inst(1);
  partition_copy(begin(getInstructions()), end(getInstructions()), begin(code),
                 begin(inst), [=](const string& str) {
                   return string::npos == str.find(substr);
                 });

  code.insert(begin(code) + order, inst.front());
  return code;
}

// This test will check the logical layout of a binary by removing each
// instruction in the pair of the INSTANTIATE_TEST_CASE_P call and moving it in
// the SPIRV source formed by combining the vector "instructions"
//
// NOTE: The test will only execute with the SPV_VALIDATE_LAYOUT_BIT flag so SSA
// and other tests are not performed
TEST_P(ValidateLayout, Layout) {
  int order;
  string instruction;
  pred_type pred;
  pred_type test_pred;  // Predicate to determine if the test should be build
  tuple<string, pred_type, pred_type> testCase;

  tie(order, testCase) = GetParam();
  tie(instruction, pred, test_pred) = testCase;

  // Skip test which break the code generation
  if (test_pred(order)) return;

  vector<string> code = GenerateCode(instruction, order);

  stringstream ss;
  copy(begin(code), end(code), ostream_iterator<string>(ss, "\n"));

  // printf("code: \n%s\n", ss.str().c_str());
  CompileSuccessfully(ss.str());
  // clang-format off
  ASSERT_EQ(pred(order), ValidateInstructions())
    << "Actual: "        << ValidateInstructions()
    << "\nExpected: "    << pred(order)
    << "\nOrder: "       << order
    << "\nInstruction: " << instruction
    << "\nCode: \n"      << ss.str();
  // clang-format on
}

TEST_F(ValidateLayout, MemoryModelMissing) {
  string str = R"(
    OpCapability Matrix
    OpExtension "TestExtension"
    %inst = OpExtInstImport "GLSL.std.450"
    OpEntryPoint GLCompute %func ""
    OpExecutionMode %func LocalSize 1 1 1
    )";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
}

TEST_F(ValidateLayout, VariableFunctionStorageGood) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
          OpDecorate %var Restrict
%intt   = OpTypeInt 32 1
%voidt  = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%ptrt   = OpTypePointer Function %intt
%func   = OpFunction %voidt None %vfunct
%funcl  = OpLabel
%var    = OpVariable %ptrt Function
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_SUCCESS, ValidateInstructions());
}

TEST_F(ValidateLayout, VariableFunctionStorageBad) {
  char str[] = R"(
          OpMemoryModel Logical GLSL450
          OpDecorate %var Restrict
%intt   = OpTypeInt 32 1
%voidt  = OpTypeVoid
%vfunct = OpTypeFunction %voidt
%ptrt   = OpTypePointer Function %intt
%var    = OpVariable %ptrt Function     ; Invalid storage class for OpVariable
%func   = OpFunction %voidt None %vfunct
%funcl  = OpLabel
          OpNop
          OpReturn
          OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
}

TEST_F(ValidateLayout, FunctionDefinitionBeforeDeclarationBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpDecorate %var Restrict
%intt    = OpTypeInt 32 1
%voidt   = OpTypeVoid
%vfunct  = OpTypeFunction %voidt
%vifunct = OpTypeFunction %voidt %intt
%ptrt    = OpTypePointer Function %intt
%func    = OpFunction %voidt None %vfunct
%funcl   = OpLabel
           OpNop
           OpReturn
           OpFunctionEnd
%func2   = OpFunction %voidt None %vifunct ; must appear before definition
%func2p  = OpFunctionParameter %intt
           OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
}

// TODO(umar): Passes but gives incorrect error message. Should be fixed after
// type checking
TEST_F(ValidateLayout, LabelBeforeFunctionParameterBad) {
  char str[] = R"(
           OpMemoryModel Logical GLSL450
           OpDecorate %var Restrict
%intt    = OpTypeInt 32 1
%voidt   = OpTypeVoid
%vfunct  = OpTypeFunction %voidt
%vifunct = OpTypeFunction %voidt %intt
%ptrt    = OpTypePointer Function %intt
%func    = OpFunction %voidt None %vifunct
%funcl   = OpLabel                    ; Label appears before function parameter
%func2p  = OpFunctionParameter %intt
           OpNop
           OpReturn
           OpFunctionEnd
)";

  CompileSuccessfully(str);
  ASSERT_EQ(SPV_ERROR_INVALID_LAYOUT, ValidateInstructions());
}
// TODO(umar): Test optional instructions
}
