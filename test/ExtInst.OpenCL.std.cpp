// Copyright (c) 2015 The Khronos Group Inc.
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

#include "UnitSPIRV.h"

#include <gmock/gmock.h>
#include "TestFixture.h"

namespace {

using spvtest::Concatenate;
using spvtest::MakeInstruction;
using spvtest::MakeVector;
using testing::Eq;

struct InstructionCase {
  uint32_t opcode;
  std::string name;
  std::string operands;
  std::vector<uint32_t> expected_operands;
};

using ExtInstOpenCLStdRoundTripTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<InstructionCase>>;

TEST_P(ExtInstOpenCLStdRoundTripTest, ParameterizedExtInst) {
  // This example should not validate.
  const std::string input =
      "%1 = OpExtInstImport \"OpenCL.std\"\n"
      "%3 = OpExtInst %2 %1 " +
      GetParam().name + " " + GetParam().operands + "\n";
  // First make sure it assembles correctly.
  EXPECT_THAT(
      CompiledInstructions(input),
      Eq(Concatenate(
          {MakeInstruction(spv::OpExtInstImport, {1}, MakeVector("OpenCL.std")),
           MakeInstruction(spv::OpExtInst, {2, 3, 1, GetParam().opcode},
                           GetParam().expected_operands)})))
      << input;
  // Now check the round trip through the disassembler.
  EXPECT_THAT(EncodeAndDecodeSuccessfully(input), input) << input;
}

#define CASE1(Enum, Name)                                      \
  {                                                            \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4", { 4 } \
  }
#define CASE2(Enum, Name)                                            \
  {                                                                  \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5", { 4, 5 } \
  }
#define CASE3(Enum, Name)                                                  \
  {                                                                        \
    uint32_t(OpenCLLIB::Entrypoints::Enum), #Name, "%4 %5 %6", { 4, 5, 6 } \
  }

// clang-format off
// OpenCL.std: 2.1 Math extended instructions
INSTANTIATE_TEST_CASE_P(
    OpenCLMath, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        // We are only testing the correctness of encoding and decoding here.
        // Semantic correctness should be the responsibility of validator.
        CASE1(Acos, acos), // enum value 0
        CASE1(Acosh, acosh),
        CASE1(Acospi, acospi),
        CASE1(Asin, asin),
        CASE1(Asinh, asinh),
        CASE1(Asinh, asinh),
        CASE1(Asinpi, asinpi),
        CASE1(Atan, atan),
        CASE2(Atan2, atan2),
        CASE1(Atanh, atanh),
        CASE1(Atanpi, atanpi),
        CASE2(Atan2pi, atan2pi),
        CASE1(Cbrt, cbrt),
        CASE1(Ceil, ceil),
        CASE1(Ceil, ceil),
        CASE2(Copysign, copysign),
        CASE1(Cos, cos),
        CASE1(Cosh, cosh),
        CASE1(Cospi, cospi),
        CASE1(Erfc, erfc),
        CASE1(Erf, erf),
        CASE1(Exp, exp),
        CASE1(Exp2, exp2),
        CASE1(Exp10, exp10),
        CASE1(Expm1, expm1),
        CASE1(Fabs, fabs),
        CASE2(Fdim, fdim),
        CASE1(Floor, floor),
        CASE3(Fma, fma),
        CASE2(Fmax, fmax),
        CASE2(Fmin, fmin),
        CASE2(Fmod, fmod),
        CASE2(Fract, fract),
        CASE2(Frexp, frexp),
        CASE2(Hypot, hypot),
        CASE1(Ilogb, ilogb),
        CASE2(Ldexp, ldexp),
        CASE1(Lgamma, lgamma),
        CASE2(Lgamma_r, lgamma_r),
        CASE1(Log, log),
        CASE1(Log2, log2),
        CASE1(Log10, log10),
        CASE1(Log1p, log1p),
        CASE3(Mad, mad),
        CASE2(Maxmag, maxmag),
        CASE2(Minmag, minmag),
        CASE2(Modf, modf),
        CASE1(Nan, nan),
        CASE2(Nextafter, nextafter),
        CASE3(Pow, pow),
        CASE1(Pown, pown),
        CASE2(Powr, powr),
        CASE2(Remainder, remainder),
        CASE3(Remquo, remquo),
        CASE1(Rint, rint),
        CASE2(Rootn, rootn),
        CASE1(Round, round),
        CASE1(Rsqrt, rsqrt),
        CASE1(Sin, sin),
        CASE2(Sincos, sincos),
        CASE1(Sinh, sinh),
        CASE1(Sinpi, sinpi),
        CASE1(Sqrt, sqrt),
        CASE1(Tan, tan),
        CASE1(Tanh, tanh),
        CASE1(Tanpi, tanpi),
        CASE1(Tgamma, tgamma),
        CASE1(Trunc, trunc),
        CASE1(Half_cos, half_cos),
        CASE2(Half_divide, half_divide),
        CASE1(Half_exp, half_exp),
        CASE1(Half_exp2, half_exp2),
        CASE1(Half_exp10, half_exp10),
        CASE1(Half_log, half_log),
        CASE1(Half_log2, half_log2),
        CASE1(Half_log10, half_log10),
        CASE2(Half_powr, half_powr),
        CASE1(Half_recip, half_recip),
        CASE1(Half_rsqrt, half_rsqrt),
        CASE1(Half_sin, half_sin),
        CASE1(Half_sqrt, half_sqrt),
        CASE1(Half_tan, half_tan),
        CASE1(Native_cos, native_cos),
        CASE2(Native_divide, native_divide),
        CASE1(Native_exp, native_exp),
        CASE1(Native_exp2, native_exp2),
        CASE1(Native_exp10, native_exp10),
        CASE1(Native_log, native_log),
        CASE1(Native_log10, native_log10),
        CASE2(Native_powr, native_powr),
        CASE1(Native_recip, native_recip),
        CASE1(Native_rsqrt, native_rsqrt),
        CASE1(Native_sin, native_sin),
        CASE1(Native_sqrt, native_sqrt),
        CASE1(Native_tan, native_tan), // enum value 94
    })));

// TODO(dneto): OpenCL.std: 2.1 Integer instructions

// OpenCL.std: 2.3 Common instrucitons
INSTANTIATE_TEST_CASE_P(
    OpenCLCommon, ExtInstOpenCLStdRoundTripTest,
    ::testing::ValuesIn(std::vector<InstructionCase>({
        CASE3(FClamp, fclamp), // enum value 95
        CASE1(Degrees, degrees),
        CASE2(FMax_common, fmax_common),
        CASE2(FMin_common, fmin_common),
        CASE3(Mix, mix),
        CASE1(Radians, radians),
        CASE2(Step, step),
        CASE3(Smoothstep, smoothstep),
        CASE1(Sign, sign), // enum value 103
    })));

// TODO(dneto): OpenCL.std: 2.4 Geometric instructions
// TODO(dneto): OpenCL.std: 2.5 Relational instructions
// TODO(dneto): OpenCL.std: 2.6 Vector data load and store instructions
// TODO(dneto): OpenCL.std: 2.7 Miscellaneous vector instructions
// TODO(dneto): OpenCL.std: 2.8 Miscellaneous instructions
// TODO(dneto): OpenCL.std: 2.9.1 Image encoding
// TODO(dneto): OpenCL.std: 2.9.2 Sampler encoding
// TODO(dneto): OpenCL.std: 2.9.3 Image read
// TODO(dneto): OpenCL.std: 2.9.4 Image write

// clang-format on

#undef CASE1
#undef CASE2
#undef CASE3

}  // anonymous namespace
