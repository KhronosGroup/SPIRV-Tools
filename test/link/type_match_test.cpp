// Copyright (c) 2019 The Khronos Group Inc.
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

#include "gmock/gmock.h"
#include "test/link/linker_fixture.h"

namespace spvtools {
namespace {

using TypeMatch = spvtest::LinkerTest;

// Basic types
#define PartInt(N) N " = OpTypeInt 32 0"
#define PartFloat(N) N " = OpTypeFloat 32"
#define PartOpaque(N) N " = OpTypeOpaque \"bar\""
#define PartSampler(N) N " = OpTypeSampler"
#define PartEvent(N) N " = OpTypeEvent"
#define PartDeviceEvent(N) N " = OpTypeDeviceEvent"
#define PartReserveId(N) N " = OpTypeReserveId"
#define PartQueue(N) N " = OpTypeQueue"
#define PartPipe(N) N " = OpTypePipe ReadWrite"
#define PartPipeStorage(N) N " = OpTypePipeStorage"
#define PartNamedBarrier(N) N " = OpTypeNamedBarrier"

// Compound types
#define PartVector(N, T) N " = OpTypeVector " T " 3"
#define PartMatrix(N, T) N " = OpTypeMatrix " T " 4"
#define PartImage(N, T) N " = OpTypeImage " T " 2D 0 0 0 0 Rgba32f"
#define PartSampledImage(N, T) N " = OpTypeSampledImage " T
#define PartArray(N, T) N " = OpTypeArray " T " %const"
#define PartRuntimeArray(N, T) N " = OpTypeRuntimeArray " T
#define PartStruct(N, T) N " = OpTypeStruct " T " " T
#define PartPointer(N, T) N " = OpTypePointer Workgroup " T
#define PartFunction(N, T) N " = OpTypeFunction " T " " T

#define MatchPart1(F, N) \
  "; CHECK: " Part##F("[[" #N ":%\\w+]]") "\n" Part##F("%" #N) "\n"
#define MatchPart2(F, N, T)                                                 \
  "; CHECK: " Part##F("[[" #N ":%\\w+]]", "[[" #T ":%\\w+]]") "\n" Part##F( \
      "%" #N, "%" #T) "\n"

#define MatchF(N, CODE)                                         \
  TEST_F(TypeMatch, N) {                                        \
    const std::string base =                                    \
        "OpCapability Linkage\n"                                \
        "OpCapability NamedBarrier\n"                           \
        "OpCapability PipeStorage\n"                            \
        "OpCapability Pipes\n"                                  \
        "OpCapability DeviceEnqueue\n"                          \
        "OpCapability Kernel\n"                                 \
        "OpCapability Shader\n"                                 \
        "OpCapability Addresses\n"                              \
        "OpDecorate %var LinkageAttributes \"foo\" "            \
        "{Import,Export}\n"                                     \
        "; CHECK: [[baseint:%\\w+]] = OpTypeInt 32 1\n"         \
        "%baseint = OpTypeInt 32 1\n"                           \
        "; CHECK: [[const:%\\w+]] = OpConstant [[baseint]] 3\n" \
        "%const = OpConstant %baseint 3\n" CODE                 \
        "; CHECK: OpVariable [[type]] Uniform\n"                \
        "%var = OpVariable %type Uniform";                      \
    ExpandAndMatch(base);                                       \
  }

#define Match1(T) MatchF(Type##T, MatchPart1(T, type))
#define Match2(T, A) \
  MatchF(T##OfType##A, MatchPart1(A, a) MatchPart2(T, type, a))
#define Match3(T, A, B)   \
  MatchF(T##Of##A##Of##B, \
         MatchPart1(B, b) MatchPart2(A, a, b) MatchPart2(T, type, a))

// Basic types
Match1(Int);
Match1(Float);
Match1(Opaque);
Match1(Sampler);
Match1(Event);
Match1(DeviceEvent);
Match1(ReserveId);
Match1(Queue);
Match1(Pipe);
Match1(PipeStorage);
Match1(NamedBarrier);

// Simpler (restricted) compound types
Match2(Vector, Float);
Match3(Matrix, Vector, Float);
Match2(Image, Float);

// Unrestricted compound types
// The following skip Array as it causes issues
#define MatchCompounds1(A) \
  Match2(RuntimeArray, A); \
  Match2(Struct, A);       \
  Match2(Pointer, A);      \
  Match2(Function, A);     \
// Match2(Array, A);  // Disabled as it fails currently
#define MatchCompounds2(A, B) \
  Match3(RuntimeArray, A, B); \
  Match3(Struct, A, B);       \
  Match3(Pointer, A, B);      \
  Match3(Function, A, B);     \
  // Match3(Array, A, B);  // Disabled as it fails currently

MatchCompounds1(Float);
// MatchCompounds2(Array, Float);
MatchCompounds2(RuntimeArray, Float);
MatchCompounds2(Struct, Float);
MatchCompounds2(Pointer, Float);
MatchCompounds2(Function, Float);

// ForwardPointer tests, which don't fit into the previous mold
#define MatchFpF(N, CODE)                                                    \
  MatchF(N,                                                                  \
         "; CHECK: [[type:%\\w+]] = OpTypeForwardPointer [[pointer:%\\w+]] " \
         "Workgroup\n"                                                       \
         "%type = OpTypeForwardPointer %pointer Workgroup\n" CODE            \
         "; CHECK: [[pointer]] = OpTypePointer Workgroup [[realtype]]\n"     \
         "%pointer = OpTypePointer Workgroup %realtype\n")
#define MatchFp1(T) MatchFpF(ForwardPointerOf##T, MatchPart1(T, realtype))
#define MatchFp2(T, A) \
  MatchFpF(ForwardPointerOf##T, MatchPart1(A, a) MatchPart2(T, realtype, a))

// Disabled currently, causes assertion failures
/*
MatchFp1(Float);
MatchFp2(Array, Float);
MatchFp2(RuntimeArray, Float);
MatchFp2(Struct, Float);
MatchFp2(Function, Float);
// */

}  // namespace
}  // namespace spvtools
