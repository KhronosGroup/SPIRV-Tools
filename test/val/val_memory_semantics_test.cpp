#include "gmock/gmock.h"
#include "test/val/val_fixtures.h"

namespace spvtools {
namespace val {
namespace {

using ::testing::Combine;
using ::testing::HasSubstr;
using ::testing::Values;
using ::testing::ValuesIn;
using ::testing::internal::ParamGenerator;

template <typename T, typename... Ts>
ParamGenerator<T> ValuesInExcept(const T* items, const size_t count,
                                 const Ts... skip) {
  std::vector<T> filtered;
  std::initializer_list<T> excluded = {skip...};
  std::copy_if(items, items + count, std::back_inserter(filtered),
               [&excluded](const T& value) {
                 return std::all_of(
                     excluded.begin(), excluded.end(),
                     [&value](const T& other) { return value != other; });
               });
  return ValuesIn(filtered);
}

struct TestResult {
  explicit TestResult(spv_result_t in_result = SPV_SUCCESS,
                      const char* in_vuid = nullptr,
                      const char* in_error = nullptr)
      : result(in_result), vuid(in_vuid), error(in_error) {}
  spv_result_t result;
  const char* vuid;
  const char* error;
};

enum SemanticsValue {
  None = 0,
  Acquire = 0x00000002,
  Release = 0x00000004,
  AcqRel = 0x00000008,
  SeqCst = 0x00000010,
  Uniform = 0x00000040,
  Subgroup = 0x00000080,
  Workgroup = 0x00000100,
  CrossWorkgroup = 0x00000200,
  AtomicCounter = 0x00000400,
  Image = 0x00000800,
  Output = 0x00001000,
  Available = 0x00002000,
  Visible = 0x00004000,
  Volatile = 0x00008000
};

enum TestOp {
  OpMemoryBarrier,
  OpControlBarrier,
  OpMemoryNamedBarrier,
  OpAtomicLoad,
  OpAtomicStore,
  OpAtomicExchange,
  OpAtomicCompareExchange,
  OpAtomicCompareExchangeEqual,
  OpAtomicCompareExchangeUnequal,
  OpAtomicIIncrement,
  OpAtomicIDecrement,
  OpAtomicIAdd,
  OpAtomicISub,
  OpAtomicSMin,
  OpAtomicUMin,
  OpAtomicSMax,
  OpAtomicUMax,
  OpAtomicAnd,
  OpAtomicOr,
  OpAtomicXor
};

const TestOp TestOps[] = {OpMemoryBarrier,
                          OpControlBarrier,
                          OpMemoryNamedBarrier,
                          OpAtomicLoad,
                          OpAtomicStore,
                          OpAtomicExchange,
                          OpAtomicCompareExchangeEqual,
                          OpAtomicCompareExchangeUnequal,
                          OpAtomicIIncrement,
                          OpAtomicIDecrement,
                          OpAtomicIAdd,
                          OpAtomicISub,
                          OpAtomicSMin,
                          OpAtomicUMin,
                          OpAtomicSMax,
                          OpAtomicUMax,
                          OpAtomicAnd,
                          OpAtomicOr,
                          OpAtomicXor};

const size_t TestOpsCount = sizeof(TestOps) / sizeof(TestOp);

std::string GenerateTestOp(const TestOp op) {
  switch (op) {
    case OpMemoryBarrier:
      return "OpMemoryBarrier %scope %semantics";
    case OpControlBarrier:
      return "OpControlBarrier %uint_2 %scope %semantics";
    case OpMemoryNamedBarrier:
      return "OpMemoryNamedBarrier %barrier %uint_2 %semantics";
    case OpAtomicLoad:
      return "%result = OpAtomicLoad %uint %var %scope %semantics";
    case OpAtomicStore:
      return "OpAtomicStore %var %scope %semantics %uint_1";
    case OpAtomicExchange:
      return "%result = OpAtomicExchange %uint %var %scope %semantics %uint_1";
    case OpAtomicCompareExchange:
      return "%result = OpAtomicCompareExchange %uint %var %scope %semantics "
             "%semantics2 %uint_1 %uint_0";
    case OpAtomicCompareExchangeEqual:
      return "%result = OpAtomicCompareExchange %uint %var %scope %semantics "
             "%semantics_min %uint_1 %uint_0";
    case OpAtomicCompareExchangeUnequal:
      return "%result = OpAtomicCompareExchange %uint %var %scope "
             "%semantics_max %semantics %uint_1 %uint_0";
    case OpAtomicIIncrement:
      return "%result = OpAtomicIIncrement %uint %var %scope %semantics";
    case OpAtomicIDecrement:
      return "%result = OpAtomicIDecrement %uint %var %scope %semantics";
    case OpAtomicIAdd:
      return "%result = OpAtomicIAdd %uint %var %scope %semantics %uint_1";
    case OpAtomicISub:
      return "%result = OpAtomicISub %uint %var %scope %semantics %uint_1";
    case OpAtomicSMin:
      return "%result = OpAtomicSMin %uint %var %scope %semantics %uint_1";
    case OpAtomicUMin:
      return "%result = OpAtomicUMin %uint %var %scope %semantics %uint_1";
    case OpAtomicSMax:
      return "%result = OpAtomicSMax %uint %var %scope %semantics %uint_1";
    case OpAtomicUMax:
      return "%result = OpAtomicUMax %uint %var %scope %semantics %uint_1";
    case OpAtomicAnd:
      return "%result = OpAtomicAnd %uint %var %scope %semantics %uint_1";
    case OpAtomicOr:
      return "%result = OpAtomicOr %uint %var %scope %semantics %uint_1";
    case OpAtomicXor:
      return "%result = OpAtomicXor %uint %var %scope %semantics %uint_1";
    default:
      return "";
  }
}

std::string GenerateVulkanCode(const TestOp op, const uint32_t semantics,
                               const uint32_t semantics2 = 0) {
  std::ostringstream ss;
  ss << R"(
OpCapability Shader
OpCapability VulkanMemoryModel
OpCapability NamedBarrier
OpExtension "SPV_KHR_vulkan_memory_model"
OpMemoryModel Logical Vulkan
OpEntryPoint GLCompute %main "main" %var
OpExecutionMode %main LocalSize 32 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%func = OpTypeFunction %void
%uint_ptr = OpTypePointer Workgroup %uint
%named_barrier = OpTypeNamedBarrier
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%scope = OpConstant %uint 5
%semantics = OpConstant %uint )"
     << semantics << R"(
%semantics2 = OpConstant %uint )"
     << semantics2 << R"(
%semantics_min = OpConstant %uint )"
     << (semantics & Volatile) << R"(
%semantics_max = OpConstant %uint )"
     << (32712 | (semantics & Volatile)) << R"(
%var = OpVariable %uint_ptr Workgroup
%main = OpFunction %void None %func
%label = OpLabel
%barrier = OpNamedBarrierInitialize %named_barrier %uint_4
)" << GenerateTestOp(op)
     << R"(
OpReturn
OpFunctionEnd)";

  return ss.str();
}

using VulkanSemantics =
    spvtest::ValidateBase<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, TestOp, TestResult>>;
using VulkanSemanticsCmpXchg = spvtest::ValidateBase<
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
               uint32_t, uint32_t, bool, uint32_t, TestResult>>;

INSTANTIATE_TEST_SUITE_P(
    ErrorMultipleMemoryOrderBits, VulkanSemantics,
    Combine(
        Values(Acquire | Release, Acquire | AcqRel, Release | AcqRel,
               Acquire | SeqCst, Release | SeqCst, AcqRel | SeqCst),
        Values(None, Uniform | Workgroup | Image | Output),
        Values(None, Available, Visible, Available | Visible),
        Values(None, Volatile),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        ValuesIn(TestOps),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10001",
            "Memory Semantics must have at most one non-relaxed memory order "
            "bit set (Acquire, Release, or AcquireRelease)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorAtomicLoadWithReleaseMemoryOrder, VulkanSemantics,
    Combine(Values(Release, AcqRel),
            Values(None, Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpAtomicLoad),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10002",
                "AtomicLoad must have Relaxed or Acquire memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorAtomicStoreWithAcquireMemoryOrder, VulkanSemantics,
    Combine(Values(Acquire, AcqRel),
            Values(None, Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Visible, Available, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpAtomicStore),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10003",
                "AtomicStore must have Relaxed or Release memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMemoryBarrierWithRelaxedMemoryOrder, VulkanSemantics,
    Combine(Values(None),
            Values(None, Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpMemoryBarrier),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10004",
                "MemoryBarrier must have Acquire, Release, or AcquireRelease "
                "memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorSequentiallyConsistentMemoryOrder, VulkanSemantics,
    Combine(Values(SeqCst), Values(None, Uniform | Workgroup | Image | Output),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesIn(TestOps),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10005",
                "Memory Semantics must not have SequentiallyConsistent "
                "memory order in Vulkan environment"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorRelaxedSemanticsWithStorageClass, VulkanSemantics,
    Combine(Values(None),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10006",
                "Memory Semantics with at least one storage class semantics "
                "flag (UniformMemory, WorkgroupMemory, ImageMemory, or "
                "OutputMemory) must have a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorNonRelaxedSemanticsWithoutStorageClass, VulkanSemantics,
    Combine(Values(Acquire, Release, AcqRel), Values(None),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpAtomicLoad, OpAtomicStore),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10007",
                "Memory Semantics with a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease) must have at least one storage "
                "class semantics flag (UniformMemory, WorkgroupMemory, "
                "ImageMemory, or OutputMemory)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorNonRelaxedSemanticsWithoutStorageClassLoad, VulkanSemantics,
    Combine(Values(Acquire), Values(None),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpAtomicLoad),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10007",
                "Memory Semantics with a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease) must have at least one storage "
                "class semantics flag (UniformMemory, WorkgroupMemory, "
                "ImageMemory, or OutputMemory)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorNonRelaxedSemanticsWithoutStorageClassStore, VulkanSemantics,
    Combine(Values(Release), Values(None),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpAtomicStore),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10007",
                "Memory Semantics with a non-relaxed memory order (Acquire, "
                "Release, or AcquireRelease) must have at least one storage "
                "class semantics flag (UniformMemory, WorkgroupMemory, "
                "ImageMemory, or OutputMemory)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMakeAvailableWithRelaxedMemoryOrder, VulkanSemantics,
    Combine(Values(None), Values(None), Values(Available, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10008",
                "Memory Semantics with MakeAvailable flag must have Release "
                "or AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMakeAvailableWithAcquireMemoryOrder, VulkanSemantics,
    Combine(Values(Acquire),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(Available, Available | Visible), Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpAtomicStore),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10008",
                "Memory Semantics with MakeAvailable flag must have Release "
                "or AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMakeVisibleWithRelaxedMemoryOrder, VulkanSemantics,
    Combine(
        Values(None), Values(None), Values(Visible), Values(None, Volatile),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "VUID-StandaloneSpirv-MemorySemantics-10009",
                          "Memory Semantics with MakeVisible flag must have "
                          "Acquire or AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMakeVisibleWithReleaseMemoryOrder, VulkanSemantics,
    Combine(
        Values(Release),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(Visible, Available | Visible), Values(None, Volatile),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        ValuesInExcept(TestOps, TestOpsCount, OpAtomicLoad),
        Values(TestResult(SPV_ERROR_INVALID_DATA,
                          "VUID-StandaloneSpirv-MemorySemantics-10009",
                          "Memory Semantics with MakeVisible flag must have "
                          "Acquire or AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorVolatileBarrierWithRelaxedSemantics, VulkanSemantics,
    Combine(Values(None), Values(None), Values(None), Values(Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpControlBarrier, OpMemoryNamedBarrier),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10010",
                "Memory Semantics Volatile flag must not be used with "
                "barrier instructions (MemoryBarrier, ControlBarrier, "
                "and MemoryNamedBarrier)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorVolatileBarrierWithNonRelaxedSemantics, VulkanSemantics,
    Combine(Values(Acquire, Acquire | Visible, Release, Release | Available,
                   AcqRel, AcqRel | Visible, AcqRel | Available,
                   AcqRel | Available | Visible),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None), Values(Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpControlBarrier, OpMemoryBarrier, OpMemoryNamedBarrier),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10010",
                "Memory Semantics Volatile flag must not be used with "
                "barrier instructions (MemoryBarrier, ControlBarrier, "
                "and MemoryNamedBarrier)"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorCompareExchangeUnequalSemanticsWithRelease, VulkanSemantics,
    Combine(Values(Release, AcqRel, AcqRel | Visible),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available), Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpAtomicCompareExchangeUnequal),
            Values(TestResult(
                SPV_ERROR_INVALID_DATA,
                "VUID-StandaloneSpirv-MemorySemantics-10011",
                "AtomicCompareExchange Unequal Memory Semantics must have "
                "Relaxed or Acquire memory order"))));

INSTANTIATE_TEST_SUITE_P(
    SuccessAtomicsRelaxed, VulkanSemantics,
    Combine(Values(None), Values(None), Values(None), Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier,
                           OpControlBarrier, OpMemoryNamedBarrier),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessAtomicsAcquire, VulkanSemantics,
    Combine(Values(Acquire),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Visible), Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier,
                           OpControlBarrier, OpMemoryNamedBarrier,
                           OpAtomicStore),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessAtomicsRelease, VulkanSemantics,
    Combine(Values(Release),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available), Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier,
                           OpControlBarrier, OpMemoryNamedBarrier, OpAtomicLoad,
                           OpAtomicCompareExchangeUnequal),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessAtomicsAcqRel, VulkanSemantics,
    Combine(Values(AcqRel),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None, Available, Visible, Available | Visible),
            Values(None, Volatile),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            ValuesInExcept(TestOps, TestOpsCount, OpMemoryBarrier,
                           OpControlBarrier, OpMemoryNamedBarrier, OpAtomicLoad,
                           OpAtomicStore, OpAtomicCompareExchangeUnequal),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessBarriersRelaxed, VulkanSemantics,
    Combine(Values(None), Values(None), Values(None), Values(None),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpControlBarrier, OpMemoryNamedBarrier),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessBarriersNonRelaxed, VulkanSemantics,
    Combine(Values(Acquire, Acquire | Visible, Release, Release | Available,
                   AcqRel, AcqRel | Available, AcqRel | Visible,
                   AcqRel | Available | Visible),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None), Values(None),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(OpControlBarrier, OpMemoryBarrier, OpMemoryNamedBarrier),
            Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    ErrorMemoryOrderTooWeakRelaxed, VulkanSemanticsCmpXchg,
    Combine(
        Values(None), Values(None), Values(None),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(None, Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(true, false), Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10012",
            "AtomicCompareExchange Unequal Memory Semantics can have Acquire "
            "memory order only if Equal Memory Semantics have Acquire or "
            "AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMemoryOrderTooWeakRelease, VulkanSemanticsCmpXchg,
    Combine(
        Values(Release),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(None, Available),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(None, Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(true, false), Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10012",
            "AtomicCompareExchange Unequal Memory Semantics can have Acquire "
            "memory order only if Equal Memory Semantics have Acquire or "
            "AcquireRelease memory order"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMissingStorageClassSemanticsFlags, VulkanSemanticsCmpXchg,
    Combine(
        Values(Acquire, AcqRel, AcqRel | Available),
        Values(Uniform | Workgroup), Values(None, Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire), Values(Uniform | Image, Output), Values(None, Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(true, false), Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10013",
            "AtomicCompareExchange Unequal Memory Semantics must not have any "
            "storage class semantics flag (UniformMemory, WorkgroupMemory, "
            "ImageMemory, or OutputMemory) or MakeVisible flag, unless these "
            "flags are also present in the Equal Memory Semantics"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMissingMakeVisibleFlag, VulkanSemanticsCmpXchg,
    Combine(
        Values(Acquire, AcqRel, AcqRel | Available),
        Values(Uniform | Workgroup | Image | Output), Values(None),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(true, false), Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10013",
            "AtomicCompareExchange Unequal Memory Semantics must not have any "
            "storage class semantics flag (UniformMemory, WorkgroupMemory, "
            "ImageMemory, or OutputMemory) or MakeVisible flag, unless these "
            "flags are also present in the Equal Memory Semantics"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMismatchingVolatileFlagsRelaxedAndRelaxed, VulkanSemanticsCmpXchg,
    Combine(
        Values(None), Values(None), Values(None),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter), Values(None),
        Values(None), Values(None),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter), Values(false),
        Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10014",
            "AtomicCompareExchange Unequal Memory Semantics Volatile flag must "
            "match the Equal Memory Semantics flag"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMismatchingVolatileFlagsNonRelaxedAndRelaxed, VulkanSemanticsCmpXchg,
    Combine(
        Values(Acquire, Acquire | Visible, Release, Release | Available, AcqRel,
               AcqRel | Visible, AcqRel | Available,
               AcqRel | Available | Visible),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(None), Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(None), Values(None), Values(None),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter), Values(false),
        Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10014",
            "AtomicCompareExchange Unequal Memory Semantics Volatile flag must "
            "match the Equal Memory Semantics flag"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMismatchingVolatileFlagsNonRelaxedAndAcquire, VulkanSemanticsCmpXchg,
    Combine(
        Values(Acquire, AcqRel, AcqRel | Available),
        Values(Uniform | Workgroup | Image | Output), Values(None, Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(None), Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(false), Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10014",
            "AtomicCompareExchange Unequal Memory Semantics Volatile flag must "
            "match the Equal Memory Semantics flag"))));

INSTANTIATE_TEST_SUITE_P(
    ErrorMismatchingVolatileFlagsNonRelaxedAndAcquireVisible,
    VulkanSemanticsCmpXchg,
    Combine(
        Values(Acquire, AcqRel, AcqRel | Available),
        Values(Uniform | Workgroup | Image | Output), Values(Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
        Values(Acquire),
        Values(Uniform, Workgroup, Image, Output,
               Uniform | Workgroup | Image | Output),
        Values(Visible),
        Values(None, Subgroup | CrossWorkgroup | AtomicCounter), Values(false),
        Values(None, Volatile),
        Values(TestResult(
            SPV_ERROR_INVALID_DATA,
            "VUID-StandaloneSpirv-MemorySemantics-10014",
            "AtomicCompareExchange Unequal Memory Semantics Volatile flag must "
            "match the Equal Memory Semantics flag"))));

INSTANTIATE_TEST_SUITE_P(
    SuccessNonRelaxedAndAcquire, VulkanSemanticsCmpXchg,
    Combine(Values(Acquire, AcqRel, AcqRel | Available),
            Values(Uniform | Workgroup | Image | Output), Values(None, Visible),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(Acquire),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(None),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(true), Values(None, Volatile), Values(TestResult())));

INSTANTIATE_TEST_SUITE_P(
    SuccessNonRelaxedAndAcquireVisible, VulkanSemanticsCmpXchg,
    Combine(Values(Acquire, AcqRel, AcqRel | Available),
            Values(Uniform | Workgroup | Image | Output), Values(Visible),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(Acquire),
            Values(Uniform, Workgroup, Image, Output,
                   Uniform | Workgroup | Image | Output),
            Values(Visible),
            Values(None, Subgroup | CrossWorkgroup | AtomicCounter),
            Values(true), Values(None, Volatile), Values(TestResult())));

TEST_P(VulkanSemantics, ModelAndEnv) {
  const uint32_t semantics = std::get<0>(GetParam()) | std::get<1>(GetParam()) |
                             std::get<2>(GetParam()) | std::get<3>(GetParam()) |
                             std::get<4>(GetParam());
  const TestOp op = std::get<5>(GetParam());
  const TestResult& result = std::get<6>(GetParam());

  CompileSuccessfully(GenerateVulkanCode(op, semantics), SPV_ENV_VULKAN_1_4);
  ASSERT_EQ(result.result, ValidateInstructions(SPV_ENV_VULKAN_1_4));
  if (result.vuid) {
    EXPECT_THAT(getDiagnosticString(), AnyVUID(result.vuid));
  }
  if (result.error) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(result.error));
  }
}

TEST_P(VulkanSemantics, Model) {
  const uint32_t semantics = std::get<0>(GetParam()) | std::get<1>(GetParam()) |
                             std::get<2>(GetParam()) | std::get<3>(GetParam()) |
                             std::get<4>(GetParam());
  const TestOp op = std::get<5>(GetParam());
  const TestResult& result = std::get<6>(GetParam());

  CompileSuccessfully(GenerateVulkanCode(op, semantics), SPV_ENV_UNIVERSAL_1_6);
  ASSERT_EQ(result.result, ValidateInstructions(SPV_ENV_UNIVERSAL_1_6));
  if (result.error) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(result.error));
  }
}

TEST_P(VulkanSemanticsCmpXchg, ModelAndEnv) {
  const uint32_t equal_volatile = std::get<9>(GetParam());
  const uint32_t unequal_volatile =
      std::get<8>(GetParam()) ? equal_volatile : Volatile ^ equal_volatile;
  const uint32_t equal = std::get<0>(GetParam()) | std::get<1>(GetParam()) |
                         std::get<2>(GetParam()) | std::get<3>(GetParam()) |
                         equal_volatile;
  const uint32_t unequal = std::get<4>(GetParam()) | std::get<5>(GetParam()) |
                           std::get<6>(GetParam()) | std::get<7>(GetParam()) |
                           unequal_volatile;
  const TestResult& result = std::get<10>(GetParam());

  CompileSuccessfully(
      GenerateVulkanCode(OpAtomicCompareExchange, equal, unequal),
      SPV_ENV_VULKAN_1_4);
  ASSERT_EQ(result.result, ValidateInstructions(SPV_ENV_VULKAN_1_4));
  if (result.vuid) {
    EXPECT_THAT(getDiagnosticString(), AnyVUID(result.vuid));
  }
  if (result.error) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(result.error));
  }
}

TEST_P(VulkanSemanticsCmpXchg, Model) {
  const uint32_t equal_volatile = std::get<9>(GetParam());
  const uint32_t unequal_volatile =
      std::get<8>(GetParam()) ? equal_volatile : Volatile ^ equal_volatile;
  const uint32_t equal = std::get<0>(GetParam()) | std::get<1>(GetParam()) |
                         std::get<2>(GetParam()) | std::get<3>(GetParam()) |
                         equal_volatile;
  const uint32_t unequal = std::get<4>(GetParam()) | std::get<5>(GetParam()) |
                           std::get<6>(GetParam()) | std::get<7>(GetParam()) |
                           unequal_volatile;
  const TestResult& result = std::get<10>(GetParam());

  CompileSuccessfully(
      GenerateVulkanCode(OpAtomicCompareExchange, equal, unequal),
      SPV_ENV_VULKAN_1_4);
  ASSERT_EQ(result.result, ValidateInstructions(SPV_ENV_UNIVERSAL_1_6));
  if (result.error) {
    EXPECT_THAT(getDiagnosticString(), HasSubstr(result.error));
  }
}

}  // namespace
}  // namespace val
}  // namespace spvtools
