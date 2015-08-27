#include <libspirv/libspirv.h>

#include <string.h>

/// Generate a spv_ext_inst_desc_t literal for a GLSL std450 extended
/// instruction with one/two/three <id> parameter(s).
#define GLSL450Inst1(name) \
  #name, GLSLstd450::GLSLstd450##name, { SPV_OPERAND_TYPE_ID }
#define GLSL450Inst2(name)                   \
  #name, GLSLstd450::GLSLstd450##name, {     \
    SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID \
  }
#define GLSL450Inst3(name)                                        \
  #name, GLSLstd450::GLSLstd450##name, {                          \
    SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID \
  }

static const spv_ext_inst_desc_t glslStd450Entries[] = {
    {GLSL450Inst1(Round)},
    {GLSL450Inst1(RoundEven)},
    {GLSL450Inst1(Trunc)},
    {GLSL450Inst1(FAbs)},
    {GLSL450Inst1(SAbs)},
    {GLSL450Inst1(FSign)},
    {GLSL450Inst1(SSign)},
    {GLSL450Inst1(Floor)},
    {GLSL450Inst1(Ceil)},
    {GLSL450Inst1(Fract)},
    {GLSL450Inst1(Radians)},
    {GLSL450Inst1(Degrees)},
    {GLSL450Inst1(Sin)},
    {GLSL450Inst1(Cos)},
    {GLSL450Inst1(Tan)},
    {GLSL450Inst1(Asin)},
    {GLSL450Inst1(Acos)},
    {GLSL450Inst1(Atan)},
    {GLSL450Inst1(Sinh)},
    {GLSL450Inst1(Cosh)},
    {GLSL450Inst1(Tanh)},
    {GLSL450Inst1(Asinh)},
    {GLSL450Inst1(Acosh)},
    {GLSL450Inst1(Atanh)},
    {GLSL450Inst2(Atan2)},
    {GLSL450Inst2(Pow)},
    {GLSL450Inst1(Exp)},
    {GLSL450Inst1(Log)},
    {GLSL450Inst1(Exp2)},
    {GLSL450Inst1(Log2)},
    {GLSL450Inst1(Sqrt)},
    // clang-format off
    {"Inversesqrt", GLSLstd450::GLSLstd450InverseSqrt, {SPV_OPERAND_TYPE_ID}},
    {GLSL450Inst1(Determinant)},
    {"Inverse", GLSLstd450::GLSLstd450MatrixInverse, {SPV_OPERAND_TYPE_ID}},
    // clang-format on
    {GLSL450Inst2(Modf)},
    {GLSL450Inst1(ModfStruct)},
    {GLSL450Inst2(FMin)},
    {GLSL450Inst2(UMin)},
    {GLSL450Inst2(SMin)},
    {GLSL450Inst2(FMax)},
    {GLSL450Inst2(UMax)},
    {GLSL450Inst2(SMax)},
    {GLSL450Inst3(FClamp)},
    {GLSL450Inst3(UClamp)},
    {GLSL450Inst3(SClamp)},
    {GLSL450Inst3(Mix)},
    {GLSL450Inst2(Step)},
    // clang-format off
    {"Smoothstep", GLSLstd450::GLSLstd450SmoothStep, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}},
    // clang-format on
    {GLSL450Inst3(Fma)},
    {GLSL450Inst2(Frexp)},
    {GLSL450Inst1(FrexpStruct)},
    {GLSL450Inst2(Ldexp)},
    // TODO: Add remaining GLSL.std.450 instructions
};

static const spv_ext_inst_desc_t openclStd12Entries[] = {
    {"placeholder", 0, {}},
    // TODO: Add remaining OpenCL.std.12 instructions
};

static const spv_ext_inst_desc_t openclStd20Entries[] = {
    {"placeholder", 0, {}},
    // TODO: Add remaining OpenCL.std.20 instructions
};

static const spv_ext_inst_desc_t openclStd21Entries[] = {
    {"placeholder", 0, {}},
    // TODO: Add remaining OpenCL.std.21 instructions
};

spv_result_t spvExtInstTableGet(spv_ext_inst_table *pExtInstTable) {
  spvCheck(!pExtInstTable, return SPV_ERROR_INVALID_POINTER);

  static const spv_ext_inst_group_t groups[] = {
      {SPV_EXT_INST_TYPE_GLSL_STD_450,
       sizeof(glslStd450Entries) / sizeof(spv_ext_inst_desc_t),
       glslStd450Entries},
      {SPV_EXT_INST_TYPE_OPENCL_STD_12,
       sizeof(openclStd12Entries) / sizeof(spv_ext_inst_desc_t),
       openclStd12Entries},
      {SPV_EXT_INST_TYPE_OPENCL_STD_20,
       sizeof(openclStd20Entries) / sizeof(spv_ext_inst_desc_t),
       openclStd20Entries},
      {SPV_EXT_INST_TYPE_OPENCL_STD_21,
       sizeof(openclStd21Entries) / sizeof(spv_ext_inst_desc_t),
       openclStd21Entries},
  };

  static const spv_ext_inst_table_t table = {
      sizeof(groups) / sizeof(spv_ext_inst_group_t), groups};

  *pExtInstTable = &table;

  return SPV_SUCCESS;
}

spv_ext_inst_type_t spvExtInstImportTypeGet(const char *name) {
  if (!strcmp("GLSL.std.450", name)) {
    return SPV_EXT_INST_TYPE_GLSL_STD_450;
  }
  if (!strcmp("OpenCL.std.12", name)) {
    return SPV_EXT_INST_TYPE_OPENCL_STD_12;
  }
  if (!strcmp("OpenCL.std.20", name)) {
    return SPV_EXT_INST_TYPE_OPENCL_STD_20;
  }
  if (!strcmp("OpenCL.std.21", name)) {
    return SPV_EXT_INST_TYPE_OPENCL_STD_21;
  }
  return SPV_EXT_INST_TYPE_NONE;
}

spv_result_t spvExtInstTableNameLookup(const spv_ext_inst_table table,
                                       const spv_ext_inst_type_t type,
                                       const char *name,
                                       spv_ext_inst_desc *pEntry) {
  spvCheck(!table, return SPV_ERROR_INVALID_TABLE);
  spvCheck(!pEntry, return SPV_ERROR_INVALID_POINTER);

  for (uint32_t groupIndex = 0; groupIndex < table->count; groupIndex++) {
    auto &group = table->groups[groupIndex];
    if (type == group.type) {
      for (uint32_t index = 0; index < group.count; index++) {
        auto &entry = group.entries[index];
        if (!strcmp(name, entry.name)) {
          *pEntry = &table->groups[groupIndex].entries[index];
          return SPV_SUCCESS;
        }
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}

spv_result_t spvExtInstTableValueLookup(const spv_ext_inst_table table,
                                        const spv_ext_inst_type_t type,
                                        const uint32_t value,
                                        spv_ext_inst_desc *pEntry) {
  spvCheck(!table, return SPV_ERROR_INVALID_TABLE);
  spvCheck(!pEntry, return SPV_ERROR_INVALID_POINTER);

  for (uint32_t groupIndex = 0; groupIndex < table->count; groupIndex++) {
    auto &group = table->groups[groupIndex];
    if (type == group.type) {
      for (uint32_t index = 0; index < group.count; index++) {
        auto &entry = group.entries[index];
        if (value == entry.ext_inst) {
          *pEntry = &table->groups[groupIndex].entries[index];
          return SPV_SUCCESS;
        }
      }
    }
  }

  return SPV_ERROR_INVALID_LOOKUP;
}
