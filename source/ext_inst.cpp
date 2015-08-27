#include <libspirv/libspirv.h>

#include <string.h>

#define GL450InstWithOneIdParam(name) \
  #name, GLSLstd450::GLSLstd450##name, { SPV_OPERAND_TYPE_ID }
#define GL450InstWithTwoIdParam(name)        \
  #name, GLSLstd450::GLSLstd450##name, {     \
    SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID \
  }
#define GL450InstWithThreeIdParam(name)                           \
  #name, GLSLstd450::GLSLstd450##name, {                          \
    SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID \
  }

static const spv_ext_inst_desc_t glslStd450Entries[] = {
    {GL450InstWithOneIdParam(Round)},
    {GL450InstWithOneIdParam(RoundEven)},
    {GL450InstWithOneIdParam(Trunc)},
    {GL450InstWithOneIdParam(FAbs)},
    {GL450InstWithOneIdParam(SAbs)},
    {GL450InstWithOneIdParam(FSign)},
    {GL450InstWithOneIdParam(SSign)},
    {GL450InstWithOneIdParam(Floor)},
    {GL450InstWithOneIdParam(Ceil)},
    {GL450InstWithOneIdParam(Fract)},
    {GL450InstWithOneIdParam(Radians)},
    {GL450InstWithOneIdParam(Degrees)},
    {GL450InstWithOneIdParam(Sin)},
    {GL450InstWithOneIdParam(Cos)},
    {GL450InstWithOneIdParam(Tan)},
    {GL450InstWithOneIdParam(Asin)},
    {GL450InstWithOneIdParam(Acos)},
    {GL450InstWithOneIdParam(Atan)},
    {GL450InstWithOneIdParam(Sinh)},
    {GL450InstWithOneIdParam(Cosh)},
    {GL450InstWithOneIdParam(Tanh)},
    {GL450InstWithOneIdParam(Asinh)},
    {GL450InstWithOneIdParam(Acosh)},
    {GL450InstWithOneIdParam(Atanh)},
    {GL450InstWithTwoIdParam(Atan2)},
    {GL450InstWithTwoIdParam(Pow)},
    {GL450InstWithOneIdParam(Exp)},
    {GL450InstWithOneIdParam(Log)},
    {GL450InstWithOneIdParam(Exp2)},
    {GL450InstWithOneIdParam(Log2)},
    {GL450InstWithOneIdParam(Sqrt)},
    // clang-format off
    {"Inversesqrt", GLSLstd450::GLSLstd450InverseSqrt, {SPV_OPERAND_TYPE_ID}},
    {GL450InstWithOneIdParam(Determinant)},
    {"Inverse", GLSLstd450::GLSLstd450MatrixInverse, {SPV_OPERAND_TYPE_ID}},
    // clang-format on
    {GL450InstWithTwoIdParam(Modf)},
    {GL450InstWithOneIdParam(ModfStruct)},
    {GL450InstWithTwoIdParam(FMin)},
    {GL450InstWithTwoIdParam(UMin)},
    {GL450InstWithTwoIdParam(SMin)},
    {GL450InstWithTwoIdParam(FMax)},
    {GL450InstWithTwoIdParam(UMax)},
    {GL450InstWithTwoIdParam(SMax)},
    {GL450InstWithThreeIdParam(FClamp)},
    {GL450InstWithThreeIdParam(UClamp)},
    {GL450InstWithThreeIdParam(SClamp)},
    {GL450InstWithThreeIdParam(Mix)},
    {GL450InstWithTwoIdParam(Step)},
    // clang-format off
    {"Smoothstep", GLSLstd450::GLSLstd450SmoothStep, {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}},
    // clang-format on
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
