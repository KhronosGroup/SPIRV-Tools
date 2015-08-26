#include <libspirv/libspirv.h>

#include <string.h>

#define GL450Inst(name) #name, GLSLstd450::GLSLstd450##name

static const spv_ext_inst_desc_t glslStd450Entries[] = {
    { GL450Inst(Round), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(RoundEven), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Trunc), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(FAbs), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SAbs), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(FSign), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SSign), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Floor), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Ceil), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Fract), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Radians), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Degrees), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Sin), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Cos), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Tan), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Asin), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Acos), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Atan), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Sinh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Cosh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Tanh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Asinh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Acosh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Atanh), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Atan2), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Pow), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Exp), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Log), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Exp2), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Log2), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Sqrt), {SPV_OPERAND_TYPE_ID}, },
    { "Inversesqrt", GLSLstd450::GLSLstd450InverseSqrt, {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Determinant), {SPV_OPERAND_TYPE_ID}, },
    { "Inverse", GLSLstd450::GLSLstd450MatrixInverse, {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Modf), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(ModfStruct), {SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(FMin), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(UMin), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SMin), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(FMax), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(UMax), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SMax), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(FClamp), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(UClamp), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SClamp), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Mix), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(Step), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
    { GL450Inst(SmoothStep), {SPV_OPERAND_TYPE_ID, SPV_OPERAND_TYPE_ID}, },
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
