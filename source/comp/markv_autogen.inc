
std::map<uint64_t, uint32_t> GetOpcodeAndNumOperandsHist() {
  return std::map<uint64_t, uint32_t>({
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 158282 },
    { CombineOpcodeAndNumOperands(SpvOpDot, 4), 151035 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 183292 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 126492 },
    { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 2), 13311 },
    { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 29952 },
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 106847 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 26350 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 28186 },
    { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 41635 },
    { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 412786 },
    { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 62905 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 118614 },
    { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 100735 },
    { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 40852 },
    { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 157091 },
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 122100 },
    { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 82930 },
    { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 161019 },
    { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 466014 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 107126 },
    { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 34775 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleDrefExplicitLod, 7), 26146 },
    { CombineOpcodeAndNumOperands(SpvOpMemoryModel, 2), 18879 },
    { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 485251 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 78011 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 18879 },
    { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 15848 },
    { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 69404 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 19998 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 40228 },
    { CombineOpcodeAndNumOperands(SpvOpCapability, 1), 22510 },
    { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 37585 },
    { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 30454 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 29021 },
    { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 342237 },
    { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 24449 },
    { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 129408 },
    { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 246535 },
    { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 503456 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 19779 },
    { CombineOpcodeAndNumOperands(SpvOpBranchConditional, 3), 24139 },
    { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 697946 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 55769 },
    { CombineOpcodeAndNumOperands(SpvOpTypeVoid, 1), 18879 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 145508 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 85583 },
    { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 34775 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 66362 },
    { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 1272902 },
    { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 22122 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 861008 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 62905 },
    { CombineOpcodeAndNumOperands(SpvOpExtInstImport, 2), 18879 },
    { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 22009 },
    { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 38275 },
    { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 12208 },
    { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 95518 },
    { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 94887 },
    { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 1942215 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 205266 },
    { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 22157 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 142749 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 24420 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 16896 },
    { CombineOpcodeAndNumOperands(SpvOpStore, 2), 604982 },
    { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 14471 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 269658 },
    { kMarkvNoneOfTheAbove, 399895 },
  });
}

std::map<uint32_t, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetOpcodeAndNumOperandsMarkovHuffmanCodecs() {
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(51, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393297, 0, 0},
      {393298, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 18, 6},
      {0, 13, 27},
      {0, 19, 12},
      {0, 15, 22},
      {0, 28, 21},
      {0, 29, 16},
      {0, 30, 10},
      {0, 17, 20},
      {0, 11, 25},
      {0, 26, 31},
      {0, 4, 32},
      {0, 34, 33},
      {0, 24, 35},
      {0, 23, 36},
      {0, 37, 14},
      {0, 38, 8},
      {0, 1, 39},
      {0, 3, 9},
      {0, 41, 40},
      {0, 42, 2},
      {0, 43, 5},
      {0, 45, 44},
      {0, 47, 46},
      {0, 48, 7},
      {0, 50, 49},
    }));

    codecs.emplace(SpvOpFMul, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(53, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262230, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393297, 0, 0},
      {393303, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 9, 23},
      {0, 21, 28},
      {0, 20, 29},
      {0, 14, 30},
      {0, 19, 8},
      {0, 31, 16},
      {0, 25, 11},
      {0, 32, 18},
      {0, 33, 15},
      {0, 4, 12},
      {0, 22, 26},
      {0, 17, 34},
      {0, 13, 6},
      {0, 24, 35},
      {0, 27, 36},
      {0, 3, 37},
      {0, 39, 38},
      {0, 2, 40},
      {0, 42, 41},
      {0, 7, 1},
      {0, 10, 43},
      {0, 5, 44},
      {0, 46, 45},
      {0, 48, 47},
      {0, 50, 49},
      {0, 52, 51},
    }));

    codecs.emplace(SpvOpFAdd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {196631, 0, 0},
      {196640, 0, 0},
      {196641, 0, 0},
      {196651, 0, 0},
      {196667, 0, 0},
      {262188, 0, 0},
      {327724, 0, 0},
      {393260, 0, 0},
      {524313, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 6, 8},
      {0, 7, 11},
      {0, 1, 12},
      {0, 13, 9},
      {0, 14, 3},
      {0, 15, 10},
      {0, 4, 2},
      {0, 17, 16},
      {0, 5, 18},
    }));

    codecs.emplace(SpvOpTypePointer, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(49, {
      {0, 0, 0},
      {65790, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262280, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {262328, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 21, 24},
      {0, 25, 26},
      {0, 15, 20},
      {0, 23, 13},
      {0, 5, 10},
      {0, 11, 27},
      {0, 17, 7},
      {0, 28, 18},
      {0, 1, 29},
      {0, 30, 2},
      {0, 31, 19},
      {0, 32, 16},
      {0, 33, 8},
      {0, 4, 3},
      {0, 9, 34},
      {0, 35, 6},
      {0, 12, 22},
      {0, 37, 36},
      {0, 39, 38},
      {0, 40, 14},
      {0, 42, 41},
      {0, 44, 43},
      {0, 46, 45},
      {0, 48, 47},
    }));

    codecs.emplace(SpvOpFSub, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(51, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393297, 0, 0},
      {393298, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {458842, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 20, 13},
      {0, 21, 27},
      {0, 6, 16},
      {0, 22, 28},
      {0, 29, 1},
      {0, 30, 12},
      {0, 31, 18},
      {0, 3, 4},
      {0, 2, 25},
      {0, 11, 32},
      {0, 33, 23},
      {0, 24, 34},
      {0, 35, 10},
      {0, 8, 15},
      {0, 7, 36},
      {0, 37, 17},
      {0, 38, 26},
      {0, 40, 39},
      {0, 41, 14},
      {0, 9, 42},
      {0, 43, 19},
      {0, 45, 44},
      {0, 47, 46},
      {0, 49, 48},
      {0, 50, 5},
    }));

    codecs.emplace(SpvOpCompositeExtract, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(49, {
      {0, 0, 0},
      {65562, 0, 0},
      {131099, 0, 0},
      {131134, 0, 0},
      {196629, 0, 0},
      {196631, 0, 0},
      {196640, 0, 0},
      {196651, 0, 0},
      {196667, 0, 0},
      {196669, 0, 0},
      {262188, 0, 0},
      {262209, 0, 0},
      {262225, 0, 0},
      {262275, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327724, 0, 0},
      {327745, 0, 0},
      {393228, 0, 0},
      {393260, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {458831, 0, 0},
      {524313, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 21, 14},
      {0, 13, 26},
      {0, 27, 20},
      {0, 18, 28},
      {0, 24, 29},
      {0, 30, 12},
      {0, 31, 22},
      {0, 3, 15},
      {0, 33, 32},
      {0, 34, 10},
      {0, 35, 17},
      {0, 1, 19},
      {0, 5, 11},
      {0, 36, 23},
      {0, 4, 16},
      {0, 2, 37},
      {0, 39, 38},
      {0, 25, 9},
      {0, 41, 40},
      {0, 43, 42},
      {0, 6, 44},
      {0, 7, 45},
      {0, 47, 46},
      {0, 8, 48},
    }));

    codecs.emplace(SpvOpVariable, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 3},
      {0, 2, 4},
    }));

    codecs.emplace(SpvOpAccessChain, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(49, {
      {0, 0, 0},
      {252, 0, 0},
      {65785, 0, 0},
      {131134, 0, 0},
      {131319, 0, 0},
      {196667, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262225, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393298, 0, 0},
      {393461, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 18, 16},
      {0, 14, 26},
      {0, 10, 27},
      {0, 28, 17},
      {0, 12, 29},
      {0, 24, 20},
      {0, 31, 30},
      {0, 32, 11},
      {0, 33, 21},
      {0, 34, 13},
      {0, 35, 9},
      {0, 15, 23},
      {0, 36, 4},
      {0, 37, 19},
      {0, 38, 3},
      {0, 7, 1},
      {0, 8, 39},
      {0, 2, 25},
      {0, 22, 40},
      {0, 42, 41},
      {0, 44, 43},
      {0, 46, 45},
      {0, 6, 5},
      {0, 48, 47},
    }));

    codecs.emplace(SpvOpLabel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(51, {
      {0, 0, 0},
      {253, 0, 0},
      {65785, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262201, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 20, 23},
      {0, 19, 27},
      {0, 28, 13},
      {0, 9, 29},
      {0, 18, 30},
      {0, 10, 31},
      {0, 12, 22},
      {0, 32, 14},
      {0, 33, 16},
      {0, 34, 24},
      {0, 35, 25},
      {0, 36, 21},
      {0, 11, 7},
      {0, 38, 37},
      {0, 40, 39},
      {0, 8, 41},
      {0, 1, 2},
      {0, 42, 5},
      {0, 17, 26},
      {0, 15, 43},
      {0, 45, 44},
      {0, 46, 3},
      {0, 48, 47},
      {0, 4, 6},
      {0, 50, 49},
    }));

    codecs.emplace(SpvOpStore, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(57, {
      {0, 0, 0},
      {65790, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262230, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393297, 0, 0},
      {393304, 0, 0},
      {458764, 0, 0},
      {458817, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 20, 26},
      {0, 23, 30},
      {0, 17, 31},
      {0, 25, 5},
      {0, 14, 32},
      {0, 24, 15},
      {0, 1, 33},
      {0, 34, 22},
      {0, 35, 16},
      {0, 36, 10},
      {0, 8, 19},
      {0, 18, 13},
      {0, 12, 37},
      {0, 38, 9},
      {0, 28, 39},
      {0, 7, 40},
      {0, 29, 6},
      {0, 11, 41},
      {0, 42, 21},
      {0, 43, 27},
      {0, 4, 44},
      {0, 45, 2},
      {0, 3, 46},
      {0, 48, 47},
      {0, 50, 49},
      {0, 52, 51},
      {0, 54, 53},
      {0, 56, 55},
    }));

    codecs.emplace(SpvOpLoad, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(53, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262280, 0, 0},
      {262286, 0, 0},
      {262288, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393304, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 15, 20},
      {0, 17, 28},
      {0, 29, 19},
      {0, 6, 11},
      {0, 31, 30},
      {0, 14, 26},
      {0, 23, 32},
      {0, 12, 4},
      {0, 21, 25},
      {0, 10, 33},
      {0, 22, 7},
      {0, 34, 27},
      {0, 18, 13},
      {0, 35, 8},
      {0, 36, 16},
      {0, 37, 3},
      {0, 39, 38},
      {0, 41, 40},
      {0, 42, 2},
      {0, 9, 43},
      {0, 44, 5},
      {0, 24, 45},
      {0, 46, 1},
      {0, 48, 47},
      {0, 50, 49},
      {0, 52, 51},
    }));

    codecs.emplace(SpvOpCompositeConstruct, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {131143, 0, 0},
      {196679, 0, 0},
      {196680, 0, 0},
      {262216, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 2},
      {0, 3, 6},
      {0, 7, 1},
      {0, 4, 8},
    }));

    codecs.emplace(SpvOpMemberDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {65556, 0, 0},
      {196631, 0, 0},
      {196640, 0, 0},
      {196651, 0, 0},
      {196667, 0, 0},
      {262188, 0, 0},
      {327724, 0, 0},
      {393260, 0, 0},
      {524313, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 8, 10},
      {0, 1, 2},
      {0, 7, 11},
      {0, 3, 12},
      {0, 9, 13},
      {0, 14, 6},
      {0, 15, 5},
      {0, 17, 16},
      {0, 4, 18},
    }));

    codecs.emplace(SpvOpConstantComposite, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {65555, 0, 0},
      {131143, 0, 0},
      {196679, 0, 0},
      {196680, 0, 0},
      {262216, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 6},
      {0, 1, 2},
      {0, 8, 7},
      {0, 5, 9},
      {0, 3, 10},
    }));

    codecs.emplace(SpvOpDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {196632, 0, 0},
      {196636, 0, 0},
      {196640, 0, 0},
      {196651, 0, 0},
      {196667, 0, 0},
      {262188, 0, 0},
      {327724, 0, 0},
      {393260, 0, 0},
      {524313, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 9},
      {0, 2, 10},
      {0, 12, 11},
      {0, 7, 8},
      {0, 13, 3},
      {0, 14, 6},
      {0, 15, 5},
      {0, 17, 16},
      {0, 18, 4},
    }));

    codecs.emplace(SpvOpConstant, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(51, {
      {0, 0, 0},
      {65790, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {196735, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 8, 20},
      {0, 22, 27},
      {0, 9, 28},
      {0, 15, 29},
      {0, 13, 30},
      {0, 21, 17},
      {0, 1, 18},
      {0, 4, 26},
      {0, 23, 31},
      {0, 12, 6},
      {0, 33, 32},
      {0, 34, 25},
      {0, 35, 19},
      {0, 14, 36},
      {0, 24, 37},
      {0, 2, 38},
      {0, 11, 10},
      {0, 40, 39},
      {0, 5, 16},
      {0, 42, 41},
      {0, 7, 43},
      {0, 3, 44},
      {0, 46, 45},
      {0, 48, 47},
      {0, 50, 49},
    }));

    codecs.emplace(SpvOpExtInst, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(45, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393303, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 19, 13},
      {0, 6, 24},
      {0, 16, 25},
      {0, 17, 26},
      {0, 27, 12},
      {0, 28, 20},
      {0, 9, 29},
      {0, 23, 14},
      {0, 22, 15},
      {0, 11, 30},
      {0, 31, 21},
      {0, 4, 18},
      {0, 3, 32},
      {0, 5, 33},
      {0, 10, 34},
      {0, 8, 35},
      {0, 36, 1},
      {0, 2, 37},
      {0, 39, 38},
      {0, 7, 40},
      {0, 42, 41},
      {0, 44, 43},
    }));

    codecs.emplace(SpvOpVectorTimesScalar, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(47, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393303, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 19, 20},
      {0, 4, 25},
      {0, 13, 26},
      {0, 15, 27},
      {0, 21, 23},
      {0, 28, 8},
      {0, 29, 12},
      {0, 14, 17},
      {0, 18, 30},
      {0, 24, 3},
      {0, 10, 22},
      {0, 11, 31},
      {0, 32, 16},
      {0, 34, 33},
      {0, 35, 5},
      {0, 36, 6},
      {0, 37, 2},
      {0, 9, 38},
      {0, 39, 7},
      {0, 40, 1},
      {0, 42, 41},
      {0, 44, 43},
      {0, 46, 45},
    }));

    codecs.emplace(SpvOpVectorShuffle, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(29, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262225, 0, 0},
      {262231, 0, 0},
      {262273, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327745, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {458831, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 11, 10},
      {0, 9, 16},
      {0, 7, 17},
      {0, 8, 18},
      {0, 19, 13},
      {0, 20, 6},
      {0, 5, 21},
      {0, 15, 22},
      {0, 3, 23},
      {0, 12, 2},
      {0, 25, 24},
      {0, 26, 1},
      {0, 4, 27},
      {0, 28, 14},
    }));

    codecs.emplace(SpvOpImageSampleImplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(47, {
      {0, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262280, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327745, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393281, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393298, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 18},
      {0, 1, 25},
      {0, 24, 23},
      {0, 16, 12},
      {0, 4, 26},
      {0, 6, 27},
      {0, 2, 28},
      {0, 20, 19},
      {0, 9, 29},
      {0, 3, 17},
      {0, 30, 10},
      {0, 21, 31},
      {0, 13, 7},
      {0, 8, 15},
      {0, 33, 32},
      {0, 34, 14},
      {0, 11, 35},
      {0, 37, 36},
      {0, 39, 38},
      {0, 22, 40},
      {0, 42, 41},
      {0, 44, 43},
      {0, 46, 45},
    }));

    codecs.emplace(SpvOpDot, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(49, {
      {0, 0, 0},
      {65785, 0, 0},
      {131134, 0, 0},
      {196669, 0, 0},
      {262209, 0, 0},
      {262221, 0, 0},
      {262224, 0, 0},
      {262225, 0, 0},
      {262230, 0, 0},
      {262273, 0, 0},
      {262275, 0, 0},
      {262277, 0, 0},
      {262286, 0, 0},
      {262292, 0, 0},
      {327692, 0, 0},
      {327760, 0, 0},
      {327761, 0, 0},
      {327762, 0, 0},
      {393228, 0, 0},
      {393295, 0, 0},
      {393296, 0, 0},
      {393298, 0, 0},
      {458764, 0, 0},
      {458831, 0, 0},
      {524367, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 9, 13},
      {0, 26, 10},
      {0, 15, 27},
      {0, 18, 22},
      {0, 29, 28},
      {0, 5, 30},
      {0, 6, 20},
      {0, 31, 12},
      {0, 11, 2},
      {0, 1, 24},
      {0, 14, 25},
      {0, 33, 32},
      {0, 23, 34},
      {0, 36, 35},
      {0, 17, 16},
      {0, 37, 19},
      {0, 38, 21},
      {0, 8, 4},
      {0, 40, 39},
      {0, 42, 41},
      {0, 44, 43},
      {0, 3, 7},
      {0, 46, 45},
      {0, 48, 47},
    }));

    codecs.emplace(SpvOpCompositeInsert, std::move(codec));
  }

  return codecs;
}

std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>>
GetLiteralStringHuffmanCodecs() {
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>(7, {
      {"", 0, 0},
      {"MainPs", 0, 0},
      {"MainVs", 0, 0},
      {"kMarkvNoneOfTheAbove", 0, 0},
      {"main", 0, 0},
      {"", 2, 3},
      {"", 1, 5},
      {"", 4, 6},
    }));

    codecs.emplace(SpvOpEntryPoint, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>(3, {
      {"", 0, 0},
      {"GLSL.std.450", 0, 0},
      {"kMarkvNoneOfTheAbove", 0, 0},
      {"", 1, 2},
    }));

    codecs.emplace(SpvOpExtInstImport, std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetNonIdWordHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(23, {
      {0, 0, 0},
      {4, 0, 0},
      {8, 0, 0},
      {26, 0, 0},
      {31, 0, 0},
      {40, 0, 0},
      {43, 0, 0},
      {46, 0, 0},
      {49, 0, 0},
      {68, 0, 0},
      {69, 0, 0},
      {71, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 11, 8},
      {0, 1, 9},
      {0, 13, 4},
      {0, 14, 2},
      {0, 15, 12},
      {0, 16, 3},
      {0, 18, 17},
      {0, 7, 10},
      {0, 5, 6},
      {0, 20, 19},
      {0, 22, 21},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {1, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {0, 0, 0},
      {4, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 3},
      {0, 2, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpEntryPoint, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {7, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExecutionMode, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {18, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 2},
      {0, 6, 5},
      {0, 7, 1},
      {0, 3, 8},
      {0, 10, 9},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExecutionMode, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {1, 0, 0},
      {32, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 1, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCapability, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {32, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 1, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {32, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFloat, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 4},
      {0, 1, 5},
      {0, 6, 3},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 4},
      {0, 2, 5},
      {0, 3, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeMatrix, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 4},
      {0, 2, 5},
      {0, 1, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 1, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {1, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 7},
      {0, 6, 8},
      {0, 1, 4},
      {0, 2, 9},
      {0, 10, 3},
      {0, 12, 11},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {8, 0, 0},
      {10, 0, 0},
      {981668463, 0, 0},
      {1055437881, 0, 0},
      {1056964608, 0, 0},
      {1065353216, 0, 0},
      {1073741824, 0, 0},
      {3212836864, 0, 0},
      {3332128768, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 10, 11},
      {0, 9, 17},
      {0, 7, 8},
      {0, 6, 16},
      {0, 5, 12},
      {0, 20, 19},
      {0, 21, 15},
      {0, 22, 13},
      {0, 4, 14},
      {0, 2, 23},
      {0, 3, 24},
      {0, 26, 25},
      {0, 1, 27},
      {0, 29, 28},
      {0, 31, 30},
      {0, 33, 32},
      {0, 18, 34},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 7},
      {0, 4, 8},
      {0, 9, 2},
      {0, 1, 5},
      {0, 10, 6},
      {0, 12, 11},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {0, 0, 0},
      {2, 0, 0},
      {6, 0, 0},
      {11, 0, 0},
      {30, 0, 0},
      {33, 0, 0},
      {34, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 8},
      {0, 9, 1},
      {0, 3, 10},
      {0, 6, 11},
      {0, 12, 2},
      {0, 7, 5},
      {0, 14, 13},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(25, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {8, 0, 0},
      {15, 0, 0},
      {16, 0, 0},
      {64, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 10, 9},
      {0, 14, 8},
      {0, 7, 12},
      {0, 5, 6},
      {0, 15, 11},
      {0, 13, 4},
      {0, 16, 3},
      {0, 18, 17},
      {0, 2, 19},
      {0, 21, 20},
      {0, 23, 22},
      {0, 1, 24},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(73, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {8, 0, 0},
      {9, 0, 0},
      {10, 0, 0},
      {11, 0, 0},
      {12, 0, 0},
      {13, 0, 0},
      {14, 0, 0},
      {15, 0, 0},
      {16, 0, 0},
      {17, 0, 0},
      {18, 0, 0},
      {19, 0, 0},
      {20, 0, 0},
      {21, 0, 0},
      {22, 0, 0},
      {23, 0, 0},
      {24, 0, 0},
      {25, 0, 0},
      {26, 0, 0},
      {27, 0, 0},
      {28, 0, 0},
      {29, 0, 0},
      {30, 0, 0},
      {31, 0, 0},
      {32, 0, 0},
      {33, 0, 0},
      {34, 0, 0},
      {37, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 34, 35},
      {0, 32, 33},
      {0, 30, 31},
      {0, 27, 29},
      {0, 26, 28},
      {0, 23, 25},
      {0, 36, 22},
      {0, 39, 38},
      {0, 41, 40},
      {0, 21, 42},
      {0, 19, 20},
      {0, 17, 18},
      {0, 14, 15},
      {0, 12, 10},
      {0, 16, 13},
      {0, 9, 11},
      {0, 7, 8},
      {0, 6, 5},
      {0, 24, 37},
      {0, 44, 43},
      {0, 3, 4},
      {0, 45, 2},
      {0, 1, 46},
      {0, 48, 47},
      {0, 50, 49},
      {0, 52, 51},
      {0, 54, 53},
      {0, 56, 55},
      {0, 58, 57},
      {0, 60, 59},
      {0, 62, 61},
      {0, 64, 63},
      {0, 66, 65},
      {0, 68, 67},
      {0, 70, 69},
      {0, 72, 71},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {4, 0, 0},
      {7, 0, 0},
      {35, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 4},
      {0, 5, 2},
      {0, 3, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(59, {
      {0, 0, 0},
      {0, 0, 0},
      {16, 0, 0},
      {28, 0, 0},
      {32, 0, 0},
      {40, 0, 0},
      {44, 0, 0},
      {48, 0, 0},
      {60, 0, 0},
      {64, 0, 0},
      {76, 0, 0},
      {80, 0, 0},
      {92, 0, 0},
      {96, 0, 0},
      {108, 0, 0},
      {112, 0, 0},
      {120, 0, 0},
      {128, 0, 0},
      {140, 0, 0},
      {144, 0, 0},
      {148, 0, 0},
      {152, 0, 0},
      {156, 0, 0},
      {160, 0, 0},
      {176, 0, 0},
      {192, 0, 0},
      {204, 0, 0},
      {208, 0, 0},
      {224, 0, 0},
      {256, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 16, 14},
      {0, 12, 20},
      {0, 6, 26},
      {0, 3, 5},
      {0, 27, 21},
      {0, 8, 28},
      {0, 10, 22},
      {0, 29, 18},
      {0, 32, 31},
      {0, 19, 24},
      {0, 15, 23},
      {0, 13, 11},
      {0, 4, 7},
      {0, 34, 33},
      {0, 36, 35},
      {0, 17, 25},
      {0, 37, 9},
      {0, 39, 38},
      {0, 40, 1},
      {0, 42, 41},
      {0, 44, 43},
      {0, 46, 45},
      {0, 2, 47},
      {0, 49, 48},
      {0, 51, 50},
      {0, 53, 52},
      {0, 55, 54},
      {0, 30, 56},
      {0, 58, 57},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 6},
      {0, 4, 7},
      {0, 8, 3},
      {0, 9, 5},
      {0, 1, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 7},
      {0, 8, 5},
      {0, 9, 1},
      {0, 4, 10},
      {0, 11, 6},
      {0, 2, 12},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 6, 8},
      {0, 5, 2},
      {0, 10, 9},
      {0, 1, 4},
      {0, 12, 11},
      {0, 7, 13},
      {0, 3, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 7, 6},
      {0, 8, 3},
      {0, 9, 2},
      {0, 5, 1},
      {0, 11, 10},
      {0, 4, 12},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(27, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {8, 0, 0},
      {9, 0, 0},
      {10, 0, 0},
      {12, 0, 0},
      {16, 0, 0},
      {23, 0, 0},
      {24, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 11, 9},
      {0, 13, 10},
      {0, 7, 8},
      {0, 15, 5},
      {0, 17, 16},
      {0, 12, 6},
      {0, 19, 18},
      {0, 21, 20},
      {0, 14, 4},
      {0, 3, 22},
      {0, 23, 2},
      {0, 24, 1},
      {0, 26, 25},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 2},
      {0, 5, 4},
      {0, 6, 1},
      {0, 8, 7},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 5},
      {0, 3, 2},
      {0, 6, 4},
      {0, 8, 7},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(21, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {4, 0, 0},
      {5, 0, 0},
      {6, 0, 0},
      {7, 0, 0},
      {8, 0, 0},
      {9, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 10, 11},
      {0, 9, 12},
      {0, 7, 5},
      {0, 8, 6},
      {0, 4, 13},
      {0, 15, 14},
      {0, 16, 3},
      {0, 17, 2},
      {0, 18, 1},
      {0, 20, 19},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {0, 0, 0},
      {1, 0, 0},
      {2, 0, 0},
      {3, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 5},
      {0, 2, 6},
      {0, 7, 1},
      {0, 4, 8},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {1, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {2, 0, 0},
      {10, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 1, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {2, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {0, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSelectionMerge, 1), std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetIdDescriptorHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 5},
      {0, 4, 6},
      {0, 1, 7},
      {0, 2, 8},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {2161102232, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {4228502127, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {139011596, 0, 0},
      {810488476, 0, 0},
      {870594305, 0, 0},
      {1742737136, 0, 0},
      {2096388952, 0, 0},
      {2855506940, 0, 0},
      {3044188332, 0, 0},
      {3487022798, 0, 0},
      {3701632935, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 9},
      {0, 2, 8},
      {0, 4, 7},
      {0, 11, 5},
      {0, 13, 12},
      {0, 14, 1},
      {0, 6, 15},
      {0, 17, 16},
      {0, 10, 18},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {139011596, 0, 0},
      {296981500, 0, 0},
      {1367301635, 0, 0},
      {2855506940, 0, 0},
      {3233393284, 0, 0},
      {3251128023, 0, 0},
      {3582002820, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 7},
      {0, 1, 3},
      {0, 2, 9},
      {0, 6, 10},
      {0, 12, 11},
      {0, 4, 13},
      {0, 8, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {296981500, 0, 0},
      {508217552, 0, 0},
      {2683080096, 0, 0},
      {3547456240, 0, 0},
      {3753486980, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 4},
      {0, 5, 2},
      {0, 8, 7},
      {0, 1, 9},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(65, {
      {0, 0, 0},
      {135486769, 0, 0},
      {440421571, 0, 0},
      {450406196, 0, 0},
      {503094540, 0, 0},
      {543621065, 0, 0},
      {827698488, 0, 0},
      {907126242, 0, 0},
      {908777857, 0, 0},
      {910429472, 0, 0},
      {1294403159, 0, 0},
      {1296054774, 0, 0},
      {1297706389, 0, 0},
      {1322549027, 0, 0},
      {1784441183, 0, 0},
      {2080953106, 0, 0},
      {2194691858, 0, 0},
      {2448331885, 0, 0},
      {2468230023, 0, 0},
      {2547657777, 0, 0},
      {2549309392, 0, 0},
      {2550961007, 0, 0},
      {2934934694, 0, 0},
      {2936586309, 0, 0},
      {2938237924, 0, 0},
      {3094180193, 0, 0},
      {3094857332, 0, 0},
      {3095831808, 0, 0},
      {3183924418, 0, 0},
      {3561562003, 0, 0},
      {3563213618, 0, 0},
      {3564865233, 0, 0},
      {4028622909, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 23, 3},
      {0, 18, 24},
      {0, 14, 22},
      {0, 17, 9},
      {0, 1, 26},
      {0, 4, 15},
      {0, 31, 25},
      {0, 8, 7},
      {0, 34, 2},
      {0, 12, 30},
      {0, 36, 35},
      {0, 13, 11},
      {0, 6, 10},
      {0, 5, 37},
      {0, 38, 27},
      {0, 40, 39},
      {0, 42, 41},
      {0, 16, 28},
      {0, 44, 43},
      {0, 21, 45},
      {0, 20, 46},
      {0, 47, 32},
      {0, 48, 19},
      {0, 29, 49},
      {0, 51, 50},
      {0, 53, 52},
      {0, 55, 54},
      {0, 57, 56},
      {0, 58, 33},
      {0, 60, 59},
      {0, 62, 61},
      {0, 64, 63},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {119981689, 0, 0},
      {162255877, 0, 0},
      {679771963, 0, 0},
      {1154919607, 0, 0},
      {1343794461, 0, 0},
      {1674803691, 0, 0},
      {1951208733, 0, 0},
      {2263349224, 0, 0},
      {2320303498, 0, 0},
      {2924146124, 0, 0},
      {2984325996, 0, 0},
      {3334207724, 0, 0},
      {3410158390, 0, 0},
      {3489360962, 0, 0},
      {3866587616, 0, 0},
      {3868239231, 0, 0},
      {3869890846, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 14, 15},
      {0, 6, 1},
      {0, 8, 13},
      {0, 19, 5},
      {0, 20, 16},
      {0, 11, 2},
      {0, 22, 21},
      {0, 17, 10},
      {0, 4, 23},
      {0, 25, 24},
      {0, 7, 26},
      {0, 9, 27},
      {0, 28, 3},
      {0, 29, 18},
      {0, 30, 12},
      {0, 32, 31},
      {0, 34, 33},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {789872778, 0, 0},
      {1951208733, 0, 0},
      {2430404313, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 4},
      {0, 1, 5},
      {0, 2, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(45, {
      {0, 0, 0},
      {142465290, 0, 0},
      {144116905, 0, 0},
      {210116709, 0, 0},
      {296981500, 0, 0},
      {529742207, 0, 0},
      {959681532, 0, 0},
      {1156369516, 0, 0},
      {1158021131, 0, 0},
      {1543646433, 0, 0},
      {1782996825, 0, 0},
      {1784648440, 0, 0},
      {2170273742, 0, 0},
      {2321729979, 0, 0},
      {2444465148, 0, 0},
      {2524697596, 0, 0},
      {2557550659, 0, 0},
      {2796901051, 0, 0},
      {2798552666, 0, 0},
      {2855506940, 0, 0},
      {3184177968, 0, 0},
      {3810805277, 0, 0},
      {3929248764, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 16, 13},
      {0, 9, 15},
      {0, 5, 20},
      {0, 14, 8},
      {0, 12, 11},
      {0, 2, 3},
      {0, 24, 21},
      {0, 1, 7},
      {0, 17, 10},
      {0, 22, 25},
      {0, 27, 26},
      {0, 18, 6},
      {0, 19, 28},
      {0, 29, 4},
      {0, 31, 30},
      {0, 33, 32},
      {0, 35, 34},
      {0, 37, 36},
      {0, 39, 38},
      {0, 41, 40},
      {0, 43, 42},
      {0, 23, 44},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1247793383, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 5},
      {0, 4, 6},
      {0, 1, 3},
      {0, 8, 7},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(59, {
      {0, 0, 0},
      {15502752, 0, 0},
      {139011596, 0, 0},
      {249378857, 0, 0},
      {251209228, 0, 0},
      {503145996, 0, 0},
      {1325348861, 0, 0},
      {1558001705, 0, 0},
      {1646147798, 0, 0},
      {1679946323, 0, 0},
      {1766401548, 0, 0},
      {1992893964, 0, 0},
      {2123388694, 0, 0},
      {2162986400, 0, 0},
      {2580096524, 0, 0},
      {2598189097, 0, 0},
      {2683080096, 0, 0},
      {2698156268, 0, 0},
      {3133016299, 0, 0},
      {3251128023, 0, 0},
      {3504158761, 0, 0},
      {3538592682, 0, 0},
      {3540244297, 0, 0},
      {3541895912, 0, 0},
      {3570219049, 0, 0},
      {3653838348, 0, 0},
      {3764205609, 0, 0},
      {3882634684, 0, 0},
      {3913885196, 0, 0},
      {4243119782, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 7, 4},
      {0, 24, 15},
      {0, 11, 13},
      {0, 5, 14},
      {0, 6, 3},
      {0, 9, 8},
      {0, 17, 12},
      {0, 27, 18},
      {0, 1, 29},
      {0, 21, 31},
      {0, 23, 22},
      {0, 33, 32},
      {0, 10, 34},
      {0, 25, 28},
      {0, 36, 35},
      {0, 38, 37},
      {0, 16, 39},
      {0, 40, 26},
      {0, 41, 19},
      {0, 20, 42},
      {0, 2, 43},
      {0, 45, 44},
      {0, 47, 46},
      {0, 49, 48},
      {0, 50, 30},
      {0, 52, 51},
      {0, 54, 53},
      {0, 56, 55},
      {0, 58, 57},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(41, {
      {0, 0, 0},
      {142465290, 0, 0},
      {158160339, 0, 0},
      {169135842, 0, 0},
      {210116709, 0, 0},
      {296981500, 0, 0},
      {910398460, 0, 0},
      {959681532, 0, 0},
      {1039111164, 0, 0},
      {1087394637, 0, 0},
      {1156369516, 0, 0},
      {2444465148, 0, 0},
      {2732195517, 0, 0},
      {2796901051, 0, 0},
      {2855506940, 0, 0},
      {3202349435, 0, 0},
      {3362723943, 0, 0},
      {3712763835, 0, 0},
      {3929248764, 0, 0},
      {4016096296, 0, 0},
      {4248015868, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 20, 6},
      {0, 7, 8},
      {0, 3, 2},
      {0, 12, 9},
      {0, 16, 15},
      {0, 19, 17},
      {0, 10, 22},
      {0, 1, 13},
      {0, 23, 11},
      {0, 25, 24},
      {0, 27, 26},
      {0, 28, 18},
      {0, 29, 4},
      {0, 31, 30},
      {0, 33, 32},
      {0, 21, 34},
      {0, 35, 5},
      {0, 37, 36},
      {0, 38, 14},
      {0, 40, 39},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(39, {
      {0, 0, 0},
      {52882140, 0, 0},
      {210116709, 0, 0},
      {296981500, 0, 0},
      {385229009, 0, 0},
      {910398460, 0, 0},
      {959681532, 0, 0},
      {1031290113, 0, 0},
      {1039111164, 0, 0},
      {1172110445, 0, 0},
      {1622381564, 0, 0},
      {1782996825, 0, 0},
      {1971252067, 0, 0},
      {2444465148, 0, 0},
      {2490492987, 0, 0},
      {2678954464, 0, 0},
      {2855506940, 0, 0},
      {3912967080, 0, 0},
      {3929248764, 0, 0},
      {4248015868, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 10},
      {0, 8, 19},
      {0, 6, 21},
      {0, 4, 1},
      {0, 9, 7},
      {0, 14, 12},
      {0, 17, 15},
      {0, 13, 22},
      {0, 24, 23},
      {0, 26, 25},
      {0, 18, 27},
      {0, 28, 2},
      {0, 29, 20},
      {0, 31, 30},
      {0, 32, 11},
      {0, 33, 3},
      {0, 35, 34},
      {0, 36, 16},
      {0, 38, 37},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {210116709, 0, 0},
      {296981500, 0, 0},
      {615748604, 0, 0},
      {910398460, 0, 0},
      {959681532, 0, 0},
      {1039111164, 0, 0},
      {1543672828, 0, 0},
      {1612361408, 0, 0},
      {2100532220, 0, 0},
      {2326636627, 0, 0},
      {2444465148, 0, 0},
      {2524697596, 0, 0},
      {2763232252, 0, 0},
      {2855506940, 0, 0},
      {3929248764, 0, 0},
      {4172568578, 0, 0},
      {4248015868, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 7, 3},
      {0, 13, 9},
      {0, 10, 8},
      {0, 4, 16},
      {0, 19, 17},
      {0, 20, 6},
      {0, 21, 12},
      {0, 22, 18},
      {0, 23, 5},
      {0, 24, 15},
      {0, 26, 25},
      {0, 27, 11},
      {0, 1, 28},
      {0, 30, 29},
      {0, 14, 31},
      {0, 2, 32},
      {0, 34, 33},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(33, {
      {0, 0, 0},
      {296981500, 0, 0},
      {615748604, 0, 0},
      {959681532, 0, 0},
      {1039111164, 0, 0},
      {1450415100, 0, 0},
      {1543672828, 0, 0},
      {1939359710, 0, 0},
      {2100532220, 0, 0},
      {2113115132, 0, 0},
      {2326636627, 0, 0},
      {2444465148, 0, 0},
      {2763232252, 0, 0},
      {2855506940, 0, 0},
      {3929248764, 0, 0},
      {4172568578, 0, 0},
      {4248015868, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 17},
      {0, 11, 3},
      {0, 16, 2},
      {0, 8, 6},
      {0, 18, 12},
      {0, 10, 7},
      {0, 9, 15},
      {0, 19, 14},
      {0, 4, 20},
      {0, 22, 21},
      {0, 24, 23},
      {0, 26, 25},
      {0, 27, 1},
      {0, 29, 28},
      {0, 31, 30},
      {0, 32, 13},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(37, {
      {0, 0, 0},
      {543621065, 0, 0},
      {827698488, 0, 0},
      {1294403159, 0, 0},
      {1296054774, 0, 0},
      {1297706389, 0, 0},
      {1322549027, 0, 0},
      {2194691858, 0, 0},
      {2468230023, 0, 0},
      {2547657777, 0, 0},
      {2549309392, 0, 0},
      {2550961007, 0, 0},
      {2934934694, 0, 0},
      {2936586309, 0, 0},
      {2938237924, 0, 0},
      {3095831808, 0, 0},
      {3183924418, 0, 0},
      {3561562003, 0, 0},
      {4028622909, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 16, 6},
      {0, 18, 10},
      {0, 13, 14},
      {0, 11, 20},
      {0, 21, 5},
      {0, 22, 2},
      {0, 3, 23},
      {0, 7, 15},
      {0, 1, 24},
      {0, 8, 17},
      {0, 9, 12},
      {0, 4, 25},
      {0, 19, 26},
      {0, 28, 27},
      {0, 30, 29},
      {0, 32, 31},
      {0, 34, 33},
      {0, 36, 35},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(37, {
      {0, 0, 0},
      {37459569, 0, 0},
      {137840602, 0, 0},
      {565334834, 0, 0},
      {625975427, 0, 0},
      {630964591, 0, 0},
      {680016782, 0, 0},
      {1009983433, 0, 0},
      {1572088444, 0, 0},
      {1584774136, 0, 0},
      {1918481917, 0, 0},
      {2790624748, 0, 0},
      {3181646225, 0, 0},
      {3560665067, 0, 0},
      {3662767579, 0, 0},
      {4053789056, 0, 0},
      {4064212479, 0, 0},
      {4192247221, 0, 0},
      {4224872590, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 3},
      {0, 6, 15},
      {0, 2, 7},
      {0, 5, 20},
      {0, 21, 14},
      {0, 22, 18},
      {0, 10, 23},
      {0, 1, 16},
      {0, 9, 24},
      {0, 8, 17},
      {0, 12, 13},
      {0, 11, 25},
      {0, 19, 26},
      {0, 28, 27},
      {0, 30, 29},
      {0, 32, 31},
      {0, 34, 33},
      {0, 36, 35},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {162255877, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {2984325996, 0, 0},
      {3334207724, 0, 0},
      {3869890846, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 7, 5},
      {0, 8, 1},
      {0, 9, 4},
      {0, 2, 10},
      {0, 6, 11},
      {0, 3, 12},
      {0, 14, 13},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(37, {
      {0, 0, 0},
      {543558236, 0, 0},
      {810488476, 0, 0},
      {870594305, 0, 0},
      {883854656, 0, 0},
      {1570165302, 0, 0},
      {1684282922, 0, 0},
      {1901166356, 0, 0},
      {1949759310, 0, 0},
      {2087004702, 0, 0},
      {2096388952, 0, 0},
      {2517964682, 0, 0},
      {2622612602, 0, 0},
      {2970183398, 0, 0},
      {3091876332, 0, 0},
      {3187066832, 0, 0},
      {3496407048, 0, 0},
      {3570411982, 0, 0},
      {3692647551, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 14, 18},
      {0, 3, 7},
      {0, 5, 9},
      {0, 6, 13},
      {0, 21, 20},
      {0, 2, 11},
      {0, 15, 22},
      {0, 16, 23},
      {0, 4, 1},
      {0, 12, 24},
      {0, 17, 8},
      {0, 26, 25},
      {0, 27, 10},
      {0, 29, 28},
      {0, 31, 30},
      {0, 33, 32},
      {0, 35, 34},
      {0, 36, 19},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(37, {
      {0, 0, 0},
      {37459569, 0, 0},
      {137840602, 0, 0},
      {522971108, 0, 0},
      {630964591, 0, 0},
      {1572088444, 0, 0},
      {1584774136, 0, 0},
      {1918481917, 0, 0},
      {1957218950, 0, 0},
      {2313593054, 0, 0},
      {2790624748, 0, 0},
      {2838165089, 0, 0},
      {3181646225, 0, 0},
      {3364388739, 0, 0},
      {3560665067, 0, 0},
      {3662767579, 0, 0},
      {4064212479, 0, 0},
      {4224872590, 0, 0},
      {4239834800, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 9, 11},
      {0, 13, 8},
      {0, 15, 18},
      {0, 5, 2},
      {0, 21, 20},
      {0, 17, 4},
      {0, 3, 22},
      {0, 14, 23},
      {0, 1, 16},
      {0, 7, 24},
      {0, 12, 6},
      {0, 26, 25},
      {0, 27, 10},
      {0, 29, 28},
      {0, 31, 30},
      {0, 33, 32},
      {0, 35, 34},
      {0, 36, 19},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {137840602, 0, 0},
      {1009983433, 0, 0},
      {1572088444, 0, 0},
      {1918481917, 0, 0},
      {2790624748, 0, 0},
      {3560665067, 0, 0},
      {3662767579, 0, 0},
      {4192247221, 0, 0},
      {4224872590, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
      {0, 11, 7},
      {0, 8, 9},
      {0, 4, 3},
      {0, 12, 6},
      {0, 5, 13},
      {0, 15, 14},
      {0, 16, 10},
      {0, 18, 17},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(25, {
      {0, 0, 0},
      {177111659, 0, 0},
      {296981500, 0, 0},
      {408465899, 0, 0},
      {2055836767, 0, 0},
      {2087004702, 0, 0},
      {2096388952, 0, 0},
      {2622612602, 0, 0},
      {2855506940, 0, 0},
      {2959147533, 0, 0},
      {3187066832, 0, 0},
      {3570411982, 0, 0},
      {3831290364, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
      {0, 12, 11},
      {0, 9, 3},
      {0, 7, 4},
      {0, 14, 8},
      {0, 10, 6},
      {0, 5, 15},
      {0, 17, 16},
      {0, 19, 18},
      {0, 21, 20},
      {0, 23, 22},
      {0, 13, 24},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(25, {
      {0, 0, 0},
      {440421571, 0, 0},
      {827698488, 0, 0},
      {907126242, 0, 0},
      {908777857, 0, 0},
      {910429472, 0, 0},
      {1294403159, 0, 0},
      {1296054774, 0, 0},
      {1297706389, 0, 0},
      {2080953106, 0, 0},
      {2468230023, 0, 0},
      {2547657777, 0, 0},
      {3561562003, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 7, 5},
      {0, 12, 11},
      {0, 9, 8},
      {0, 13, 14},
      {0, 15, 4},
      {0, 17, 16},
      {0, 18, 3},
      {0, 19, 10},
      {0, 6, 1},
      {0, 21, 20},
      {0, 22, 2},
      {0, 24, 23},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(29, {
      {0, 0, 0},
      {1079999262, 0, 0},
      {2311941439, 0, 0},
      {2313593054, 0, 0},
      {2838165089, 0, 0},
      {2839816704, 0, 0},
      {2841468319, 0, 0},
      {3364388739, 0, 0},
      {3366040354, 0, 0},
      {3367691969, 0, 0},
      {3369343584, 0, 0},
      {4239834800, 0, 0},
      {4241486415, 0, 0},
      {4243138030, 0, 0},
      {4244789645, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 9, 10},
      {0, 14, 12},
      {0, 6, 13},
      {0, 8, 1},
      {0, 2, 5},
      {0, 3, 4},
      {0, 17, 16},
      {0, 11, 18},
      {0, 7, 19},
      {0, 21, 20},
      {0, 23, 22},
      {0, 25, 24},
      {0, 27, 26},
      {0, 15, 28},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(49, {
      {0, 0, 0},
      {112745085, 0, 0},
      {116376005, 0, 0},
      {400248103, 0, 0},
      {406044930, 0, 0},
      {522971108, 0, 0},
      {625975427, 0, 0},
      {680016782, 0, 0},
      {1009983433, 0, 0},
      {1062250709, 0, 0},
      {1410849099, 0, 0},
      {1918481917, 0, 0},
      {2190437442, 0, 0},
      {2790624748, 0, 0},
      {2879917723, 0, 0},
      {2882994691, 0, 0},
      {3181646225, 0, 0},
      {3263901372, 0, 0},
      {3390051757, 0, 0},
      {3560665067, 0, 0},
      {3662767579, 0, 0},
      {3717523241, 0, 0},
      {3945795573, 0, 0},
      {4101009465, 0, 0},
      {4290024976, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 23},
      {0, 8, 22},
      {0, 3, 21},
      {0, 9, 5},
      {0, 2, 10},
      {0, 26, 17},
      {0, 27, 14},
      {0, 16, 4},
      {0, 29, 28},
      {0, 12, 30},
      {0, 15, 31},
      {0, 32, 18},
      {0, 6, 7},
      {0, 20, 33},
      {0, 24, 34},
      {0, 35, 13},
      {0, 19, 36},
      {0, 11, 37},
      {0, 39, 38},
      {0, 41, 40},
      {0, 43, 42},
      {0, 44, 25},
      {0, 46, 45},
      {0, 48, 47},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {142465290, 0, 0},
      {144116905, 0, 0},
      {529742207, 0, 0},
      {1156369516, 0, 0},
      {1158021131, 0, 0},
      {1543646433, 0, 0},
      {1782996825, 0, 0},
      {1784648440, 0, 0},
      {1930923350, 0, 0},
      {2170273742, 0, 0},
      {2557550659, 0, 0},
      {2705477184, 0, 0},
      {2796901051, 0, 0},
      {2798552666, 0, 0},
      {3184177968, 0, 0},
      {3810805277, 0, 0},
      {4198082194, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 11, 9},
      {0, 6, 17},
      {0, 3, 19},
      {0, 15, 12},
      {0, 20, 16},
      {0, 10, 13},
      {0, 21, 4},
      {0, 1, 22},
      {0, 14, 23},
      {0, 5, 24},
      {0, 26, 25},
      {0, 2, 8},
      {0, 27, 18},
      {0, 7, 28},
      {0, 30, 29},
      {0, 32, 31},
      {0, 34, 33},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(41, {
      {0, 0, 0},
      {142465290, 0, 0},
      {144116905, 0, 0},
      {198967948, 0, 0},
      {529742207, 0, 0},
      {825595257, 0, 0},
      {1156369516, 0, 0},
      {1158021131, 0, 0},
      {1452222566, 0, 0},
      {1782996825, 0, 0},
      {1784648440, 0, 0},
      {1839499483, 0, 0},
      {2170273742, 0, 0},
      {2466126792, 0, 0},
      {2796901051, 0, 0},
      {2798552666, 0, 0},
      {3184177968, 0, 0},
      {3480031018, 0, 0},
      {3810805277, 0, 0},
      {4106658327, 0, 0},
      {4198082194, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 20},
      {0, 16, 13},
      {0, 11, 5},
      {0, 4, 17},
      {0, 19, 8},
      {0, 18, 12},
      {0, 23, 22},
      {0, 25, 24},
      {0, 27, 26},
      {0, 6, 28},
      {0, 14, 29},
      {0, 1, 30},
      {0, 15, 2},
      {0, 10, 31},
      {0, 9, 32},
      {0, 34, 33},
      {0, 35, 7},
      {0, 21, 36},
      {0, 38, 37},
      {0, 40, 39},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {144116905, 0, 0},
      {1158021131, 0, 0},
      {1784648440, 0, 0},
      {2798552666, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 5},
      {0, 4, 2},
      {0, 6, 3},
      {0, 8, 7},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {142465290, 0, 0},
      {1782996825, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
      {0, 4, 3},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {679771963, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 4},
      {0, 2, 5},
      {0, 6, 1},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(19, {
      {0, 0, 0},
      {177111659, 0, 0},
      {837715723, 0, 0},
      {1203545131, 0, 0},
      {1352628475, 0, 0},
      {1367301635, 0, 0},
      {2055836767, 0, 0},
      {2204920111, 0, 0},
      {3619787319, 0, 0},
      {3701632935, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 4},
      {0, 2, 7},
      {0, 11, 9},
      {0, 6, 12},
      {0, 8, 5},
      {0, 3, 13},
      {0, 15, 14},
      {0, 17, 16},
      {0, 10, 18},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(17, {
      {0, 0, 0},
      {488500848, 0, 0},
      {495107308, 0, 0},
      {2096388952, 0, 0},
      {2157103435, 0, 0},
      {2622612602, 0, 0},
      {3496407048, 0, 0},
      {3570411982, 0, 0},
      {3713290482, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 8},
      {0, 3, 4},
      {0, 10, 2},
      {0, 12, 11},
      {0, 6, 13},
      {0, 7, 5},
      {0, 15, 14},
      {0, 9, 16},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {495107308, 0, 0},
      {2096388952, 0, 0},
      {2622612602, 0, 0},
      {3496407048, 0, 0},
      {3570411982, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
      {0, 3, 7},
      {0, 8, 4},
      {0, 9, 5},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {679771963, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 4},
      {0, 3, 5},
      {0, 6, 1},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(5, {
      {0, 0, 0},
      {1319785741, 0, 0},
      {3753486980, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
      {0, 3, 4},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {296981500, 0, 0},
      {959681532, 0, 0},
      {1141965917, 0, 0},
      {2855506940, 0, 0},
      {3091876332, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 2},
      {0, 4, 3},
      {0, 1, 7},
      {0, 9, 8},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {269823086, 0, 0},
      {296981500, 0, 0},
      {959681532, 0, 0},
      {2219733501, 0, 0},
      {2855506940, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 4},
      {0, 7, 3},
      {0, 5, 2},
      {0, 9, 8},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {296981500, 0, 0},
      {959681532, 0, 0},
      {1227221002, 0, 0},
      {2855506940, 0, 0},
      {3692647551, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 1, 5},
      {0, 4, 7},
      {0, 9, 8},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {18776483, 0, 0},
      {296981500, 0, 0},
      {959681532, 0, 0},
      {2855506940, 0, 0},
      {3874089391, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 1},
      {0, 7, 3},
      {0, 4, 8},
      {0, 9, 2},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 5},
      {0, 1, 6},
      {0, 7, 4},
      {0, 2, 8},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(1, {
      {0, 0, 0},
      {1111111111111111111, 0, 0},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(17, {
      {0, 0, 0},
      {495107308, 0, 0},
      {850497536, 0, 0},
      {1584369690, 0, 0},
      {1890300748, 0, 0},
      {2043873558, 0, 0},
      {2338272340, 0, 0},
      {3312467582, 0, 0},
      {3504158761, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 8},
      {0, 10, 4},
      {0, 7, 3},
      {0, 6, 11},
      {0, 12, 5},
      {0, 13, 2},
      {0, 15, 14},
      {0, 9, 16},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {545678922, 0, 0},
      {630592085, 0, 0},
      {679771963, 0, 0},
      {899570100, 0, 0},
      {929101967, 0, 0},
      {1100599986, 0, 0},
      {1103903216, 0, 0},
      {1369578001, 0, 0},
      {2320303498, 0, 0},
      {2926633629, 0, 0},
      {3334207724, 0, 0},
      {3486057732, 0, 0},
      {3705139860, 0, 0},
      {3800912395, 0, 0},
      {3822983876, 0, 0},
      {4141567741, 0, 0},
      {4292991777, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 10, 2},
      {0, 12, 17},
      {0, 13, 7},
      {0, 20, 19},
      {0, 5, 16},
      {0, 15, 21},
      {0, 3, 22},
      {0, 23, 18},
      {0, 1, 9},
      {0, 8, 24},
      {0, 11, 14},
      {0, 4, 25},
      {0, 27, 26},
      {0, 6, 28},
      {0, 30, 29},
      {0, 32, 31},
      {0, 34, 33},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(1, {
      {0, 0, 0},
      {1111111111111111111, 0, 0},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {296981500, 0, 0},
      {2517964682, 0, 0},
      {2855506940, 0, 0},
      {3044188332, 0, 0},
      {3570411982, 0, 0},
      {3764205609, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 1},
      {0, 8, 6},
      {0, 9, 2},
      {0, 10, 3},
      {0, 11, 5},
      {0, 7, 12},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {371428004, 0, 0},
      {1543280290, 0, 0},
      {2162986400, 0, 0},
      {3808408202, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 4, 1},
      {0, 7, 6},
      {0, 5, 8},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {236660303, 0, 0},
      {488500848, 0, 0},
      {495107308, 0, 0},
      {1858116930, 0, 0},
      {2231688008, 0, 0},
      {2693892518, 0, 0},
      {3566035349, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 6, 5},
      {0, 4, 7},
      {0, 10, 9},
      {0, 1, 2},
      {0, 12, 11},
      {0, 13, 3},
      {0, 8, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {883854656, 0, 0},
      {2036361232, 0, 0},
      {2356768706, 0, 0},
      {2637132451, 0, 0},
      {3237903670, 0, 0},
      {3829682756, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 7},
      {0, 8, 6},
      {0, 3, 4},
      {0, 10, 9},
      {0, 2, 11},
      {0, 12, 1},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {646282397, 0, 0},
      {1352628475, 0, 0},
      {1543798545, 0, 0},
      {1545450160, 0, 0},
      {2517964682, 0, 0},
      {2532518896, 0, 0},
      {3619787319, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 6},
      {0, 1, 4},
      {0, 10, 9},
      {0, 5, 2},
      {0, 12, 11},
      {0, 13, 7},
      {0, 8, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {2855506940, 0, 0},
      {3266548732, 0, 0},
      {3732640764, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
      {0, 5, 4},
      {0, 3, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 5},
      {0, 2, 6},
      {0, 1, 3},
      {0, 8, 7},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(1, {
      {0, 0, 0},
      {1111111111111111111, 0, 0},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {296981500, 0, 0},
      {959681532, 0, 0},
      {1570165302, 0, 0},
      {2096388952, 0, 0},
      {3653838348, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 5},
      {0, 3, 2},
      {0, 8, 7},
      {0, 9, 4},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {643418617, 0, 0},
      {959681532, 0, 0},
      {1092948665, 0, 0},
      {2517964682, 0, 0},
      {2683080096, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 3},
      {0, 4, 1},
      {0, 8, 7},
      {0, 9, 2},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 5},
      {0, 4, 6},
      {0, 1, 7},
      {0, 2, 8},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(17, {
      {0, 0, 0},
      {615982737, 0, 0},
      {1139547465, 0, 0},
      {1178317551, 0, 0},
      {2330636993, 0, 0},
      {2589449658, 0, 0},
      {3579593979, 0, 0},
      {3730093054, 0, 0},
      {3944781937, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 6},
      {0, 5, 7},
      {0, 8, 1},
      {0, 10, 4},
      {0, 11, 2},
      {0, 13, 12},
      {0, 15, 14},
      {0, 9, 16},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(23, {
      {0, 0, 0},
      {249378857, 0, 0},
      {296981500, 0, 0},
      {1203545131, 0, 0},
      {1265796414, 0, 0},
      {1319785741, 0, 0},
      {2855506940, 0, 0},
      {3091876332, 0, 0},
      {3187066832, 0, 0},
      {3508792859, 0, 0},
      {3619787319, 0, 0},
      {3653838348, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 9},
      {0, 4, 1},
      {0, 7, 10},
      {0, 13, 8},
      {0, 3, 14},
      {0, 6, 15},
      {0, 11, 16},
      {0, 18, 17},
      {0, 20, 19},
      {0, 21, 2},
      {0, 22, 12},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(33, {
      {0, 0, 0},
      {296981500, 0, 0},
      {443558693, 0, 0},
      {1203545131, 0, 0},
      {1265796414, 0, 0},
      {1319785741, 0, 0},
      {1558001705, 0, 0},
      {1955104493, 0, 0},
      {2234361374, 0, 0},
      {2598189097, 0, 0},
      {2775815164, 0, 0},
      {3244209297, 0, 0},
      {3753486980, 0, 0},
      {3886529747, 0, 0},
      {4069810315, 0, 0},
      {4164704452, 0, 0},
      {4273793488, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 10, 7},
      {0, 15, 13},
      {0, 8, 16},
      {0, 6, 2},
      {0, 3, 9},
      {0, 14, 5},
      {0, 11, 18},
      {0, 20, 19},
      {0, 22, 21},
      {0, 23, 12},
      {0, 24, 1},
      {0, 25, 4},
      {0, 27, 26},
      {0, 29, 28},
      {0, 31, 30},
      {0, 17, 32},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(9, {
      {0, 0, 0},
      {679771963, 0, 0},
      {1951208733, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 5},
      {0, 3, 6},
      {0, 1, 7},
      {0, 8, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(1, {
      {0, 0, 0},
      {1111111111111111111, 0, 0},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {810488476, 0, 0},
      {1054461787, 0, 0},
      {1158929937, 0, 0},
      {1203545131, 0, 0},
      {2096388952, 0, 0},
      {3929248764, 0, 0},
      {4008405264, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 2},
      {0, 7, 6},
      {0, 1, 4},
      {0, 5, 9},
      {0, 11, 10},
      {0, 13, 12},
      {0, 8, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(13, {
      {0, 0, 0},
      {171307615, 0, 0},
      {593829839, 0, 0},
      {959681532, 0, 0},
      {1684282922, 0, 0},
      {1811839150, 0, 0},
      {3929248764, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 1},
      {0, 6, 3},
      {0, 4, 2},
      {0, 9, 8},
      {0, 11, 10},
      {0, 7, 12},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(7, {
      {0, 0, 0},
      {679771963, 0, 0},
      {2320303498, 0, 0},
      {3334207724, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 3, 4},
      {0, 2, 5},
      {0, 1, 6},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(11, {
      {0, 0, 0},
      {2504802016, 0, 0},
      {3032677281, 0, 0},
      {3560552546, 0, 0},
      {3797961332, 0, 0},
      {3886529747, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 1},
      {0, 4, 3},
      {0, 7, 5},
      {0, 9, 8},
      {0, 6, 10},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(35, {
      {0, 0, 0},
      {354479447, 0, 0},
      {1325348861, 0, 0},
      {1367301635, 0, 0},
      {1368383673, 0, 0},
      {1646147798, 0, 0},
      {1679946323, 0, 0},
      {1766401548, 0, 0},
      {2096388952, 0, 0},
      {2123388694, 0, 0},
      {2362972044, 0, 0},
      {2660843182, 0, 0},
      {2698156268, 0, 0},
      {2970183398, 0, 0},
      {3133016299, 0, 0},
      {3187066832, 0, 0},
      {3882634684, 0, 0},
      {4243119782, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 11},
      {0, 6, 5},
      {0, 12, 9},
      {0, 16, 14},
      {0, 4, 17},
      {0, 10, 7},
      {0, 3, 1},
      {0, 19, 13},
      {0, 21, 20},
      {0, 23, 22},
      {0, 25, 24},
      {0, 27, 26},
      {0, 29, 28},
      {0, 8, 15},
      {0, 31, 30},
      {0, 33, 32},
      {0, 18, 34},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(37, {
      {0, 0, 0},
      {810488476, 0, 0},
      {959681532, 0, 0},
      {1232501371, 0, 0},
      {1372785527, 0, 0},
      {1526654696, 0, 0},
      {1684282922, 0, 0},
      {1901166356, 0, 0},
      {2244928358, 0, 0},
      {2314864456, 0, 0},
      {2524697596, 0, 0},
      {2568098594, 0, 0},
      {3117071189, 0, 0},
      {3188115516, 0, 0},
      {3554463148, 0, 0},
      {3691770462, 0, 0},
      {3929248764, 0, 0},
      {4060703604, 0, 0},
      {4092487128, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 4, 7},
      {0, 12, 8},
      {0, 17, 13},
      {0, 15, 18},
      {0, 14, 10},
      {0, 9, 5},
      {0, 2, 1},
      {0, 20, 6},
      {0, 22, 21},
      {0, 3, 23},
      {0, 11, 24},
      {0, 26, 25},
      {0, 28, 27},
      {0, 30, 29},
      {0, 32, 31},
      {0, 33, 16},
      {0, 35, 34},
      {0, 19, 36},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {1951208733, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(15, {
      {0, 0, 0},
      {170690025, 0, 0},
      {669982125, 0, 0},
      {1625742020, 0, 0},
      {2071351379, 0, 0},
      {2291766425, 0, 0},
      {3104643263, 0, 0},
      {3602108619, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 5, 1},
      {0, 6, 7},
      {0, 2, 3},
      {0, 9, 4},
      {0, 11, 10},
      {0, 13, 12},
      {0, 8, 14},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(45, {
      {0, 0, 0},
      {50385656, 0, 0},
      {615982737, 0, 0},
      {837715723, 0, 0},
      {1237148906, 0, 0},
      {1364157225, 0, 0},
      {1499923635, 0, 0},
      {1766401548, 0, 0},
      {2012838864, 0, 0},
      {2096388952, 0, 0},
      {2161102232, 0, 0},
      {2197874825, 0, 0},
      {2279700640, 0, 0},
      {2362972044, 0, 0},
      {2589449658, 0, 0},
      {2881302403, 0, 0},
      {2936040203, 0, 0},
      {2970183398, 0, 0},
      {3187066832, 0, 0},
      {3362830643, 0, 0},
      {3538158875, 0, 0},
      {3635542517, 0, 0},
      {3997432565, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 19, 10},
      {0, 3, 1},
      {0, 5, 6},
      {0, 20, 2},
      {0, 4, 22},
      {0, 25, 24},
      {0, 26, 12},
      {0, 16, 15},
      {0, 17, 9},
      {0, 14, 13},
      {0, 27, 21},
      {0, 7, 18},
      {0, 8, 28},
      {0, 11, 29},
      {0, 31, 30},
      {0, 33, 32},
      {0, 35, 34},
      {0, 37, 36},
      {0, 39, 38},
      {0, 41, 40},
      {0, 43, 42},
      {0, 23, 44},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(41, {
      {0, 0, 0},
      {139011596, 0, 0},
      {342159236, 0, 0},
      {837715723, 0, 0},
      {876867882, 0, 0},
      {1356063462, 0, 0},
      {1766401548, 0, 0},
      {2096388952, 0, 0},
      {2197874825, 0, 0},
      {2362972044, 0, 0},
      {2517964682, 0, 0},
      {2589449658, 0, 0},
      {2683080096, 0, 0},
      {2919626325, 0, 0},
      {2936040203, 0, 0},
      {2970183398, 0, 0},
      {2996594997, 0, 0},
      {3187066832, 0, 0},
      {3913885196, 0, 0},
      {3997432565, 0, 0},
      {4010499223, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 2, 3},
      {0, 5, 4},
      {0, 20, 16},
      {0, 19, 1},
      {0, 11, 9},
      {0, 14, 13},
      {0, 22, 17},
      {0, 23, 12},
      {0, 15, 24},
      {0, 8, 7},
      {0, 26, 25},
      {0, 6, 27},
      {0, 28, 18},
      {0, 30, 29},
      {0, 32, 31},
      {0, 34, 33},
      {0, 36, 35},
      {0, 37, 10},
      {0, 39, 38},
      {0, 21, 40},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>(3, {
      {0, 0, 0},
      {1036475267, 0, 0},
      {1111111111111111111, 0, 0},
      {0, 1, 2},
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLabel, 0), std::move(codec));
  }

  return codecs;
}
