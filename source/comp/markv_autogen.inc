
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
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 3520 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 960 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 960 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3243 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 1440 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 9663 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 960 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 11477 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1040 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 6 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 3203 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 352 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3623 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 5819 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 3448 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4800 },
      { kMarkvNoneOfTheAbove, 564 },
    }));

    codecs.emplace(SpvOpImageSampleExplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 7 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 323 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 3 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 115 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 2914 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 23 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3154 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3725 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 5008 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 800 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 7012 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 293 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1200 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3629 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 335 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 1603 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1588 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 782 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 8 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 2488 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 7 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 240 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 1200 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 5132 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 29 },
      { kMarkvNoneOfTheAbove, 430 },
    }));

    codecs.emplace(SpvOpFDiv, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 22964 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 5796 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 11328 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 11344 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 9686 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 8760 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 4880 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 10560 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 10200 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(SpvOpSampledImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 4 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 3424 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 3176 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 2604 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4168 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 209 },
      { CombineOpcodeAndNumOperands(SpvOpFMod, 4), 728 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1266 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 17307 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 26006 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1234 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 12 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 2416 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 155 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 50545 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 39711 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 5941 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 28013 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 10350 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 106893 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2247 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 2426 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 16642 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 21664 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 12627 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 26878 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 5770 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 6139 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 59 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1528 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 6481 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 6143 },
      { kMarkvNoneOfTheAbove, 4146 },
    }));

    codecs.emplace(SpvOpFMul, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThanEqual, 4), 833 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 10216 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 3868 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1217 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 3187 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 11769 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 5 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 227 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 13619 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 663 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 8167 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 1085 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 47493 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 34792 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 33797 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 3222 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 18592 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 236 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2647 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 9214 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 25159 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 2473 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 7 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 9578 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 12024 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 40033 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 6167 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3486 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 12880 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 72 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 4639 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 6240 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 210 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 3770 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 8662 },
      { kMarkvNoneOfTheAbove, 3485 },
    }));

    codecs.emplace(SpvOpFAdd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 14 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 10), 287 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 1690 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 280 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 106 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 13), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 6), 2378 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 181653 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 3592 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 1368 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 22370 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 16194 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 367 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 6 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 6164 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 60 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1208 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 667 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2865 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 59 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 4 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 568 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 8), 118 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 5 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 5 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 9), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 11), 1 },
      { kMarkvNoneOfTheAbove, 2468 },
    }));

    codecs.emplace(SpvOpTypePointer, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 481 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4640 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 69 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 11106 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 8960 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5374 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 13232 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 9001 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 5779 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 2793 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 20966 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 9666 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 2697 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 10738 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 127 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 14087 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 3899 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 7248 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 2466 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 9456 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 804 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2380 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 2347 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 4348 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 3291 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 332 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4657 },
      { kMarkvNoneOfTheAbove, 1684 },
    }));

    codecs.emplace(SpvOpFSub, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 1159 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 4150 },
      { CombineOpcodeAndNumOperands(SpvOpConvertSToF, 3), 4846 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 952 },
      { CombineOpcodeAndNumOperands(SpvOpIMul, 4), 3360 },
      { kMarkvNoneOfTheAbove, 145 },
    }));

    codecs.emplace(SpvOpIAdd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 82380 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 18335 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 2552 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 29082 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 12085 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleDrefExplicitLod, 7), 26146 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 1600 },
      { CombineOpcodeAndNumOperands(SpvOpShiftLeftLogical, 4), 4844 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 411888 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThanEqual, 4), 3417 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 13835 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 84211 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 12449 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 4030 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 6102 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 800 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 808 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 26522 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 28805 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 38172 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 5709 },
      { CombineOpcodeAndNumOperands(SpvOpBitcast, 3), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 328 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 61873 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 2535 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 35580 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 13374 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 10221 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 16008 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 2967 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 3192 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 5094 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 4852 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 5449 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 1056 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 105 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 2408 },
      { kMarkvNoneOfTheAbove, 12219 },
    }));

    codecs.emplace(SpvOpCompositeExtract, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 359 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 891 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 6 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 25 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 603 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 20 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 2486 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 2149 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 76 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 3650 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 256 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 2123 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1600 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 1604 },
      { kMarkvNoneOfTheAbove, 158 },
    }));

    codecs.emplace(SpvOpVectorTimesMatrix, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 38275 },
      { kMarkvNoneOfTheAbove, 382 },
    }));

    codecs.emplace(SpvOpBranch, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 44026 },
      { kMarkvNoneOfTheAbove, 440 },
    }));

    codecs.emplace(SpvOpFunctionEnd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 24139 },
      { kMarkvNoneOfTheAbove, 241 },
    }));

    codecs.emplace(SpvOpBranchConditional, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 7 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 18 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 47 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 4 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 837 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 525 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 407 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 3195 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 13624 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 4846 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 1601 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 15652 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 10 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5080 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 260 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 120 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 57 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 81 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 163 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 16 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 153 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 8 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 16 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 800 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 22 },
      { kMarkvNoneOfTheAbove, 498 },
    }));

    codecs.emplace(SpvOpFunctionCall, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 16 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 16 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 4716 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 3262 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 5446 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 6663 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 2960 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 104343 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 7124 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 14352 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 391556 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 7796 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 19985 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 5422 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 9101 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 13382 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 131 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 211 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 55 },
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 1737 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 829 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 276 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1080 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 6988 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 123 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 53605 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 488 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 11515 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 2 },
      { kMarkvNoneOfTheAbove, 7350 },
    }));

    codecs.emplace(SpvOpVariable, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 485475 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 107499 },
      { kMarkvNoneOfTheAbove, 5929 },
    }));

    codecs.emplace(SpvOpAccessChain, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 51 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 18 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 69 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 14 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 89 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 51 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 789 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 348 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3599 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 5861 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 1956 },
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 1 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 455 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 2455 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 42272 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 33922 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 754 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 7179 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 517 },
      { CombineOpcodeAndNumOperands(SpvOpLoopMerge, 3), 867 },
      { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 123 },
      { CombineOpcodeAndNumOperands(SpvOpPhi, 6), 9677 },
      { CombineOpcodeAndNumOperands(SpvOpIMul, 4), 145 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 2 },
      { CombineOpcodeAndNumOperands(SpvOpKill, 0), 4020 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1684 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 170 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1826 },
      { CombineOpcodeAndNumOperands(SpvOpSLessThan, 4), 1090 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 7 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 960 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 7087 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 1093 },
      { kMarkvNoneOfTheAbove, 1544 },
    }));

    codecs.emplace(SpvOpLabel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 17 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 22105 },
      { kMarkvNoneOfTheAbove, 221 },
    }));

    codecs.emplace(SpvOpReturn, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 43110 },
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 19795 },
      { kMarkvNoneOfTheAbove, 629 },
    }));

    codecs.emplace(SpvOpFunction, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 20546 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 4887 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 802 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 6127 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 17764 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 1 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 2428 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 756 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 4813 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 205 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 803 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 155 },
      { kMarkvNoneOfTheAbove, 694 },
    }));

    codecs.emplace(SpvOpTypeVector, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 42473 },
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 43110 },
      { kMarkvNoneOfTheAbove, 855 },
    }));

    codecs.emplace(SpvOpFunctionParameter, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 52 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 40800 },
      { kMarkvNoneOfTheAbove, 408 },
    }));

    codecs.emplace(SpvOpReturnValue, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 18879 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(SpvOpTypeVoid, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 90 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 4188 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 13), 768 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1407 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1019 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2423 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 3352 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 209221 },
      { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 21998 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2203 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 23178 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 65 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 1743 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 104 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 4923 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 1482 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 15046 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 166986 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 12581 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 579 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 11), 359 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 469 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 29667 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 5 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 56176 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 39174 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1832 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 463 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 895 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 431 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 235 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 14), 81 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 7 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 82 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 50 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 9), 323 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 109 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 498 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 10), 6 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 12), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 15), 25 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 17), 24 },
      { kMarkvNoneOfTheAbove, 6758 },
    }));

    codecs.emplace(SpvOpStore, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 2400 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 5 },
      { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 2), 13298 },
      { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpSource, 2), 45 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 3127 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(SpvOpEntryPoint, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 7), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 11), 14 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 2200 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 26 },
      { CombineOpcodeAndNumOperands(SpvOpIMul, 4), 1910 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 4342 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 82716 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 9), 1 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 13661 },
      { CombineOpcodeAndNumOperands(SpvOpBitcast, 3), 4866 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 1209 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 67125 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 11350 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 64466 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 81564 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 86635 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 122868 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 12474 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 25992 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 11660 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 13127 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 31909 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 125057 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 4471 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 45388 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 5861 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 4940 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 153858 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 56687 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 48604 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 34729 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1238 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 3089 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 32528 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 456 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 6457 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 19573 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 6217 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 7136 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdNotEqual, 4), 1360 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 25162 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 1145 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 1849 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 38811 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 124 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 94 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThan, 4), 1615 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 525 },
      { kMarkvNoneOfTheAbove, 18540 },
    }));

    codecs.emplace(SpvOpLoad, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 26), 4 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 17), 310 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 6), 1092 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 14), 2807 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 19), 953 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 8), 1263 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 7), 3259 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 10), 387 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 11), 811 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 13), 2936 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 12), 1678 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 18), 554 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 15), 912 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 20), 641 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 21), 185 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 16), 210 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 22), 63 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 5), 654 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 23), 48 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 9), 58 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 4), 6 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 24), 32 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 25), 16 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(SpvOpMemoryModel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 32 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 18846 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(SpvOpTypeFloat, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 6 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 684 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 14110 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 4748 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 32 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 11467 },
      { CombineOpcodeAndNumOperands(SpvOpSelect, 5), 1152 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 9868 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 11377 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 2402 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 60405 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 14435 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 8835 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 35545 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 21039 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 66445 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 11258 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 6668 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 5371 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 1085 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 1336 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 2075 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 9202 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 43934 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 16878 },
      { CombineOpcodeAndNumOperands(SpvOpMatrixTimesVector, 4), 1447 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 2520 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 44524 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 6693 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 16 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 12655 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 436 },
      { kMarkvNoneOfTheAbove, 4422 },
    }));

    codecs.emplace(SpvOpCompositeConstruct, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 9), 2 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 14), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 5 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 10), 71 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 16), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 3937 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 8), 204 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 6), 992 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 2034 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 10124 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 6199 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 1331 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 1234 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 840 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 1851 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 789 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 18854 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 2890 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 12), 768 },
      { kMarkvNoneOfTheAbove, 540 },
    }));

    codecs.emplace(SpvOpTypeFunction, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemoryModel, 2), 18879 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(SpvOpExtInstImport, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 20423 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 14352 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(SpvOpTypeImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 5 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 84279 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 79096 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 1873722 },
      { kMarkvNoneOfTheAbove, 20371 },
    }));

    codecs.emplace(SpvOpMemberDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCapability, 1), 3631 },
      { CombineOpcodeAndNumOperands(SpvOpExtInstImport, 2), 18879 },
      { kMarkvNoneOfTheAbove, 225 },
    }));

    codecs.emplace(SpvOpCapability, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 25798 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 579 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 20 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 1812 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 8 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 206 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 1952 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 1 },
      { kMarkvNoneOfTheAbove, 304 },
    }));

    codecs.emplace(SpvOpTypeInt, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 12 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 16288 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 3643 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 2527 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 5349 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2897 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 11010 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 9559 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 385 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 95125 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 10888 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 12 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 29 },
      { kMarkvNoneOfTheAbove, 1623 },
    }));

    codecs.emplace(SpvOpConstantComposite, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 12 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 48 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 340 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 5180 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 19195 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 7564 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2400 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(SpvOpTypeSampledImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 5 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 1155 },
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 7845 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 54929 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 3748 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 4044 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 3685 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 3728 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 2752 },
      { kMarkvNoneOfTheAbove, 1089 },
    }));

    codecs.emplace(SpvOpTypeStruct, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 15 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 891 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1598 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 582 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 19 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1442 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 1441 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 2302 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 4320 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 1791 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 256 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 27 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 5 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 178 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 2697 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 509 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 442 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 80 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 8685 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 390 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 2245 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 32 },
      { kMarkvNoneOfTheAbove, 299 },
    }));

    codecs.emplace(SpvOpFNegate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 16413 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVoid, 1), 18879 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 66078 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 15791 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 468825 },
      { kMarkvNoneOfTheAbove, 5859 },
    }));

    codecs.emplace(SpvOpDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 520 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 801 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 6244 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 622 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 3965 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 7 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 8270 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2285 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 498 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 141 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 10 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 2 },
      { kMarkvNoneOfTheAbove, 244 },
    }));

    codecs.emplace(SpvOpTypeMatrix, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 4 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 10483 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 64856 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 41330 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 195059 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 18 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 9601 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 3321 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 46831 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 35273 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 28243 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 3136 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 1772 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 13079 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 20 },
      { kMarkvNoneOfTheAbove, 4666 },
    }));

    codecs.emplace(SpvOpConstant, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 18 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1440 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 1841 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpConstantTrue, 2), 1956 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 2367 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 3288 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 81 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 349 },
      { kMarkvNoneOfTheAbove, 122 },
    }));

    codecs.emplace(SpvOpTypeBool, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 8 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 16165 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 72 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 18 },
      { kMarkvNoneOfTheAbove, 375 },
    }));

    codecs.emplace(SpvOpTypeArray, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 8 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 15 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4682 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4436 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 11942 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 2037 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 25829 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 16672 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5755 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 32437 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 33083 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 25838 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 6995 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2385 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 4587 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 63556 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThanEqual, 4), 418 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 6574 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 4 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 3723 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 8319 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 13259 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 27004 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 21730 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 625 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 7029 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 32 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 58196 },
      { kMarkvNoneOfTheAbove, 3920 },
    }));

    codecs.emplace(SpvOpExtInst, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 381 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 15 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 2766 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 949 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 7119 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 35761 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 20869 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 6423 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 14927 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 423 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 443 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 36 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 12084 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 14222 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 709 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 8268 },
      { CombineOpcodeAndNumOperands(SpvOpConvertFToS, 3), 162 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 10398 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 5984 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 136 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2465 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 3592 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 3835 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 5116 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 4 },
      { kMarkvNoneOfTheAbove, 1570 },
    }));

    codecs.emplace(SpvOpVectorTimesScalar, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4420 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 26630 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 9 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 518 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 439 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 4976 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 387 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 2775 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 23378 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 113 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 39955 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 41569 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 18558 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 85740 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 4075 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 78211 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 9875 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 859 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 2440 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 73737 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 20004 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 51881 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 11783 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 20395 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1077 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1542 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 14950 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1761 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 15548 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 6067 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 7422 },
      { kMarkvNoneOfTheAbove, 6181 },
    }));

    codecs.emplace(SpvOpVectorShuffle, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 4 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 14802 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 33741 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 153 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 61368 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 102 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 5086 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 4202 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 76 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 6136 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 384 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 4 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 440 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 27 },
      { kMarkvNoneOfTheAbove, 1265 },
    }));

    codecs.emplace(SpvOpImageSampleImplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFwidth, 3), 216 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 5 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1314 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1712 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 9752 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1506 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 5790 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 1805 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 9382 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 26051 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 8605 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 4244 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 5261 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 8612 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 4720 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 6056 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 5828 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 6280 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 32 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 28 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 14574 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 11873 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 2 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2905 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3764 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 8488 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 2217 },
      { kMarkvNoneOfTheAbove, 1523 },
    }));

    codecs.emplace(SpvOpDot, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 5 },
      { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 2), 13 },
      { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpSource, 2), 96 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 13201 },
      { kMarkvNoneOfTheAbove, 133 },
    }));

    codecs.emplace(SpvOpExecutionMode, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpBranchConditional, 3), 22009 },
      { kMarkvNoneOfTheAbove, 220 },
    }));

    codecs.emplace(SpvOpSelectionMerge, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3461 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1056 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 814 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 2926 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 407 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1463 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 9504 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 1463 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1056 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 828 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 3168 },
      { kMarkvNoneOfTheAbove, 261 },
    }));

    codecs.emplace(SpvOpImageSampleDrefExplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 9612 },
      { CombineOpcodeAndNumOperands(SpvOpUndef, 2), 12544 },
      { kMarkvNoneOfTheAbove, 221 },
    }));

    codecs.emplace(SpvOpUndef, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 18 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 61 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 4 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 195 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 62 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThanEqual, 4), 814 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 10572 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 31114 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 4107 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 8103 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 2732 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 883 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 8 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 3153 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 8979 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 40252 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 2335 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 202 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 5813 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 13076 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2858 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 9302 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 12347 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 2876 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 2453 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 217 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2381 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 145 },
      { kMarkvNoneOfTheAbove, 1858 },
    }));

    codecs.emplace(SpvOpCompositeInsert, std::move(codec));
  }

  return codecs;
}

std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>>
GetLiteralStringHuffmanCodecs() {
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>({
      { std::string("main"), 11091 },
      { std::string("MainPs"), 5388 },
      { std::string("MainVs"), 2400 },
      { std::string("kMarkvNoneOfTheAbove"), 188 },
    }));

    codecs.emplace(SpvOpEntryPoint, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>({
      { std::string("GLSL.std.450"), 18879 },
      { std::string("kMarkvNoneOfTheAbove"), 188 },
    }));

    codecs.emplace(SpvOpExtInstImport, std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetNonIdWordHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4, 8982 },
      { 8, 11714 },
      { 9, 472 },
      { 10, 3550 },
      { 26, 21184 },
      { 27, 466 },
      { 29, 1381 },
      { 30, 619 },
      { 31, 10071 },
      { 32, 1003 },
      { 37, 3409 },
      { 40, 79897 },
      { 43, 78206 },
      { 46, 76057 },
      { 48, 634 },
      { 49, 4019 },
      { 66, 1859 },
      { 67, 2314 },
      { 68, 8273 },
      { 69, 66141 },
      { 71, 6379 },
      { kMarkvNoneOfTheAbove, 4471 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 18879 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 18879 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 5577 },
      { 4, 13291 },
      { kMarkvNoneOfTheAbove, 199 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpEntryPoint, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 7, 13229 },
      { 8, 62 },
      { kMarkvNoneOfTheAbove, 161 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExecutionMode, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 2 },
      { 2, 1 },
      { 3, 3 },
      { 4, 1 },
      { 18, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExecutionMode, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 18868 },
      { 32, 3616 },
      { kMarkvNoneOfTheAbove, 251 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCapability, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 32, 30454 },
      { kMarkvNoneOfTheAbove, 304 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 18868 },
      { 1, 11586 },
      { kMarkvNoneOfTheAbove, 304 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 32, 18879 },
      { kMarkvNoneOfTheAbove, 188 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFloat, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 23651 },
      { 3, 18475 },
      { 4, 27278 },
      { kMarkvNoneOfTheAbove, 694 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 604 },
      { 3, 7030 },
      { 4, 16815 },
      { kMarkvNoneOfTheAbove, 244 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeMatrix, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 21096 },
      { 2, 9019 },
      { 3, 4660 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 31793 },
      { 1, 2982 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 34752 },
      { kMarkvNoneOfTheAbove, 370 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 34775 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 34775 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 34775 },
      { kMarkvNoneOfTheAbove, 347 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 41123 },
      { 1, 57058 },
      { 2, 62123 },
      { 3, 37791 },
      { 6, 14777 },
      { 7, 33663 },
      { kMarkvNoneOfTheAbove, 2465 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 38384 },
      { 1, 20168 },
      { 2, 29077 },
      { 3, 19135 },
      { 4, 9732 },
      { 5, 9234 },
      { 6, 8164 },
      { 7, 7341 },
      { 8, 6197 },
      { 9, 4195 },
      { 10, 4972 },
      { 11, 2361 },
      { 12, 3863 },
      { 13, 3071 },
      { 14, 2725 },
      { 15, 2720 },
      { 16, 4509 },
      { 17, 4296 },
      { 18, 3805 },
      { 19, 3893 },
      { 20, 3796 },
      { 21, 3622 },
      { 22, 3459 },
      { 23, 3802 },
      { 24, 2862 },
      { 25, 1281 },
      { 26, 1508 },
      { 27, 1982 },
      { 28, 1707 },
      { 29, 2906 },
      { 30, 2422 },
      { 31, 2351 },
      { 32, 1922 },
      { 33, 1210 },
      { 34, 1018 },
      { 35, 923 },
      { 36, 846 },
      { 37, 851 },
      { 38, 844 },
      { 39, 752 },
      { 40, 828 },
      { 41, 978 },
      { 42, 978 },
      { 43, 978 },
      { 44, 828 },
      { 45, 794 },
      { 256, 3064 },
      { 507307272, 2904 },
      { 864026611, 1750 },
      { 869711765, 468 },
      { 922746870, 472 },
      { 925353388, 602 },
      { 981668463, 4888 },
      { 997553156, 1440 },
      { 1014330372, 1440 },
      { 1020708227, 1440 },
      { 1028443341, 1937 },
      { 1032953056, 3888 },
      { 1033463938, 1440 },
      { 1033463943, 1463 },
      { 1039998884, 1440 },
      { 1039998950, 1463 },
      { 1040187392, 1619 },
      { 1042401985, 3888 },
      { 1044220635, 3888 },
      { 1045622707, 1440 },
      { 1045622740, 1463 },
      { 1048576000, 2394 },
      { 1053609165, 3904 },
      { 1053790359, 3888 },
      { 1054423330, 480 },
      { 1054448026, 1968 },
      { 1055437881, 9474 },
      { 1056300230, 3888 },
      { 1056964608, 17499 },
      { 1058056805, 3888 },
      { 1059286575, 3888 },
      { 1061158912, 1962 },
      { 1061997773, 4395 },
      { 1064514355, 1955 },
      { 1064854933, 3888 },
      { 1065353216, 18701 },
      { 1065354055, 870 },
      { 1067450368, 468 },
      { 1068289229, 984 },
      { 1069547520, 1787 },
      { 1073741824, 14579 },
      { 1077936128, 2003 },
      { 1082130432, 4445 },
      { 1083416238, 1040 },
      { 1090519040, 654 },
      { 1091567616, 3904 },
      { 1098907648, 1048 },
      { 1107296256, 1009 },
      { 1115422720, 2431 },
      { 1115684864, 1211 },
      { 1124073472, 2881 },
      { 1132396544, 2906 },
      { 1132462080, 481 },
      { 1140850688, 1603 },
      { 1199562752, 2127 },
      { 3179067684, 3888 },
      { 3180973575, 3888 },
      { 3182651297, 3888 },
      { 3196448879, 3888 },
      { 3201906978, 480 },
      { 3204448256, 2024 },
      { 3204993516, 3888 },
      { 3205248529, 3888 },
      { 3207137644, 3888 },
      { 3208642560, 1945 },
      { 3211081967, 3888 },
      { 3212836864, 8554 },
      { 3215772877, 984 },
      { 3221225472, 594 },
      { 3263168512, 1228 },
      { 3271557120, 1228 },
      { 3332128768, 5774 },
      { kMarkvNoneOfTheAbove, 12469 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 62905 },
      { kMarkvNoneOfTheAbove, 629 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 160700 },
      { 1, 103581 },
      { 2, 42809 },
      { 3, 64316 },
      { 6, 142767 },
      { 7, 183773 },
      { kMarkvNoneOfTheAbove, 6979 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 15742 },
      { 2, 84278 },
      { 6, 35085 },
      { 11, 10950 },
      { 18, 691 },
      { 30, 154654 },
      { 33, 81053 },
      { 34, 203509 },
      { kMarkvNoneOfTheAbove, 5883 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 245709 },
      { 1, 51550 },
      { 2, 29794 },
      { 3, 24660 },
      { 4, 19385 },
      { 5, 18258 },
      { 6, 15819 },
      { 7, 11641 },
      { 8, 6292 },
      { 9, 2910 },
      { 10, 1961 },
      { 11, 1250 },
      { 12, 3010 },
      { 13, 2671 },
      { 14, 2963 },
      { 15, 6334 },
      { 16, 19387 },
      { 17, 965 },
      { 18, 2438 },
      { 19, 1290 },
      { 20, 802 },
      { 64, 15692 },
      { kMarkvNoneOfTheAbove, 5322 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 115743 },
      { 1, 96871 },
      { 2, 94349 },
      { 3, 94337 },
      { 4, 62949 },
      { 5, 62956 },
      { 6, 62932 },
      { 7, 62844 },
      { 8, 62825 },
      { 9, 62732 },
      { 10, 62759 },
      { 11, 62738 },
      { 12, 62738 },
      { 13, 62730 },
      { 14, 62724 },
      { 15, 62753 },
      { 16, 62684 },
      { 17, 62556 },
      { 18, 62553 },
      { 19, 62433 },
      { 20, 62033 },
      { 21, 46205 },
      { 22, 45165 },
      { 23, 76079 },
      { 24, 43805 },
      { 25, 27988 },
      { 26, 27508 },
      { 27, 27508 },
      { 28, 26548 },
      { 29, 26466 },
      { 30, 26460 },
      { 31, 26454 },
      { 32, 26351 },
      { 33, 25823 },
      { 34, 21719 },
      { 35, 17399 },
      { 36, 15671 },
      { 37, 47007 },
      { 38, 15671 },
      { kMarkvNoneOfTheAbove, 20407 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4, 94035 },
      { 7, 94035 },
      { 11, 4980 },
      { 35, 1843200 },
      { kMarkvNoneOfTheAbove, 21223 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 84284 },
      { 3, 2416 },
      { 16, 141303 },
      { 28, 26463 },
      { 32, 47213 },
      { 36, 10808 },
      { 40, 26386 },
      { 44, 25824 },
      { 48, 47176 },
      { 60, 31332 },
      { 64, 62843 },
      { 68, 4753 },
      { 72, 4762 },
      { 76, 36076 },
      { 80, 47098 },
      { 84, 10799 },
      { 88, 6055 },
      { 92, 21711 },
      { 96, 47113 },
      { 100, 6054 },
      { 104, 4824 },
      { 108, 20406 },
      { 112, 47047 },
      { 116, 4749 },
      { 120, 20406 },
      { 124, 10795 },
      { 128, 62732 },
      { 132, 10795 },
      { 136, 10795 },
      { 140, 37375 },
      { 144, 46957 },
      { 148, 21705 },
      { 152, 26521 },
      { 156, 36072 },
      { 160, 46971 },
      { 172, 6068 },
      { 176, 46943 },
      { 188, 4750 },
      { 192, 62657 },
      { 200, 4683 },
      { 204, 21721 },
      { 208, 31268 },
      { 220, 5510 },
      { 224, 31280 },
      { 232, 5470 },
      { 236, 8411 },
      { 240, 14309 },
      { 248, 9331 },
      { 256, 39822 },
      { 272, 17599 },
      { 288, 16719 },
      { 292, 16308 },
      { 296, 16631 },
      { 300, 16308 },
      { 304, 16079 },
      { 316, 15748 },
      { 320, 16117 },
      { 332, 16071 },
      { 336, 15718 },
      { 348, 15668 },
      { 352, 15715 },
      { 364, 15668 },
      { 368, 15671 },
      { 372, 15668 },
      { 376, 15668 },
      { 384, 15685 },
      { 392, 15668 },
      { 400, 15686 },
      { 416, 15671 },
      { 424, 15668 },
      { 432, 15671 },
      { 448, 15671 },
      { 460, 15668 },
      { 464, 15671 },
      { 468, 15668 },
      { 472, 15668 },
      { 476, 15668 },
      { 480, 15715 },
      { 488, 15668 },
      { 492, 15668 },
      { 496, 15671 },
      { 512, 15671 },
      { 640, 15712 },
      { kMarkvNoneOfTheAbove, 31271 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 391888 },
      { 1, 10653 },
      { 2, 51930 },
      { 3, 38119 },
      { 4, 78493 },
      { kMarkvNoneOfTheAbove, 6196 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 23717 },
      { 1, 378166 },
      { 2, 11876 },
      { 3, 65970 },
      { 4, 13539 },
      { 5, 78296 },
      { kMarkvNoneOfTheAbove, 5715 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 17289 },
      { 1, 5162 },
      { 2, 273658 },
      { 3, 17266 },
      { 4, 5921 },
      { 5, 5035 },
      { 6, 63941 },
      { kMarkvNoneOfTheAbove, 3882 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 6289 },
      { 1, 4949 },
      { 2, 2378 },
      { 3, 96617 },
      { 4, 481 },
      { 5, 6397 },
      { 6, 1503 },
      { kMarkvNoneOfTheAbove, 1186 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 245096 },
      { 1, 185519 },
      { 2, 168763 },
      { 3, 104061 },
      { 4, 20571 },
      { 5, 29186 },
      { 6, 9174 },
      { 7, 9712 },
      { 8, 13948 },
      { 9, 13648 },
      { 10, 10496 },
      { 11, 6960 },
      { 12, 12176 },
      { 13, 8976 },
      { 14, 7912 },
      { 15, 4424 },
      { 16, 11384 },
      { 17, 6504 },
      { 18, 6656 },
      { 19, 7168 },
      { 20, 9560 },
      { 21, 6216 },
      { 22, 8360 },
      { 23, 32128 },
      { 24, 12712 },
      { 25, 2184 },
      { 26, 2792 },
      { 27, 5352 },
      { 28, 4792 },
      { 29, 5520 },
      { 30, 5520 },
      { 31, 4776 },
      { 32, 2208 },
      { kMarkvNoneOfTheAbove, 10426 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 31904 },
      { 1, 20033 },
      { 2, 20948 },
      { 3, 24408 },
      { 4, 1221 },
      { 5, 1121 },
      { 6, 959 },
      { 7, 940 },
      { 8, 624 },
      { 9, 840 },
      { 10, 741 },
      { 11, 725 },
      { 12, 614 },
      { 13, 602 },
      { 14, 342 },
      { 15, 342 },
      { 16, 342 },
      { 17, 342 },
      { 18, 342 },
      { 19, 340 },
      { 20, 132 },
      { 23, 190 },
      { 24, 254 },
      { 25, 245 },
      { 26, 244 },
      { 27, 244 },
      { 28, 244 },
      { 29, 1025 },
      { 30, 1025 },
      { 31, 961 },
      { 32, 970 },
      { 33, 970 },
      { 34, 970 },
      { 35, 870 },
      { 36, 809 },
      { 37, 809 },
      { 38, 809 },
      { 39, 729 },
      { 40, 790 },
      { 41, 790 },
      { 42, 790 },
      { 43, 790 },
      { 44, 790 },
      { 45, 790 },
      { kMarkvNoneOfTheAbove, 1292 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 4224 },
      { 1, 4224 },
      { 2, 4224 },
      { 3, 4224 },
      { kMarkvNoneOfTheAbove, 168 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 43828 },
      { 1, 30228 },
      { 2, 25282 },
      { 3, 17333 },
      { 4, 9388 },
      { 5, 10208 },
      { 6, 9726 },
      { 7, 10652 },
      { 8, 7084 },
      { 9, 2860 },
      { 10, 580 },
      { kMarkvNoneOfTheAbove, 1671 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 4760 },
      { 1, 3944 },
      { 2, 2424 },
      { 3, 13292 },
      { kMarkvNoneOfTheAbove, 244 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 35 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 28186 },
      { 10, 26350 },
      { kMarkvNoneOfTheAbove, 545 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 26146 },
      { kMarkvNoneOfTheAbove, 261 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 22009 },
      { kMarkvNoneOfTheAbove, 220 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSelectionMerge, 1), std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetIdDescriptorHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 156525 },
      { 1951208733, 208304 },
      { 2320303498, 9118 },
      { 3334207724, 13282 },
      { kMarkvNoneOfTheAbove, 3872 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 34183582, 1248 },
      { 223800276, 1320 },
      { 295018543, 1512 },
      { 439764402, 1395 },
      { 443558693, 3792 },
      { 583624926, 1248 },
      { 599185303, 1596 },
      { 779021139, 1722 },
      { 1015552308, 1596 },
      { 1027242654, 1228 },
      { 1077859090, 1280 },
      { 1104362365, 1416 },
      { 1132589448, 1248 },
      { 1236389532, 1596 },
      { 1739837626, 1368 },
      { 1955104493, 1644 },
      { 2161102232, 4811 },
      { 2197874825, 1368 },
      { 2217833278, 1222 },
      { 2244470522, 1596 },
      { 2532518896, 1828 },
      { 2789375411, 1356 },
      { 3061690214, 1368 },
      { 3287039847, 2685 },
      { 3357301402, 2634 },
      { 3365041621, 1248 },
      { 3510257966, 1368 },
      { 3534235309, 1440 },
      { 4018237905, 1656 },
      { 4145966869, 1440 },
      { 4272200782, 2634 },
      { kMarkvNoneOfTheAbove, 337238 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4228502127, 387229 },
      { kMarkvNoneOfTheAbove, 3872 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50998433, 1280 },
      { 139011596, 10009 },
      { 181902171, 1598 },
      { 296981500, 3679 },
      { 321630747, 2304 },
      { 416853049, 1440 },
      { 464259778, 1222 },
      { 615982737, 2136 },
      { 669982125, 1368 },
      { 759277550, 1968 },
      { 810488476, 4591 },
      { 870594305, 4102 },
      { 922996215, 1248 },
      { 969500141, 1368 },
      { 1015552308, 1596 },
      { 1139547465, 3792 },
      { 1203545131, 3649 },
      { 1220643281, 1248 },
      { 1220749418, 1440 },
      { 1367301635, 3777 },
      { 1395923345, 1248 },
      { 1554194368, 1248 },
      { 1742737136, 4886 },
      { 1755648697, 1971 },
      { 1962162282, 1488 },
      { 1964254745, 1368 },
      { 2055836767, 1730 },
      { 2096388952, 6878 },
      { 2124837447, 1596 },
      { 2161102232, 2634 },
      { 2321729979, 3431 },
      { 2346547796, 1632 },
      { 2399809085, 1228 },
      { 2432827426, 1228 },
      { 2455417440, 1416 },
      { 2572638469, 2064 },
      { 2614879967, 3644 },
      { 2855506940, 20158 },
      { 2919796598, 1356 },
      { 2970183398, 1977 },
      { 2976066508, 2820 },
      { 3044188332, 4811 },
      { 3061690214, 1368 },
      { 3091876332, 1685 },
      { 3104643263, 1974 },
      { 3107165180, 1922 },
      { 3187066832, 2159 },
      { 3413713311, 2269 },
      { 3487022798, 4171 },
      { 3602693817, 2748 },
      { 3678875745, 1416 },
      { 3701632935, 4025 },
      { 3829325073, 1248 },
      { 4040340620, 2221 },
      { 4174489262, 1596 },
      { 4272200782, 2634 },
      { kMarkvNoneOfTheAbove, 235038 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 72782198, 1968 },
      { 139011596, 4521 },
      { 296981500, 7796 },
      { 300939750, 1920 },
      { 401211099, 924 },
      { 429277936, 1222 },
      { 505940164, 1020 },
      { 538168945, 936 },
      { 603915804, 936 },
      { 688216667, 1920 },
      { 706016261, 1632 },
      { 790502615, 1015 },
      { 810488476, 941 },
      { 993150979, 1488 },
      { 1203545131, 2666 },
      { 1206726575, 1600 },
      { 1265796414, 1632 },
      { 1314843976, 1920 },
      { 1367301635, 3776 },
      { 1378082995, 1020 },
      { 1410311776, 936 },
      { 1443829854, 1068 },
      { 1448448666, 864 },
      { 1468919488, 912 },
      { 1496351055, 1600 },
      { 1619778288, 2112 },
      { 1684282922, 1694 },
      { 1848784182, 1488 },
      { 1901166356, 1032 },
      { 2095546797, 1248 },
      { 2096388952, 2140 },
      { 2162986400, 1951 },
      { 2197874825, 1248 },
      { 2246405597, 1488 },
      { 2250225826, 1020 },
      { 2282454607, 936 },
      { 2328748202, 1920 },
      { 2348201466, 1080 },
      { 2597020383, 1824 },
      { 2633682514, 1488 },
      { 2817335337, 1747 },
      { 2855506940, 72723 },
      { 2936040203, 864 },
      { 2955375511, 2496 },
      { 3122368657, 1194 },
      { 3154597438, 936 },
      { 3184381405, 1920 },
      { 3187066832, 1208 },
      { 3233393284, 3128 },
      { 3251128023, 12499 },
      { 3260309823, 1488 },
      { 3441531391, 1632 },
      { 3496407048, 1691 },
      { 3582002820, 2832 },
      { 3647586740, 1488 },
      { 3653838348, 1207 },
      { 3730093054, 2112 },
      { 3759072440, 1248 },
      { 3928764629, 1020 },
      { 3969279737, 1920 },
      { 3994511488, 1020 },
      { 4026740269, 1068 },
      { 4274214049, 1488 },
      { kMarkvNoneOfTheAbove, 95354 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 99347751, 720 },
      { 102542696, 744 },
      { 107497541, 1248 },
      { 112452386, 704 },
      { 139011596, 900 },
      { 296981500, 68240 },
      { 429277936, 1344 },
      { 451957774, 960 },
      { 508217552, 2063 },
      { 573901046, 672 },
      { 774727851, 640 },
      { 801484894, 624 },
      { 920604853, 624 },
      { 925559698, 624 },
      { 1022915255, 695 },
      { 1209418480, 1141 },
      { 1287937401, 528 },
      { 1319785741, 659 },
      { 1392080469, 528 },
      { 1538342947, 1344 },
      { 1541020250, 888 },
      { 1587209598, 480 },
      { 1594733696, 480 },
      { 1631434666, 1560 },
      { 1636389511, 1160 },
      { 1684282922, 1312 },
      { 1859128680, 528 },
      { 1901166356, 820 },
      { 2004567202, 644 },
      { 2119793999, 539 },
      { 2280400314, 480 },
      { 2538917932, 600 },
      { 2677264274, 528 },
      { 2683080096, 1951 },
      { 2854085372, 528 },
      { 2879917501, 480 },
      { 3059119137, 672 },
      { 3174324790, 528 },
      { 3194725903, 576 },
      { 3358097187, 480 },
      { 3547456240, 1820 },
      { 3614752756, 576 },
      { 3753486980, 3123 },
      { 3811268385, 624 },
      { 3953733490, 816 },
      { 3990925720, 600 },
      { kMarkvNoneOfTheAbove, 52069 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 166253838, 2452 },
      { 679771963, 18457 },
      { 1247793383, 4833 },
      { 2261697609, 2089 },
      { 2263349224, 3858 },
      { 2320303498, 18817 },
      { 3334207724, 18879 },
      { kMarkvNoneOfTheAbove, 713 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 6923 },
      { 1415510495, 2470 },
      { 1951208733, 56153 },
      { 2430404313, 3858 },
      { kMarkvNoneOfTheAbove, 694 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1389644742, 15668 },
      { 3232633974, 2416 },
      { 3278176820, 3062 },
      { 3648138580, 169 },
      { 3687777340, 177 },
      { 3694383800, 256 },
      { 3697687030, 15697 },
      { kMarkvNoneOfTheAbove, 515 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 2443 },
      { 2160380860, 15692 },
      { 3334207724, 19384 },
      { kMarkvNoneOfTheAbove, 441 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 144116905, 2416 },
      { 827246872, 210 },
      { 1545298048, 176 },
      { 2715370488, 3062 },
      { 2798552666, 31371 },
      { 3812456892, 297 },
      { kMarkvNoneOfTheAbove, 428 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 40653745, 15722 },
      { 119981689, 3062 },
      { 153085016, 640 },
      { 451382997, 800 },
      { 545678922, 2947 },
      { 899570100, 1320 },
      { 929101967, 1230 },
      { 1070791291, 15668 },
      { 1100599986, 3556 },
      { 1103903216, 2148 },
      { 1154919607, 15668 },
      { 1199157863, 480 },
      { 1258105452, 640 },
      { 1369578001, 1668 },
      { 1372881231, 360 },
      { 1674803691, 4104 },
      { 1677700667, 400 },
      { 1989520052, 2411 },
      { 2593884753, 528 },
      { 2664825925, 400 },
      { 2924146124, 15657 },
      { 2926633629, 432 },
      { 3249265647, 336 },
      { 3345288309, 860 },
      { 3410158390, 4320 },
      { 3489360962, 2732 },
      { 3495967422, 650 },
      { 3504981554, 1728 },
      { 3705139860, 1399 },
      { 3822983876, 660 },
      { 4141567741, 1008 },
      { 4234287173, 444 },
      { 4240893633, 1038 },
      { kMarkvNoneOfTheAbove, 4897 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 17764 },
      { 1951208733, 11138 },
      { 2160380860, 15668 },
      { 3278176820, 3062 },
      { 3334207724, 61106 },
      { kMarkvNoneOfTheAbove, 1175 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 29450 },
      { 1951208733, 5234 },
      { 2160380860, 15668 },
      { 2320303498, 13655 },
      { 3232633974, 2411 },
      { 3334207724, 23544 },
      { kMarkvNoneOfTheAbove, 960 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 13688 },
      { 1951208733, 26807 },
      { 2160380860, 15671 },
      { 2320303498, 15669 },
      { 3334207724, 12701 },
      { kMarkvNoneOfTheAbove, 900 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 20487 },
      { 1951208733, 11144 },
      { 2160380860, 15668 },
      { 2320303498, 17766 },
      { 3334207724, 18037 },
      { kMarkvNoneOfTheAbove, 907 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 20516 },
      { 1951208733, 26526 },
      { 2263349224, 1920 },
      { 3334207724, 32436 },
      { kMarkvNoneOfTheAbove, 1062 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15660 },
      { 1951208733, 26443 },
      { 2320303498, 893 },
      { 3334207724, 38329 },
      { kMarkvNoneOfTheAbove, 846 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15672 },
      { 1951208733, 41752 },
      { 2320303498, 8238 },
      { 3334207724, 14344 },
      { kMarkvNoneOfTheAbove, 863 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15662 },
      { 1951208733, 43273 },
      { 2320303498, 2431 },
      { 3334207724, 15968 },
      { kMarkvNoneOfTheAbove, 818 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 38047 },
      { 2320303498, 9879 },
      { 3334207724, 25196 },
      { kMarkvNoneOfTheAbove, 806 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15660 },
      { 1951208733, 26240 },
      { 2320303498, 622 },
      { 3334207724, 24546 },
      { kMarkvNoneOfTheAbove, 739 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15673 },
      { 1951208733, 27114 },
      { 2320303498, 4803 },
      { 3334207724, 15973 },
      { kMarkvNoneOfTheAbove, 669 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 31334 },
      { 1951208733, 31264 },
      { kMarkvNoneOfTheAbove, 723 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15772 },
      { 1951208733, 42042 },
      { 2320303498, 4803 },
      { kMarkvNoneOfTheAbove, 703 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 16219 },
      { 1951208733, 25937 },
      { 2320303498, 15660 },
      { 3334207724, 4844 },
      { kMarkvNoneOfTheAbove, 650 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 19772 },
      { 1951208733, 22504 },
      { 2320303498, 20337 },
      { kMarkvNoneOfTheAbove, 693 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 35765 },
      { 1951208733, 25191 },
      { 2320303498, 1640 },
      { kMarkvNoneOfTheAbove, 693 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 16), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 18036 },
      { 1951208733, 25289 },
      { 2320303498, 3603 },
      { 3334207724, 15712 },
      { kMarkvNoneOfTheAbove, 626 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 17), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 16108 },
      { 1951208733, 44045 },
      { 3334207724, 2400 },
      { kMarkvNoneOfTheAbove, 628 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 18), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1200 },
      { 1951208733, 59193 },
      { 2320303498, 2160 },
      { kMarkvNoneOfTheAbove, 625 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 19), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15910 },
      { 1951208733, 45643 },
      { 2320303498, 880 },
      { kMarkvNoneOfTheAbove, 624 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 20), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 886 },
      { 1951208733, 28611 },
      { 2320303498, 32536 },
      { kMarkvNoneOfTheAbove, 620 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 21), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 182 },
      { 1951208733, 14298 },
      { 2320303498, 15668 },
      { 3334207724, 16057 },
      { kMarkvNoneOfTheAbove, 462 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 22), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1337 },
      { 1951208733, 12183 },
      { 2320303498, 15988 },
      { 3334207724, 15657 },
      { kMarkvNoneOfTheAbove, 451 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 23), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4190 },
      { 1951208733, 8930 },
      { 2160380860, 15657 },
      { 2320303498, 320 },
      { 3334207724, 15668 },
      { kMarkvNoneOfTheAbove, 447 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 24), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4422 },
      { 1951208733, 7738 },
      { 2320303498, 15988 },
      { 3334207724, 15657 },
      { kMarkvNoneOfTheAbove, 438 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 25), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 2265 },
      { 1951208733, 10052 },
      { 2320303498, 15671 },
      { kMarkvNoneOfTheAbove, 279 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 26), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4430 },
      { 1951208733, 7407 },
      { 2320303498, 15671 },
      { kMarkvNoneOfTheAbove, 275 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 27), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 20090 },
      { 1951208733, 7416 },
      { kMarkvNoneOfTheAbove, 277 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 28), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 2263 },
      { 1951208733, 24277 },
      { kMarkvNoneOfTheAbove, 273 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 29), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4104 },
      { 1951208733, 22347 },
      { kMarkvNoneOfTheAbove, 279 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 30), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4320 },
      { 1951208733, 22028 },
      { 2320303498, 112 },
      { kMarkvNoneOfTheAbove, 264 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 31), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1728 },
      { 1951208733, 24095 },
      { 2320303498, 631 },
      { kMarkvNoneOfTheAbove, 264 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 32), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 21719 },
      { 2320303498, 4632 },
      { kMarkvNoneOfTheAbove, 263 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 33), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 1728 },
      { 2320303498, 24095 },
      { kMarkvNoneOfTheAbove, 258 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 34), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 15668 },
      { 2320303498, 6051 },
      { kMarkvNoneOfTheAbove, 217 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 35), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 15668 },
      { 2320303498, 1728 },
      { kMarkvNoneOfTheAbove, 176 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 36), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15671 },
      { kMarkvNoneOfTheAbove, 156 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 37), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1389644742, 15668 },
      { kMarkvNoneOfTheAbove, 159 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 38), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3697687030, 15668 },
      { kMarkvNoneOfTheAbove, 159 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 39), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2320303498, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 40), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2320303498, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 41), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 42), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 43), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 44), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 45), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 46), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 47), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 48), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 49), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 50), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 51), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 85880059, 800 },
      { 135486769, 4099 },
      { 304448521, 1320 },
      { 436416061, 1012 },
      { 440421571, 4980 },
      { 450406196, 2596 },
      { 503094540, 4320 },
      { 543621065, 7788 },
      { 626892406, 1147 },
      { 628544021, 1107 },
      { 827698488, 6438 },
      { 869050696, 1728 },
      { 907126242, 4841 },
      { 908777857, 4841 },
      { 910429472, 3182 },
      { 1113409935, 797 },
      { 1294403159, 6309 },
      { 1296054774, 6050 },
      { 1297706389, 5621 },
      { 1322549027, 6084 },
      { 1784441183, 3052 },
      { 2080953106, 4274 },
      { 2194691858, 10303 },
      { 2448331885, 3858 },
      { 2466255445, 768 },
      { 2468230023, 2957 },
      { 2547657777, 15339 },
      { 2549309392, 13490 },
      { 2550961007, 12727 },
      { 2894051250, 1184 },
      { 2929019254, 998 },
      { 2934934694, 2989 },
      { 2936586309, 2639 },
      { 2938237924, 2725 },
      { 3077271274, 860 },
      { 3092528578, 2064 },
      { 3094180193, 4500 },
      { 3094857332, 3973 },
      { 3095831808, 7788 },
      { 3183924418, 10249 },
      { 3207966516, 965 },
      { 3282979782, 879 },
      { 3433956341, 1008 },
      { 3561562003, 18879 },
      { 3563213618, 5294 },
      { 3564865233, 4594 },
      { 3585511591, 1412 },
      { 4028622909, 13556 },
      { 4039938779, 2411 },
      { 4050155669, 853 },
      { kMarkvNoneOfTheAbove, 17352 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 119981689, 3052 },
      { 162255877, 7788 },
      { 451382997, 800 },
      { 545678922, 811 },
      { 679771963, 32314 },
      { 789872778, 917 },
      { 1100599986, 1012 },
      { 1103903216, 1912 },
      { 1154919607, 13556 },
      { 1343794461, 4500 },
      { 1415510495, 1033 },
      { 1674803691, 4099 },
      { 1951208733, 22622 },
      { 1989520052, 2411 },
      { 2160380860, 2331 },
      { 2263349224, 4498 },
      { 2320303498, 28849 },
      { 2924146124, 10249 },
      { 2984325996, 7788 },
      { 3334207724, 48357 },
      { 3345288309, 1120 },
      { 3410158390, 4320 },
      { 3489360962, 2816 },
      { 3495967422, 840 },
      { 3504981554, 1728 },
      { 3800912395, 2130 },
      { 3802564010, 1716 },
      { 3866587616, 2596 },
      { 3868239231, 6084 },
      { 3869890846, 10303 },
      { 3998230222, 2064 },
      { 4240893633, 1376 },
      { kMarkvNoneOfTheAbove, 13008 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 75986790, 18879 },
      { 95470391, 768 },
      { 170378107, 315 },
      { 172029722, 679 },
      { 204234270, 368 },
      { 205885885, 312 },
      { 244668133, 768 },
      { 265778447, 360 },
      { 616435646, 168 },
      { 618087261, 168 },
      { 753954113, 291 },
      { 1000070091, 800 },
      { 1308462133, 169 },
      { 1671139745, 338 },
      { 1774874546, 240 },
      { 1776526161, 330 },
      { 1887808856, 872 },
      { 1889460471, 1596 },
      { 1917966999, 530 },
      { 2044728014, 528 },
      { 2192810893, 312 },
      { 2293247016, 779 },
      { 2503194620, 842 },
      { 2605012269, 168 },
      { 2608484640, 768 },
      { 2615111110, 463 },
      { 2668769415, 768 },
      { 2759951687, 1866 },
      { 2761603302, 2578 },
      { 2856623532, 280 },
      { 2945369269, 168 },
      { 2956189845, 1596 },
      { 3085119011, 190 },
      { 3367313400, 214 },
      { 3447882276, 400 },
      { 3633746133, 1605 },
      { 3635397748, 1604 },
      { 3710645347, 234 },
      { 3712296962, 567 },
      { 3715846592, 169 },
      { 3727494858, 839 },
      { 3747079365, 828 },
      { 3748965853, 238 },
      { 3750617468, 278 },
      { 4018820793, 211 },
      { 4022124023, 211 },
      { 4024173916, 324 },
      { 4215670524, 826 },
      { 4217322139, 1014 },
      { kMarkvNoneOfTheAbove, 5761 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 15893 },
      { 899570100, 180 },
      { 929101967, 414 },
      { 1100599986, 180 },
      { 1951208733, 2972 },
      { 2320303498, 4214 },
      { 3056042030, 22039 },
      { 3334207724, 5051 },
      { 3357250579, 280 },
      { 3705139860, 490 },
      { 3800912395, 879 },
      { 3802564010, 228 },
      { kMarkvNoneOfTheAbove, 992 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 283209196, 124 },
      { 436416061, 324 },
      { 679771963, 1633 },
      { 789872778, 154 },
      { 815757910, 312 },
      { 827698488, 1708 },
      { 1164221089, 104 },
      { 1294403159, 4904 },
      { 1296054774, 10773 },
      { 1297706389, 5700 },
      { 1525861001, 190 },
      { 1579585816, 312 },
      { 1675764636, 338 },
      { 1824016656, 260 },
      { 1951208733, 126 },
      { 1991787192, 168 },
      { 2180701723, 124 },
      { 2194691858, 741 },
      { 2320303498, 1378 },
      { 2881886868, 224 },
      { 2926633629, 672 },
      { 3249265647, 312 },
      { 3334207724, 1481 },
      { 3472123498, 104 },
      { 3674863070, 224 },
      { 4050155669, 261 },
      { 4141567741, 936 },
      { 4155122613, 214 },
      { kMarkvNoneOfTheAbove, 802 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 956 },
      { 827698488, 2297 },
      { 1294403159, 1268 },
      { 1296054774, 6049 },
      { 1297706389, 817 },
      { 1951208733, 627 },
      { 2320303498, 1699 },
      { 3334207724, 521 },
      { kMarkvNoneOfTheAbove, 171 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 699 },
      { 827698488, 1143 },
      { 1294403159, 688 },
      { 1296054774, 5342 },
      { 1951208733, 64 },
      { 2194691858, 162 },
      { 2320303498, 583 },
      { 3334207724, 94 },
      { kMarkvNoneOfTheAbove, 98 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 2000 },
      { 1294403159, 800 },
      { 1296054774, 2984 },
      { 1297706389, 80 },
      { 1951208733, 286 },
      { kMarkvNoneOfTheAbove, 75 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 1584 },
      { 1294403159, 79 },
      { 1296054774, 799 },
      { 1951208733, 155 },
      { 3334207724, 165 },
      { kMarkvNoneOfTheAbove, 38 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 5 },
      { 827698488, 1051 },
      { 1951208733, 236 },
      { 2320303498, 162 },
      { kMarkvNoneOfTheAbove, 16 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543621065, 280 },
      { 827698488, 769 },
      { 1951208733, 79 },
      { kMarkvNoneOfTheAbove, 16 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 769 },
      { 1951208733, 77 },
      { 3095831808, 280 },
      { kMarkvNoneOfTheAbove, 15 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1296054774, 768 },
      { 1951208733, 3 },
      { kMarkvNoneOfTheAbove, 8 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1296054774, 768 },
      { 2320303498, 3 },
      { kMarkvNoneOfTheAbove, 7 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 2 },
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 1 },
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 169625 },
      { 1951208733, 247571 },
      { 2430404313, 48818 },
      { kMarkvNoneOfTheAbove, 4660 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 51041423, 3459 },
      { 52882140, 3888 },
      { 72782198, 1968 },
      { 142465290, 10625 },
      { 144116905, 9543 },
      { 158160339, 3888 },
      { 169135842, 3888 },
      { 210116709, 9474 },
      { 290391815, 2725 },
      { 296981500, 18701 },
      { 385229009, 3888 },
      { 438318340, 1982 },
      { 529742207, 8164 },
      { 628331516, 1945 },
      { 677668732, 3893 },
      { 778500192, 1440 },
      { 825595257, 1887 },
      { 910398460, 3904 },
      { 917019124, 2361 },
      { 959681532, 17499 },
      { 1031290113, 3888 },
      { 1039111164, 2431 },
      { 1064945649, 2861 },
      { 1087394637, 3888 },
      { 1092948665, 4395 },
      { 1156369516, 10624 },
      { 1158021131, 8511 },
      { 1172110445, 3888 },
      { 1304296041, 4495 },
      { 1400019344, 1463 },
      { 1450415100, 2003 },
      { 1452222566, 2906 },
      { 1543646433, 6153 },
      { 1543672828, 2906 },
      { 1612361408, 1440 },
      { 1622381564, 2394 },
      { 1691572958, 3622 },
      { 1755648697, 3904 },
      { 1782996825, 10684 },
      { 1784648440, 9177 },
      { 1930923350, 3069 },
      { 1939359710, 1440 },
      { 1971252067, 3888 },
      { 1979847999, 1955 },
      { 2078849875, 1508 },
      { 2113115132, 1787 },
      { 2135340676, 2904 },
      { 2170273742, 9233 },
      { 2268204687, 1440 },
      { 2285081596, 1619 },
      { 2318200267, 3805 },
      { 2321729979, 4888 },
      { 2326636627, 1463 },
      { 2444465148, 8554 },
      { 2466126792, 2351 },
      { 2490492987, 3888 },
      { 2524697596, 5774 },
      { 2557550659, 4972 },
      { 2678954464, 3888 },
      { 2705477184, 3802 },
      { 2715370488, 3062 },
      { 2732195517, 3888 },
      { 2775815164, 1962 },
      { 2796901051, 10748 },
      { 2798552666, 18329 },
      { 2855506940, 18523 },
      { 2860348412, 2127 },
      { 2922615804, 1603 },
      { 2937761472, 1440 },
      { 2944827576, 2720 },
      { 3092754101, 1707 },
      { 3107165180, 2024 },
      { 3168953855, 1750 },
      { 3184177968, 7338 },
      { 3202349435, 3888 },
      { 3266548732, 4445 },
      { 3332104493, 3796 },
      { 3362723943, 3888 },
      { 3571454885, 3863 },
      { 3712763835, 3888 },
      { 3743748793, 1937 },
      { 3810805277, 9648 },
      { 3912967080, 3888 },
      { 3929248764, 14579 },
      { 3958731802, 4296 },
      { 3997952447, 1440 },
      { 4016096296, 3888 },
      { 4106658327, 2422 },
      { 4172568578, 1463 },
      { 4198082194, 4193 },
      { 4248015868, 2881 },
      { kMarkvNoneOfTheAbove, 40067 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 55763 },
      { 1247793383, 17659 },
      { 2320303498, 48695 },
      { 3334207724, 40225 },
      { kMarkvNoneOfTheAbove, 1640 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 15502752, 4088 },
      { 46736908, 804 },
      { 139011596, 13776 },
      { 149720480, 800 },
      { 249378857, 3769 },
      { 251209228, 1945 },
      { 503145996, 2894 },
      { 836581417, 801 },
      { 882718761, 1600 },
      { 1289566249, 816 },
      { 1325348861, 3888 },
      { 1558001705, 2428 },
      { 1646147798, 3888 },
      { 1679946323, 3888 },
      { 1766401548, 5798 },
      { 1992893964, 2740 },
      { 2123388694, 3888 },
      { 2162986400, 2642 },
      { 2580096524, 2873 },
      { 2598189097, 2428 },
      { 2683080096, 8245 },
      { 2698156268, 3888 },
      { 2763960513, 1463 },
      { 3015046341, 1463 },
      { 3133016299, 3888 },
      { 3251128023, 9474 },
      { 3504158761, 10890 },
      { 3535289452, 1445 },
      { 3536941067, 1444 },
      { 3538592682, 4828 },
      { 3540244297, 4831 },
      { 3541895912, 4831 },
      { 3570219049, 2428 },
      { 3653838348, 7071 },
      { 3764205609, 8382 },
      { 3882634684, 3888 },
      { 3913885196, 7043 },
      { 3982047273, 1228 },
      { 4024252457, 805 },
      { 4243119782, 3888 },
      { 4255182614, 1440 },
      { kMarkvNoneOfTheAbove, 5365 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 4832 },
      { 158160339, 3888 },
      { 169135842, 3888 },
      { 210116709, 9474 },
      { 296981500, 23262 },
      { 615748604, 1208 },
      { 910398460, 1951 },
      { 959681532, 3725 },
      { 1039111164, 2430 },
      { 1087394637, 3888 },
      { 1156369516, 4828 },
      { 1450415100, 809 },
      { 1543672828, 1228 },
      { 2100532220, 1228 },
      { 2170273742, 1444 },
      { 2285081596, 1600 },
      { 2326636627, 1463 },
      { 2444465148, 5689 },
      { 2732195517, 3888 },
      { 2763232252, 1228 },
      { 2796901051, 4831 },
      { 2855506940, 41703 },
      { 2922615804, 800 },
      { 2937761472, 1440 },
      { 3202349435, 3888 },
      { 3362723943, 3888 },
      { 3712763835, 3888 },
      { 3810805277, 1444 },
      { 3929248764, 8615 },
      { 4016096296, 3888 },
      { 4172568578, 1463 },
      { 4248015868, 2007 },
      { kMarkvNoneOfTheAbove, 4176 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 52882140, 3888 },
      { 210116709, 9474 },
      { 296981500, 23181 },
      { 385229009, 3888 },
      { 615748604, 1207 },
      { 910398460, 1947 },
      { 959681532, 3668 },
      { 1031290113, 3888 },
      { 1039111164, 2430 },
      { 1172110445, 3888 },
      { 1450415100, 811 },
      { 1543672828, 1228 },
      { 1622381564, 1650 },
      { 1782996825, 17658 },
      { 1971252067, 3888 },
      { 2100532220, 1228 },
      { 2268204687, 1440 },
      { 2326636627, 1463 },
      { 2444465148, 5664 },
      { 2490492987, 3888 },
      { 2678954464, 3888 },
      { 2763232252, 1228 },
      { 2855506940, 41580 },
      { 2922615804, 800 },
      { 3912967080, 3888 },
      { 3929248764, 8710 },
      { 4172568578, 1463 },
      { 4248015868, 2011 },
      { kMarkvNoneOfTheAbove, 4037 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 210116709, 9474 },
      { 296981500, 28114 },
      { 615748604, 1200 },
      { 910398460, 1945 },
      { 959681532, 3559 },
      { 1039111164, 2429 },
      { 1092948665, 472 },
      { 1450415100, 810 },
      { 1543672828, 1228 },
      { 1612361408, 1440 },
      { 2100532220, 1228 },
      { 2326636627, 1463 },
      { 2444465148, 7138 },
      { 2524697596, 2894 },
      { 2763232252, 1228 },
      { 2855506940, 22434 },
      { 3929248764, 4518 },
      { 4172568578, 1463 },
      { 4248015868, 2004 },
      { kMarkvNoneOfTheAbove, 1915 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 4864 },
      { 615748604, 1200 },
      { 673708384, 133 },
      { 959681532, 810 },
      { 1039111164, 2428 },
      { 1450415100, 804 },
      { 1543672828, 1228 },
      { 1939359710, 1440 },
      { 2100532220, 1228 },
      { 2113115132, 1606 },
      { 2326636627, 1463 },
      { 2444465148, 921 },
      { 2763232252, 1228 },
      { 2855506940, 16480 },
      { 3929248764, 1657 },
      { 4172568578, 1463 },
      { 4248015868, 1200 },
      { kMarkvNoneOfTheAbove, 477 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 21653 },
      { 929101967, 414 },
      { 1951208733, 4481 },
      { 2320303498, 4760 },
      { 3056042030, 22105 },
      { 3334207724, 6029 },
      { 3357250579, 280 },
      { 3705139860, 490 },
      { 3800912395, 879 },
      { 3802564010, 228 },
      { kMarkvNoneOfTheAbove, 1447 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 35240468, 360 },
      { 123060826, 400 },
      { 184634770, 312 },
      { 359054425, 463 },
      { 459968607, 280 },
      { 619875033, 211 },
      { 904486530, 237 },
      { 945128292, 291 },
      { 950731750, 234 },
      { 1058429216, 312 },
      { 1182296898, 528 },
      { 1238120570, 768 },
      { 1429389803, 1866 },
      { 1652168174, 214 },
      { 1717510093, 768 },
      { 1766422419, 768 },
      { 1775308984, 1605 },
      { 1776629361, 278 },
      { 1824526196, 580 },
      { 1957265068, 1850 },
      { 1998433745, 800 },
      { 2055664760, 3192 },
      { 2303184249, 5849 },
      { 2451531615, 853 },
      { 2507457870, 842 },
      { 2550501832, 768 },
      { 2590402790, 240 },
      { 2649103430, 2133 },
      { 2780190687, 253 },
      { 2831059514, 1596 },
      { 3167253437, 211 },
      { 3269075805, 2239 },
      { 3323202731, 828 },
      { 3361419439, 1339 },
      { 3464197236, 330 },
      { 3472029049, 530 },
      { 3518630848, 312 },
      { 3604842236, 18882 },
      { 3653985133, 921 },
      { 4091916710, 190 },
      { 4121643374, 324 },
      { 4185590212, 338 },
      { 4233562270, 860 },
      { 4235213885, 274 },
      { kMarkvNoneOfTheAbove, 7105 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 75986790, 18882 },
      { 95470391, 768 },
      { 170378107, 315 },
      { 172029722, 921 },
      { 204234270, 368 },
      { 205885885, 312 },
      { 244668133, 768 },
      { 265778447, 360 },
      { 753954113, 291 },
      { 1000070091, 800 },
      { 1671139745, 338 },
      { 1774874546, 240 },
      { 1776526161, 330 },
      { 1887808856, 1850 },
      { 1889460471, 1596 },
      { 1917966999, 530 },
      { 2044728014, 528 },
      { 2192810893, 312 },
      { 2293247016, 2239 },
      { 2503194620, 842 },
      { 2608484640, 768 },
      { 2615111110, 463 },
      { 2668769415, 768 },
      { 2759951687, 1866 },
      { 2761603302, 5849 },
      { 2856623532, 280 },
      { 2956189845, 3192 },
      { 3085119011, 190 },
      { 3367313400, 214 },
      { 3447882276, 400 },
      { 3633746133, 1605 },
      { 3635397748, 2133 },
      { 3710645347, 234 },
      { 3712296962, 580 },
      { 3727494858, 853 },
      { 3747079365, 828 },
      { 3748965853, 238 },
      { 3750617468, 278 },
      { 4018820793, 211 },
      { 4022124023, 211 },
      { 4024173916, 324 },
      { 4215670524, 1339 },
      { 4217322139, 1134 },
      { kMarkvNoneOfTheAbove, 6986 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 436416061, 324 },
      { 543621065, 280 },
      { 679771963, 3414 },
      { 815757910, 312 },
      { 827698488, 11457 },
      { 1294403159, 8751 },
      { 1296054774, 39135 },
      { 1297706389, 7175 },
      { 1579585816, 312 },
      { 1675764636, 338 },
      { 1824016656, 260 },
      { 1951208733, 1668 },
      { 2194691858, 903 },
      { 2320303498, 3837 },
      { 2926633629, 672 },
      { 3095831808, 280 },
      { 3249265647, 312 },
      { 3334207724, 2781 },
      { 4050155669, 286 },
      { 4141567741, 936 },
      { kMarkvNoneOfTheAbove, 3005 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionParameter, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 522971108, 39135 },
      { 615341051, 324 },
      { 718301639, 280 },
      { 985750227, 286 },
      { 1395113939, 7175 },
      { 1510333659, 1668 },
      { 1642805350, 312 },
      { 1846856260, 312 },
      { 1957218950, 11457 },
      { 1977038330, 2781 },
      { 1978689945, 3414 },
      { 1980341560, 3837 },
      { 2262220987, 936 },
      { 2674422363, 672 },
      { 3197739982, 280 },
      { 3465954368, 903 },
      { 3941049054, 260 },
      { 3945795573, 8751 },
      { 4080527786, 338 },
      { 4154758669, 312 },
      { kMarkvNoneOfTheAbove, 3005 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionParameter, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 23062 },
      { 899570100, 180 },
      { 929101967, 414 },
      { 1100599986, 180 },
      { 1951208733, 4946 },
      { 2320303498, 5402 },
      { 3056042030, 3606 },
      { 3334207724, 6609 },
      { 3357250579, 280 },
      { 3705139860, 490 },
      { 3800912395, 891 },
      { 3802564010, 251 },
      { kMarkvNoneOfTheAbove, 969 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 57149555, 549 },
      { 86116519, 768 },
      { 168339452, 256 },
      { 181902171, 5625 },
      { 284226441, 240 },
      { 314809953, 495 },
      { 330249537, 2312 },
      { 527665290, 168 },
      { 545363837, 800 },
      { 707478563, 243 },
      { 740921498, 567 },
      { 807276090, 643 },
      { 824323032, 495 },
      { 835458563, 169 },
      { 1162127370, 927 },
      { 1245448751, 768 },
      { 1277245109, 160 },
      { 1375043498, 312 },
      { 1380991098, 211 },
      { 1603937321, 1603 },
      { 1708264968, 528 },
      { 1717555224, 400 },
      { 1765126703, 145 },
      { 1838993983, 249 },
      { 1949856502, 2134 },
      { 2108571893, 235 },
      { 2110223508, 278 },
      { 2293637521, 828 },
      { 2377112119, 206 },
      { 2378763734, 575 },
      { 2512398201, 185 },
      { 2516325050, 280 },
      { 2645135839, 339 },
      { 2708915136, 1779 },
      { 2894979602, 768 },
      { 2903897222, 344 },
      { 2976581453, 168 },
      { 3054834317, 1596 },
      { 3075866530, 168 },
      { 3085157904, 185 },
      { 3242843022, 169 },
      { 3266028549, 828 },
      { 3296691317, 181 },
      { 3299488628, 519 },
      { 3322500634, 168 },
      { 3345707173, 434 },
      { 3536390697, 1197 },
      { 3584683259, 182 },
      { 3647606635, 768 },
      { 3760372982, 253 },
      { 3823959661, 312 },
      { 3839389658, 324 },
      { 4124281183, 237 },
      { 4130950286, 1632 },
      { 4169878842, 312 },
      { 4174489262, 3192 },
      { 4237497041, 330 },
      { kMarkvNoneOfTheAbove, 8309 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 35240468, 344 },
      { 36096192, 168 },
      { 123060826, 400 },
      { 184634770, 312 },
      { 459968607, 280 },
      { 619875033, 211 },
      { 950731750, 136 },
      { 1058429216, 312 },
      { 1182296898, 528 },
      { 1238120570, 768 },
      { 1271484400, 168 },
      { 1429389803, 1632 },
      { 1717510093, 768 },
      { 1766422419, 768 },
      { 1775308984, 1603 },
      { 1817271123, 182 },
      { 1917336504, 1197 },
      { 1957265068, 1779 },
      { 1998433745, 800 },
      { 2055664760, 3192 },
      { 2303184249, 5625 },
      { 2308565678, 128 },
      { 2451531615, 434 },
      { 2496297824, 168 },
      { 2507457870, 828 },
      { 2550501832, 768 },
      { 2590402790, 224 },
      { 2649103430, 2134 },
      { 2831059514, 1596 },
      { 2836440943, 124 },
      { 3269075805, 2312 },
      { 3361419439, 703 },
      { 3457269042, 168 },
      { 3464197236, 185 },
      { 3472029049, 549 },
      { 3518630848, 312 },
      { 3587381650, 122 },
      { 3653985133, 927 },
      { 4185590212, 339 },
      { 4233562270, 122 },
      { kMarkvNoneOfTheAbove, 3095 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 748 },
      { 162167595, 243 },
      { 535067202, 224 },
      { 701281393, 240 },
      { 837715723, 173 },
      { 1320550031, 190 },
      { 1630583316, 672 },
      { 1913735398, 260 },
      { 1918481917, 6162 },
      { 1955871800, 312 },
      { 1977038330, 294 },
      { 2053214130, 338 },
      { 2443959748, 828 },
      { 2564745684, 256 },
      { 2622612602, 662 },
      { 2677252364, 330 },
      { 2736026107, 253 },
      { 2790624748, 17618 },
      { 2882994691, 324 },
      { 2888125966, 168 },
      { 2970183398, 730 },
      { 3253403867, 1376 },
      { 3427283542, 224 },
      { 3570411982, 872 },
      { 3619787319, 185 },
      { 3662767579, 7119 },
      { 3884846406, 312 },
      { 3910458990, 312 },
      { 3927915220, 936 },
      { 4224872590, 2029 },
      { kMarkvNoneOfTheAbove, 2719 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 744062262, 182 },
      { 810488476, 55 },
      { 1040775722, 59 },
      { 1280126114, 181 },
      { 1367301635, 771 },
      { 1684282922, 107 },
      { 1918481917, 1552 },
      { 1978689945, 291 },
      { 1980341560, 830 },
      { 2443959748, 848 },
      { 2629265310, 120 },
      { 2790624748, 7956 },
      { 2970183398, 122 },
      { 3044188332, 185 },
      { 3496407048, 61 },
      { 3662767579, 873 },
      { 3887377256, 519 },
      { 3971481069, 102 },
      { 4224872590, 2432 },
      { kMarkvNoneOfTheAbove, 552 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 162 },
      { 837715723, 120 },
      { 1352628475, 145 },
      { 1918481917, 889 },
      { 1978689945, 410 },
      { 1980341560, 188 },
      { 2096388952, 78 },
      { 2622612602, 91 },
      { 2790624748, 6976 },
      { 2970183398, 873 },
      { 3510682541, 55 },
      { 3783543823, 122 },
      { 4224872590, 1235 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1510333659, 76 },
      { 1684282922, 191 },
      { 1918481917, 800 },
      { 2790624748, 4584 },
      { 3662767579, 137 },
      { 4224872590, 2039 },
      { kMarkvNoneOfTheAbove, 344 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 161668409, 19 },
      { 188347929, 13 },
      { 653708953, 11 },
      { 976111724, 69 },
      { 1510333659, 76 },
      { 1918481917, 80 },
      { 2790624748, 803 },
      { 3033873113, 11 },
      { 3499234137, 12 },
      { 3525913657, 16 },
      { 3552593177, 14 },
      { 3570411982, 160 },
      { 4224872590, 1601 },
      { kMarkvNoneOfTheAbove, 183 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 226836633, 10 },
      { 296981500, 14 },
      { 718877177, 6 },
      { 745556697, 5 },
      { 798915737, 6 },
      { 1510333659, 152 },
      { 1684282922, 69 },
      { 2444465148, 6 },
      { 2713718873, 6 },
      { 3495546641, 162 },
      { 3564402361, 9 },
      { 4056442905, 7 },
      { 4083122425, 7 },
      { 4123141705, 5 },
      { 4224872590, 1052 },
      { kMarkvNoneOfTheAbove, 103 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 161668409, 15 },
      { 188347929, 10 },
      { 215027449, 7 },
      { 296981500, 7 },
      { 653708953, 10 },
      { 680388473, 4 },
      { 1119069977, 4 },
      { 1510333659, 76 },
      { 1584774136, 280 },
      { 2049792025, 6 },
      { 2444465148, 7 },
      { 2568512089, 4 },
      { 3033873113, 10 },
      { 3499234137, 8 },
      { 3525913657, 11 },
      { 3552593177, 7 },
      { 4224872590, 769 },
      { kMarkvNoneOfTheAbove, 56 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 825595257, 37 },
      { 1064945649, 6 },
      { 1290956281, 4 },
      { 1510333659, 76 },
      { 2096388952, 10 },
      { 2248357849, 6 },
      { 2713718873, 7 },
      { 3187066832, 4 },
      { 3205759417, 4 },
      { 4064212479, 280 },
      { 4163160985, 17 },
      { 4224872590, 770 },
      { kMarkvNoneOfTheAbove, 64 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 123108003, 7 },
      { 296981500, 19 },
      { 595410904, 9 },
      { 1466938734, 11 },
      { 1503477720, 6 },
      { 1816558243, 11 },
      { 1990431740, 13 },
      { 2724625059, 7 },
      { 2790624748, 768 },
      { 2812498065, 22 },
      { 3160388974, 8 },
      { 3745223676, 8 },
      { 3982311384, 4 },
      { kMarkvNoneOfTheAbove, 16 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 94145952, 3 },
      { 1054641568, 18 },
      { 1269075360, 15 },
      { 1675922848, 8 },
      { 2038205856, 22 },
      { 2433519008, 5 },
      { 2636942752, 4 },
      { 2790624748, 768 },
      { 2840366496, 32 },
      { 2851900832, 6 },
      { 2964622752, 12 },
      { 3654061472, 5 },
      { kMarkvNoneOfTheAbove, 8 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 107544081, 2 },
      { 125015036, 1 },
      { 586244865, 10 },
      { 1033081852, 36 },
      { 1064945649, 2 },
      { 1155765244, 2 },
      { 1304296041, 1 },
      { 1543646433, 7 },
      { 1782996825, 7 },
      { 1941148668, 4 },
      { 2002490364, 8 },
      { 2022347217, 3 },
      { 2063832060, 1 },
      { 2487708241, 1 },
      { 2726532092, 2 },
      { 2849215484, 2 },
      { 2966409025, 2 },
      { 3445109809, 3 },
      { 3458449569, 8 },
      { 3634598908, 20 },
      { 3695940604, 5 },
      { 3923810593, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 247698428, 1 },
      { 309040124, 12 },
      { 333554713, 6 },
      { 572905105, 8 },
      { 1033081852, 2 },
      { 2002490364, 1 },
      { 2009007457, 1 },
      { 2487708241, 3 },
      { 3634598908, 1 },
      { 3695940604, 8 },
      { 3923810593, 6 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 247698428, 9 },
      { 1033081852, 5 },
      { 2002490364, 2 },
      { 2910557180, 1 },
      { 3757282300, 7 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1033081852, 11 },
      { 1094423548, 3 },
      { 2002490364, 2 },
      { 3757282300, 8 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 16), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 135486769, 4099 },
      { 450406196, 2879 },
      { 503094540, 4320 },
      { 543621065, 50382 },
      { 827698488, 31911 },
      { 1294403159, 40166 },
      { 1296054774, 73442 },
      { 1297706389, 23479 },
      { 1322549027, 7499 },
      { 1784441183, 3052 },
      { 2194691858, 48598 },
      { 2448331885, 3858 },
      { 2468230023, 51648 },
      { 2547657777, 59039 },
      { 2549309392, 12073 },
      { 2550961007, 19702 },
      { 2934934694, 53926 },
      { 2936586309, 17687 },
      { 2938237924, 14790 },
      { 3094180193, 6012 },
      { 3095831808, 43266 },
      { 3183924418, 10218 },
      { 3561562003, 51481 },
      { 3563213618, 3120 },
      { 3564865233, 3437 },
      { 4028622909, 13531 },
      { 4039938779, 2411 },
      { 4050155669, 2197 },
      { kMarkvNoneOfTheAbove, 46702 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 48598 },
      { 112745085, 2411 },
      { 137840602, 17687 },
      { 565334834, 7499 },
      { 625975427, 10218 },
      { 630964591, 19702 },
      { 680016782, 13531 },
      { 769422756, 3120 },
      { 1009983433, 14790 },
      { 1093210099, 2879 },
      { 1572088444, 51648 },
      { 1584774136, 50382 },
      { 1641565587, 3437 },
      { 1918481917, 40166 },
      { 2190437442, 4099 },
      { 2790624748, 73442 },
      { 3085467405, 3858 },
      { 3181646225, 59039 },
      { 3192069648, 6012 },
      { 3253403867, 2197 },
      { 3390051757, 4320 },
      { 3560665067, 53926 },
      { 3662767579, 23479 },
      { 4053789056, 12073 },
      { 4064212479, 43266 },
      { 4192247221, 51481 },
      { 4224872590, 31911 },
      { 4290024976, 3052 },
      { kMarkvNoneOfTheAbove, 46702 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 162255877, 63938 },
      { 679771963, 209209 },
      { 789872778, 5809 },
      { 1154919607, 8288 },
      { 1343794461, 6012 },
      { 1951208733, 375067 },
      { 2263349224, 6764 },
      { 2320303498, 97959 },
      { 2924146124, 7192 },
      { 2984325996, 56390 },
      { 3334207724, 326179 },
      { 3868239231, 10753 },
      { 3869890846, 61816 },
      { kMarkvNoneOfTheAbove, 50255 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 169674806, 4133 },
      { 269823086, 11590 },
      { 408465899, 9383 },
      { 451264926, 4321 },
      { 543558236, 56110 },
      { 810488476, 37985 },
      { 850497536, 7192 },
      { 870594305, 18683 },
      { 883854656, 60901 },
      { 1033363654, 6764 },
      { 1069781886, 6012 },
      { 1141965917, 10022 },
      { 1323407757, 4742 },
      { 1570165302, 21190 },
      { 1684282922, 26076 },
      { 1742737136, 9185 },
      { 1901166356, 15318 },
      { 1949759310, 63658 },
      { 2043873558, 8288 },
      { 2087004702, 19741 },
      { 2096388952, 88899 },
      { 2157103435, 9408 },
      { 2219733501, 12525 },
      { 2356768706, 10739 },
      { 2443959748, 6126 },
      { 2517964682, 37601 },
      { 2614879967, 8393 },
      { 2622612602, 62157 },
      { 2660843182, 8903 },
      { 2959147533, 9457 },
      { 2970183398, 24585 },
      { 3044188332, 11640 },
      { 3091876332, 14937 },
      { 3187066832, 46898 },
      { 3244209297, 5903 },
      { 3487022798, 7466 },
      { 3496407048, 52811 },
      { 3570411982, 68029 },
      { 3692647551, 12734 },
      { 3713290482, 11737 },
      { 3831290364, 10009 },
      { kMarkvNoneOfTheAbove, 363380 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 28782128, 4321 },
      { 30433743, 4742 },
      { 37459569, 60901 },
      { 137840602, 24585 },
      { 522971108, 46898 },
      { 565334834, 10739 },
      { 625975427, 7192 },
      { 630964591, 37601 },
      { 680016782, 8288 },
      { 1009983433, 6126 },
      { 1079999262, 9408 },
      { 1395113939, 8903 },
      { 1572088444, 26076 },
      { 1584774136, 63658 },
      { 1649426421, 5903 },
      { 1918481917, 62157 },
      { 1957218950, 15318 },
      { 2311941439, 12525 },
      { 2313593054, 14937 },
      { 2790624748, 88899 },
      { 2838165089, 12734 },
      { 2839816704, 11590 },
      { 2841468319, 10022 },
      { 3085467405, 6764 },
      { 3181646225, 68029 },
      { 3192069648, 6012 },
      { 3253403867, 4133 },
      { 3364388739, 18683 },
      { 3366040354, 9185 },
      { 3367691969, 8393 },
      { 3369343584, 7466 },
      { 3560665067, 52811 },
      { 3662767579, 21190 },
      { 3945795573, 11737 },
      { 4053789056, 11640 },
      { 4064212479, 56110 },
      { 4224872590, 37985 },
      { 4239834800, 19741 },
      { 4241486415, 9457 },
      { 4243138030, 10009 },
      { 4244789645, 9383 },
      { kMarkvNoneOfTheAbove, 363380 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 137840602, 23401 },
      { 522971108, 4329 },
      { 769422756, 3107 },
      { 1009983433, 14792 },
      { 1079999262, 5472 },
      { 1558345254, 3611 },
      { 1572088444, 54630 },
      { 1641565587, 3382 },
      { 1918481917, 58464 },
      { 2311941439, 3121 },
      { 2313593054, 2721 },
      { 2790624748, 110466 },
      { 2838165089, 3078 },
      { 2994529201, 2639 },
      { 2996180816, 2639 },
      { 2997832431, 2639 },
      { 3027538652, 2411 },
      { 3253403867, 3156 },
      { 3364388739, 4545 },
      { 3560665067, 61574 },
      { 3662767579, 27441 },
      { 3945795573, 3334 },
      { 4192247221, 52730 },
      { 4224872590, 50029 },
      { kMarkvNoneOfTheAbove, 107320 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 139011596, 2274 },
      { 177111659, 6083 },
      { 296981500, 6088 },
      { 408465899, 8381 },
      { 495107308, 2310 },
      { 810488476, 5745 },
      { 870594305, 3842 },
      { 1367301635, 2058 },
      { 1901166356, 2517 },
      { 2055836767, 8540 },
      { 2087004702, 16395 },
      { 2096388952, 13614 },
      { 2204920111, 5168 },
      { 2517964682, 2900 },
      { 2622612602, 8920 },
      { 2660843182, 2958 },
      { 2842919847, 4812 },
      { 2855506940, 9672 },
      { 2959147533, 8501 },
      { 3044188332, 1971 },
      { 3187066832, 14439 },
      { 3504158761, 6008 },
      { 3570411982, 7666 },
      { 3619787319, 2438 },
      { 3653838348, 2158 },
      { 3692647551, 2310 },
      { 3764205609, 3449 },
      { 3831290364, 8378 },
      { 3913885196, 1964 },
      { kMarkvNoneOfTheAbove, 439472 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 440421571, 82012 },
      { 827698488, 170411 },
      { 907126242, 55506 },
      { 908777857, 28054 },
      { 910429472, 10962 },
      { 1294403159, 90554 },
      { 1296054774, 11684 },
      { 1297706389, 16509 },
      { 2080953106, 18995 },
      { 2468230023, 62482 },
      { 2547657777, 11724 },
      { 2549309392, 2089 },
      { 2550961007, 2499 },
      { 3094857332, 4610 },
      { 3561562003, 16461 },
      { 3563213618, 2403 },
      { 3564865233, 3203 },
      { kMarkvNoneOfTheAbove, 8745 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 27130513, 3330 },
      { 28782128, 4321 },
      { 30433743, 4742 },
      { 32085358, 3293 },
      { 155458798, 2764 },
      { 157110413, 2445 },
      { 163402553, 2237 },
      { 165054168, 2291 },
      { 213642219, 3261 },
      { 215293834, 1886 },
      { 216945449, 2300 },
      { 221900294, 3725 },
      { 545986953, 2096 },
      { 979993429, 2020 },
      { 1079999262, 10176 },
      { 1302400505, 3016 },
      { 1313182965, 1888 },
      { 1314834580, 1887 },
      { 1315613425, 2342 },
      { 1317265040, 2337 },
      { 1558345254, 3611 },
      { 1649426421, 5903 },
      { 2311941439, 15646 },
      { 2313593054, 17658 },
      { 2602027658, 2160 },
      { 2838165089, 15812 },
      { 2839816704, 13281 },
      { 2841468319, 10107 },
      { 2863084840, 1886 },
      { 2994529201, 3915 },
      { 2996180816, 3915 },
      { 2997832431, 3915 },
      { 3027538652, 2411 },
      { 3187387500, 3522 },
      { 3189039115, 3552 },
      { 3364388739, 23228 },
      { 3366040354, 10563 },
      { 3367691969, 9286 },
      { 3369343584, 8554 },
      { 3716914380, 2908 },
      { 3928842969, 4085 },
      { 3930494584, 2626 },
      { 3932146199, 2475 },
      { 3945482286, 2160 },
      { 4105051793, 2419 },
      { 4239834800, 20420 },
      { 4241486415, 9510 },
      { 4243138030, 10095 },
      { 4244789645, 9695 },
      { kMarkvNoneOfTheAbove, 311228 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 112745085, 6022 },
      { 116376005, 11317 },
      { 137840602, 4870 },
      { 400248103, 8928 },
      { 406044930, 14594 },
      { 468372467, 2568 },
      { 522971108, 9186 },
      { 615341051, 4736 },
      { 625975427, 28360 },
      { 630964591, 2137 },
      { 680016782, 27906 },
      { 763027711, 2560 },
      { 977312655, 5120 },
      { 1009983433, 7074 },
      { 1062250709, 9534 },
      { 1395113939, 2148 },
      { 1410849099, 10560 },
      { 1642805350, 2544 },
      { 1692932387, 3680 },
      { 1698730948, 3360 },
      { 1827244161, 3880 },
      { 1918481917, 51631 },
      { 2096472894, 2800 },
      { 2190437442, 23694 },
      { 2299842241, 2460 },
      { 2433358586, 1960 },
      { 2593325766, 2205 },
      { 2785441472, 4968 },
      { 2790624748, 39200 },
      { 2879917723, 13248 },
      { 2882994691, 24268 },
      { 2902069960, 1920 },
      { 3090408469, 4536 },
      { 3181646225, 15686 },
      { 3255947500, 2560 },
      { 3263901372, 11745 },
      { 3268751013, 3126 },
      { 3347863687, 2430 },
      { 3390051757, 24308 },
      { 3560665067, 49720 },
      { 3617689692, 2000 },
      { 3662767579, 33304 },
      { 3717523241, 8504 },
      { 3854557817, 1836 },
      { 3910458990, 2184 },
      { 3941049054, 2928 },
      { 3945795573, 6897 },
      { 4080527786, 4088 },
      { 4101009465, 5940 },
      { 4290024976, 38424 },
      { kMarkvNoneOfTheAbove, 39249 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 51041423, 5073 },
      { 142465290, 29886 },
      { 144116905, 65121 },
      { 290391815, 4842 },
      { 438318340, 2203 },
      { 529742207, 14081 },
      { 677668732, 3519 },
      { 917019124, 2972 },
      { 1064945649, 5290 },
      { 1156369516, 25119 },
      { 1158021131, 48703 },
      { 1304296041, 5215 },
      { 1452222566, 2312 },
      { 1543646433, 13130 },
      { 1691572958, 3742 },
      { 1782996825, 98487 },
      { 1784648440, 63380 },
      { 1930923350, 5938 },
      { 2170273742, 24423 },
      { 2318200267, 3425 },
      { 2466126792, 2210 },
      { 2557550659, 7744 },
      { 2705477184, 14450 },
      { 2796901051, 24244 },
      { 2798552666, 47648 },
      { 2944827576, 2676 },
      { 3092754101, 2021 },
      { 3184177968, 14454 },
      { 3332104493, 4702 },
      { 3571454885, 5088 },
      { 3810805277, 19142 },
      { 3958731802, 4802 },
      { 4106658327, 2018 },
      { 4198082194, 10573 },
      { kMarkvNoneOfTheAbove, 10270 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 5160 },
      { 144116905, 5552 },
      { 198967948, 917 },
      { 290391815, 371 },
      { 529742207, 1024 },
      { 586244865, 826 },
      { 677668732, 367 },
      { 825595257, 1017 },
      { 917019124, 790 },
      { 973521782, 792 },
      { 1064945649, 275 },
      { 1156369516, 3980 },
      { 1158021131, 15046 },
      { 1212872174, 843 },
      { 1304296041, 371 },
      { 1452222566, 1073 },
      { 1543646433, 676 },
      { 1600149091, 826 },
      { 1782996825, 8937 },
      { 1784648440, 8065 },
      { 1839499483, 1017 },
      { 1930923350, 658 },
      { 2170273742, 1188 },
      { 2226776400, 752 },
      { 2318200267, 371 },
      { 2466126792, 1004 },
      { 2557550659, 807 },
      { 2614053317, 826 },
      { 2796901051, 4614 },
      { 2798552666, 5678 },
      { 2853403709, 843 },
      { 2944827576, 371 },
      { 3184177968, 1004 },
      { 3240680626, 826 },
      { 3480031018, 1017 },
      { 3571454885, 674 },
      { 3810805277, 1299 },
      { 3867307935, 843 },
      { 3958731802, 371 },
      { 4106658327, 1073 },
      { 4198082194, 896 },
      { 4254584852, 826 },
      { kMarkvNoneOfTheAbove, 6547 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 144116905, 1633 },
      { 1158021131, 1633 },
      { 1784648440, 1645 },
      { 2798552666, 1633 },
      { kMarkvNoneOfTheAbove, 109 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 1 },
      { 1782996825, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 269626 },
      { 2320303498, 183291 },
      { 3334207724, 118614 },
      { kMarkvNoneOfTheAbove, 5748 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 177111659, 7269 },
      { 413918748, 1938 },
      { 529383565, 3274 },
      { 646282397, 2524 },
      { 837715723, 8081 },
      { 1019457583, 4351 },
      { 1022544883, 2009 },
      { 1054461787, 4395 },
      { 1097775533, 2112 },
      { 1136775085, 2800 },
      { 1191015885, 1901 },
      { 1196280518, 1978 },
      { 1203545131, 27250 },
      { 1352628475, 7058 },
      { 1367301635, 20971 },
      { 1918742169, 2832 },
      { 1922045399, 2832 },
      { 2055836767, 16365 },
      { 2183547611, 2112 },
      { 2204920111, 7357 },
      { 2358141757, 2112 },
      { 2572638469, 3936 },
      { 2597020383, 1824 },
      { 2842919847, 4827 },
      { 3619787319, 21822 },
      { 3701632935, 9269 },
      { 3783543823, 3280 },
      { 4245257809, 2696 },
      { 4265894873, 2112 },
      { kMarkvNoneOfTheAbove, 395992 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 236660303, 4474 },
      { 342159236, 2832 },
      { 371428004, 4876 },
      { 373079619, 3654 },
      { 488500848, 6483 },
      { 495107308, 12067 },
      { 864295921, 3116 },
      { 1071164424, 1784 },
      { 1136911283, 1922 },
      { 1178317551, 2112 },
      { 1510422521, 2800 },
      { 1570165302, 2477 },
      { 1822823090, 2911 },
      { 1858116930, 2284 },
      { 1977038330, 2765 },
      { 2096388952, 10459 },
      { 2157103435, 8640 },
      { 2231688008, 1824 },
      { 2604576561, 2444 },
      { 2622612602, 45615 },
      { 2771938750, 2112 },
      { 2777172031, 3168 },
      { 2996594997, 2832 },
      { 3187066832, 2268 },
      { 3496407048, 45072 },
      { 3570411982, 68988 },
      { 3609540589, 1939 },
      { 3713290482, 5842 },
      { 3797761273, 4224 },
      { kMarkvNoneOfTheAbove, 315295 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 236660303, 4474 },
      { 342159236, 2832 },
      { 488500848, 5064 },
      { 495107308, 11673 },
      { 864295921, 3116 },
      { 1178317551, 2112 },
      { 1510422521, 2800 },
      { 1570165302, 3307 },
      { 1858116930, 2284 },
      { 1977038330, 2765 },
      { 2096388952, 10619 },
      { 2157103435, 3936 },
      { 2231688008, 1824 },
      { 2604576561, 2444 },
      { 2622612602, 29950 },
      { 2771938750, 2112 },
      { 2777172031, 3168 },
      { 2996594997, 2832 },
      { 3496407048, 38031 },
      { 3570411982, 68437 },
      { 3609540589, 1939 },
      { 3713290482, 4064 },
      { 3797761273, 4224 },
      { kMarkvNoneOfTheAbove, 363272 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 199758 },
      { 1146476634, 1600 },
      { 2160380860, 1367 },
      { 2320303498, 76406 },
      { 3334207724, 144141 },
      { 3800912395, 1821 },
      { 3802564010, 3542 },
      { kMarkvNoneOfTheAbove, 4437 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 107497541, 1308 },
      { 289648234, 1416 },
      { 348584153, 1416 },
      { 369686787, 1341 },
      { 429277936, 2569 },
      { 449954059, 1356 },
      { 508217552, 2861 },
      { 742917749, 1368 },
      { 1032593647, 1429 },
      { 1158929937, 1453 },
      { 1209418480, 2086 },
      { 1319785741, 6170 },
      { 1321616112, 1944 },
      { 1417363940, 1416 },
      { 1541020250, 1629 },
      { 1564342316, 1656 },
      { 1578775276, 4224 },
      { 1631434666, 1560 },
      { 1636389511, 1360 },
      { 2012838864, 2040 },
      { 2262137600, 2112 },
      { 2281956980, 2136 },
      { 2359973133, 1958 },
      { 2464905186, 1368 },
      { 2613179511, 1338 },
      { 2621255555, 1872 },
      { 2817335337, 1747 },
      { 2881302403, 1843 },
      { 3063300848, 2711 },
      { 3151638847, 3256 },
      { 3233393284, 3130 },
      { 3323682385, 1644 },
      { 3337532056, 2112 },
      { 3456899824, 1416 },
      { 3547456240, 1820 },
      { 3675926744, 2112 },
      { 3753486980, 9693 },
      { 3931641900, 1550 },
      { 3970432934, 1368 },
      { kMarkvNoneOfTheAbove, 347284 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 153013225, 1560 },
      { 296836635, 1957 },
      { 296981500, 17937 },
      { 778500192, 1440 },
      { 810488476, 3300 },
      { 848380423, 1416 },
      { 900522183, 3168 },
      { 910398460, 1948 },
      { 959681532, 5276 },
      { 1141965917, 7559 },
      { 1287304304, 4224 },
      { 1323407757, 1453 },
      { 1417363940, 1416 },
      { 1471851763, 1368 },
      { 1526654696, 2136 },
      { 1654776395, 1416 },
      { 1684282922, 1986 },
      { 1739837626, 1368 },
      { 1791352211, 1568 },
      { 2195550588, 2041 },
      { 2319227476, 1496 },
      { 2491124112, 1656 },
      { 2789375411, 1356 },
      { 2807448986, 1560 },
      { 2817579280, 1550 },
      { 2835131395, 1368 },
      { 2847102741, 1485 },
      { 2855506940, 10484 },
      { 2860348412, 2338 },
      { 3079287749, 3168 },
      { 3091876332, 5830 },
      { 3168953855, 1747 },
      { 3374978006, 1419 },
      { 3399062057, 1416 },
      { 3510257966, 1368 },
      { 3554463148, 2112 },
      { 3579593979, 1644 },
      { 3757851979, 1331 },
      { 3759503594, 1392 },
      { 3761155209, 1402 },
      { 3762806824, 1471 },
      { 3902853271, 1299 },
      { 4140081844, 2112 },
      { kMarkvNoneOfTheAbove, 317531 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 14244860, 1416 },
      { 150820676, 1568 },
      { 153013225, 1560 },
      { 269823086, 7477 },
      { 289648234, 1416 },
      { 296981500, 17955 },
      { 678695941, 1496 },
      { 810488476, 3272 },
      { 850592577, 1656 },
      { 870594305, 1469 },
      { 910398460, 1948 },
      { 959681532, 12976 },
      { 1206571206, 1485 },
      { 1287304304, 4224 },
      { 1323407757, 3494 },
      { 1471851763, 1368 },
      { 1526654696, 2136 },
      { 1684282922, 1986 },
      { 1734446471, 1419 },
      { 1758530522, 1416 },
      { 2117320444, 1331 },
      { 2118972059, 1392 },
      { 2120623674, 1402 },
      { 2122275289, 1471 },
      { 2219733501, 5692 },
      { 2262321736, 1299 },
      { 2807448986, 1560 },
      { 2817579280, 1550 },
      { 2835131395, 1368 },
      { 2855506940, 21902 },
      { 2860348412, 2338 },
      { 2951272396, 1957 },
      { 3079287749, 2112 },
      { 3168953855, 1747 },
      { 3502816184, 1416 },
      { 3510257966, 1368 },
      { 3554463148, 2112 },
      { 3997952447, 1440 },
      { 4140081844, 2112 },
      { 4182141402, 1644 },
      { kMarkvNoneOfTheAbove, 304122 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 34183582, 1248 },
      { 93914936, 1384 },
      { 94303122, 1184 },
      { 117998987, 1416 },
      { 153013225, 1560 },
      { 296981500, 8290 },
      { 451264926, 3014 },
      { 473485679, 1121 },
      { 476788909, 1298 },
      { 478440524, 1287 },
      { 480092139, 1339 },
      { 481743754, 1414 },
      { 810488476, 2942 },
      { 871966503, 1056 },
      { 910398460, 1948 },
      { 918189168, 1800 },
      { 933769938, 1073 },
      { 959681532, 4669 },
      { 1149665466, 1188 },
      { 1166917451, 1104 },
      { 1227221002, 3663 },
      { 1310740861, 1957 },
      { 1323407757, 1453 },
      { 1341516288, 1620 },
      { 1373166395, 1056 },
      { 1445161581, 1620 },
      { 1461645203, 1073 },
      { 1471851763, 1368 },
      { 1526654696, 2136 },
      { 1561718045, 3456 },
      { 1593584949, 3150 },
      { 1684282922, 1657 },
      { 1800404122, 1054 },
      { 1862284649, 1416 },
      { 2213411495, 1056 },
      { 2668680621, 1416 },
      { 2805256437, 1509 },
      { 2807448986, 1560 },
      { 2835131395, 1368 },
      { 2855506940, 12748 },
      { 2860348412, 2338 },
      { 3000904950, 3150 },
      { 3107413701, 1620 },
      { 3168953855, 1747 },
      { 3333131702, 1430 },
      { 3365041621, 1248 },
      { 3456899824, 1416 },
      { 3505028338, 1656 },
      { 3510257966, 1368 },
      { 3554463148, 2112 },
      { 3606320646, 1620 },
      { 3692647551, 7146 },
      { 3861006967, 1376 },
      { 4126287524, 1128 },
      { 4140081844, 2112 },
      { kMarkvNoneOfTheAbove, 236143 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 18776483, 1600 },
      { 37009196, 1013 },
      { 277023757, 512 },
      { 296981500, 23843 },
      { 348988933, 813 },
      { 451264926, 973 },
      { 564884461, 999 },
      { 804899022, 826 },
      { 810488476, 835 },
      { 870594305, 1250 },
      { 876864198, 826 },
      { 900522183, 1056 },
      { 928261291, 1055 },
      { 959681532, 1962 },
      { 1164724902, 1429 },
      { 1323407757, 1453 },
      { 1332774287, 826 },
      { 1404739463, 826 },
      { 1447712361, 814 },
      { 1450415100, 938 },
      { 1513770932, 800 },
      { 1620634991, 879 },
      { 1692600167, 1218 },
      { 1860649552, 826 },
      { 1932614728, 926 },
      { 2087004702, 962 },
      { 2148510256, 783 },
      { 2220475432, 1338 },
      { 2388524817, 826 },
      { 2460489993, 826 },
      { 2676385521, 788 },
      { 2748350697, 1262 },
      { 2855506940, 19013 },
      { 2860348412, 518 },
      { 2916400082, 792 },
      { 2988365258, 512 },
      { 3061856840, 790 },
      { 3063508455, 790 },
      { 3065160070, 790 },
      { 3066811685, 790 },
      { 3068463300, 790 },
      { 3070114915, 790 },
      { 3071766530, 489 },
      { 3073418145, 569 },
      { 3075069760, 489 },
      { 3076721375, 489 },
      { 3078372990, 790 },
      { 3080024605, 890 },
      { 3081676220, 790 },
      { 3083327835, 790 },
      { 3084979450, 781 },
      { 3086631065, 845 },
      { 3088282680, 925 },
      { 3114708520, 602 },
      { 3116360135, 614 },
      { 3118011750, 723 },
      { 3119663365, 719 },
      { 3121314980, 825 },
      { 3124618210, 936 },
      { 3126269825, 949 },
      { 3127921440, 1121 },
      { 3129573055, 913 },
      { 3131224670, 1181 },
      { 3132876285, 1251 },
      { 3134527900, 1128 },
      { 3136179515, 1341 },
      { 3204260786, 674 },
      { 3264086791, 1056 },
      { 3276225962, 978 },
      { 3444275347, 792 },
      { 3516240523, 512 },
      { 3588205699, 973 },
      { 3732136051, 658 },
      { 3804101227, 1188 },
      { 3874089391, 1738 },
      { 4044115788, 603 },
      { 4116080964, 882 },
      { kMarkvNoneOfTheAbove, 35901 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 60816 },
      { 789872778, 5176 },
      { 1951208733, 780809 },
      { 2320303498, 31784 },
      { 3334207724, 101284 },
      { kMarkvNoneOfTheAbove, 15011 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 126463145, 4224 },
      { 171307615, 5440 },
      { 342159236, 4432 },
      { 354479447, 7192 },
      { 593829839, 4272 },
      { 743407979, 4224 },
      { 898191441, 3256 },
      { 900522183, 6336 },
      { 1265796414, 8288 },
      { 1287304304, 4224 },
      { 1356063462, 3016 },
      { 1368383673, 4224 },
      { 1526654696, 4272 },
      { 1766994680, 3168 },
      { 1793544760, 3000 },
      { 1811839150, 4480 },
      { 2234361374, 4272 },
      { 2279700640, 3200 },
      { 2383939514, 4224 },
      { 2780898906, 3168 },
      { 2996594997, 4432 },
      { 3413713311, 5246 },
      { 3554463148, 4224 },
      { 3635542517, 4272 },
      { kMarkvNoneOfTheAbove, 887794 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 13107491, 6076 },
      { 257136089, 3200 },
      { 293528591, 3236 },
      { 321459212, 4273 },
      { 425022309, 3977 },
      { 490769168, 4192 },
      { 495107308, 12488 },
      { 517919178, 6432 },
      { 617312262, 3920 },
      { 708736129, 3300 },
      { 753756604, 3516 },
      { 765238787, 3160 },
      { 796985462, 4480 },
      { 819503463, 4561 },
      { 850497536, 70744 },
      { 948086521, 5271 },
      { 1004589179, 3400 },
      { 1120149824, 3200 },
      { 1165671422, 3304 },
      { 1203545131, 5871 },
      { 1297165140, 3877 },
      { 1335363438, 3816 },
      { 1351676723, 4920 },
      { 1391866096, 3160 },
      { 1584369690, 32340 },
      { 1631216488, 3700 },
      { 1691646294, 5484 },
      { 1779143013, 3160 },
      { 1858116930, 3632 },
      { 1890300748, 18320 },
      { 1915438939, 4248 },
      { 1918742169, 4248 },
      { 1922045399, 4248 },
      { 1961990747, 5141 },
      { 2037710159, 4844 },
      { 2037814253, 3304 },
      { 2043873558, 54992 },
      { 2096388952, 3020 },
      { 2169307971, 3877 },
      { 2257843797, 4000 },
      { 2262220987, 3432 },
      { 2338272340, 50624 },
      { 2405770322, 3160 },
      { 2498042266, 4002 },
      { 2563789125, 5686 },
      { 2588618056, 5698 },
      { 2645120714, 3800 },
      { 2864863800, 3994 },
      { 2909957084, 3304 },
      { 2975894973, 4484 },
      { 3041450802, 3825 },
      { 3151638847, 3256 },
      { 3187066832, 3552 },
      { 3244716568, 4320 },
      { 3271748023, 3664 },
      { 3304438238, 3156 },
      { 3312467582, 37632 },
      { 3325419312, 4968 },
      { 3370185097, 4075 },
      { 3419674548, 3160 },
      { 3435931956, 5640 },
      { 3504158761, 10272 },
      { 3602522282, 5324 },
      { 3653059026, 8320 },
      { 3716353056, 3168 },
      { 3782099915, 3304 },
      { 3838648480, 3334 },
      { 3847846774, 3577 },
      { 3913593633, 4101 },
      { 3989799199, 3748 },
      { 3997038726, 6123 },
      { 4046301857, 3160 },
      { 4092654294, 3800 },
      { 4176581069, 3150 },
      { 4242327928, 4752 },
      { 4285652249, 3700 },
      { kMarkvNoneOfTheAbove, 426683 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 11808 },
      { 630592085, 1880 },
      { 679771963, 8738 },
      { 899570100, 17230 },
      { 906176560, 1360 },
      { 929101967, 5712 },
      { 1100599986, 30526 },
      { 1103903216, 3240 },
      { 1107206446, 760 },
      { 1369578001, 14610 },
      { 1372881231, 1000 },
      { 2320303498, 11309 },
      { 2926633629, 2112 },
      { 3249265647, 1392 },
      { 3334207724, 16899 },
      { 3486057732, 2272 },
      { 3674863070, 616 },
      { 3705139860, 3360 },
      { 3800912395, 15609 },
      { 3802564010, 1246 },
      { 3822983876, 7390 },
      { 4141567741, 4872 },
      { 4292991777, 2200 },
      { kMarkvNoneOfTheAbove, 2699 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 125792961, 800 },
      { 132755933, 696 },
      { 156014509, 720 },
      { 436066778, 696 },
      { 463084678, 720 },
      { 531559080, 720 },
      { 565233904, 792 },
      { 578132535, 696 },
      { 600906020, 696 },
      { 602222721, 696 },
      { 694743357, 960 },
      { 760554870, 720 },
      { 996663016, 960 },
      { 1022309772, 720 },
      { 1351676723, 720 },
      { 1496901698, 1600 },
      { 1502470404, 696 },
      { 1522901980, 696 },
      { 1548254487, 720 },
      { 1637661947, 696 },
      { 1788504755, 696 },
      { 2092468906, 696 },
      { 2094647776, 696 },
      { 2127660080, 800 },
      { 2213946343, 720 },
      { 2225172640, 696 },
      { 2259467579, 720 },
      { 2263866576, 720 },
      { 2600961503, 696 },
      { 2727022058, 1600 },
      { 2752967311, 696 },
      { 2864705739, 1600 },
      { 3021406120, 720 },
      { 3044723416, 696 },
      { 3052439312, 696 },
      { 3136865519, 696 },
      { 3297860332, 800 },
      { 3352361837, 720 },
      { 3670298840, 720 },
      { 3712946115, 1056 },
      { 3732709413, 1056 },
      { 3764662384, 800 },
      { 3788324110, 720 },
      { 3928555688, 800 },
      { 4083347580, 696 },
      { 4098876453, 720 },
      { 4147239510, 720 },
      { 4199470013, 792 },
      { 4211577142, 720 },
      { 4218799564, 720 },
      { 4290374884, 720 },
      { kMarkvNoneOfTheAbove, 128216 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 17185761, 800 },
      { 117250846, 696 },
      { 296981500, 1941 },
      { 330388453, 696 },
      { 346929928, 720 },
      { 533021259, 800 },
      { 564302770, 700 },
      { 680157484, 720 },
      { 721450866, 700 },
      { 798549062, 792 },
      { 853200279, 720 },
      { 864295921, 792 },
      { 900522183, 1056 },
      { 973908139, 696 },
      { 983243705, 696 },
      { 1033363654, 1280 },
      { 1037370721, 800 },
      { 1464587427, 600 },
      { 1670691893, 792 },
      { 1686512349, 696 },
      { 1849065716, 696 },
      { 1917602962, 1600 },
      { 1965902997, 800 },
      { 2121980967, 600 },
      { 2311072371, 1600 },
      { 2339901602, 696 },
      { 2517964682, 5432 },
      { 2542834724, 792 },
      { 2558655180, 696 },
      { 2736881867, 800 },
      { 2855506940, 10084 },
      { 2888753905, 720 },
      { 2950446516, 960 },
      { 3044188332, 3012 },
      { 3079287749, 1056 },
      { 3153451899, 696 },
      { 3214537066, 700 },
      { 3234673086, 720 },
      { 3349230696, 720 },
      { 3504158761, 1400 },
      { 3570411982, 16600 },
      { 3652695478, 696 },
      { 3764205609, 4800 },
      { 3940720663, 960 },
      { 4180570743, 696 },
      { 4221373527, 792 },
      { kMarkvNoneOfTheAbove, 93823 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 132755933, 696 },
      { 156014509, 720 },
      { 255227811, 720 },
      { 371186900, 696 },
      { 371428004, 2272 },
      { 374731234, 667 },
      { 531559080, 720 },
      { 565233904, 792 },
      { 578132535, 696 },
      { 591140762, 800 },
      { 600906020, 696 },
      { 602222721, 696 },
      { 656610661, 696 },
      { 760554870, 720 },
      { 996663016, 960 },
      { 1022309772, 720 },
      { 1496901698, 1600 },
      { 1502470404, 696 },
      { 1522901980, 696 },
      { 1536350567, 960 },
      { 1543280290, 2136 },
      { 1548254487, 720 },
      { 1788504755, 696 },
      { 2064733527, 816 },
      { 2092468906, 696 },
      { 2094647776, 696 },
      { 2162986400, 2112 },
      { 2225172640, 696 },
      { 2259467579, 720 },
      { 2263866576, 720 },
      { 2360004627, 1600 },
      { 2507709226, 800 },
      { 2600961503, 696 },
      { 2727022058, 1600 },
      { 2752967311, 696 },
      { 2864705739, 1600 },
      { 3021406120, 720 },
      { 3052439312, 696 },
      { 3136865519, 696 },
      { 3297860332, 800 },
      { 3352361837, 720 },
      { 3598957382, 800 },
      { 3619787319, 620 },
      { 3655201337, 1140 },
      { 3670298840, 720 },
      { 3774892253, 800 },
      { 3788324110, 720 },
      { 3808408202, 2544 },
      { 3951925872, 696 },
      { 3952316364, 840 },
      { 4098876453, 720 },
      { 4147239510, 720 },
      { 4199470013, 792 },
      { 4211577142, 720 },
      { 4217306348, 960 },
      { 4218799564, 720 },
      { 4290374884, 720 },
      { kMarkvNoneOfTheAbove, 116757 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3866587616, 2064 },
      { 3868239231, 18972 },
      { 3869890846, 74482 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1164218401, 12960 },
      { 2036361232, 74202 },
      { 2637132451, 6012 },
      { 3237903670, 2064 },
      { kMarkvNoneOfTheAbove, 1235 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543558236, 87162 },
      { 1069781886, 6012 },
      { 1596005536, 2064 },
      { kMarkvNoneOfTheAbove, 1235 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1949759310, 95238 },
      { kMarkvNoneOfTheAbove, 1235 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3334207724, 126527 },
      { kMarkvNoneOfTheAbove, 1265 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 236660303, 6367 },
      { 347505241, 480 },
      { 426360862, 960 },
      { 439998433, 540 },
      { 488500848, 6208 },
      { 495107308, 14623 },
      { 868652905, 700 },
      { 1191735827, 720 },
      { 1265998516, 388 },
      { 1309728002, 440 },
      { 1365842164, 388 },
      { 1396344138, 960 },
      { 1508074873, 768 },
      { 1553476262, 924 },
      { 1642818143, 525 },
      { 1851510470, 592 },
      { 1858116930, 2524 },
      { 1863199739, 600 },
      { 1979978194, 484 },
      { 1986584654, 1038 },
      { 2092100514, 468 },
      { 2098706974, 950 },
      { 2231688008, 1824 },
      { 2232491275, 528 },
      { 2329992200, 468 },
      { 2637935122, 624 },
      { 2693892518, 2112 },
      { 2759250216, 380 },
      { 2839765116, 644 },
      { 2855895374, 390 },
      { 2913136690, 388 },
      { 3012980338, 380 },
      { 3327770644, 798 },
      { 3362344229, 1008 },
      { 3398925952, 380 },
      { 3448018532, 527 },
      { 3457985288, 768 },
      { 3566035349, 2112 },
      { 3657635382, 976 },
      { 3702405475, 480 },
      { 3757479030, 976 },
      { 3797204453, 576 },
      { 4291477370, 700 },
      { kMarkvNoneOfTheAbove, 69106 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 883854656, 59847 },
      { 1962971231, 675 },
      { 2036361232, 49574 },
      { 2356768706, 6015 },
      { 2637132451, 6012 },
      { 3237903670, 2064 },
      { 3829682756, 2340 },
      { kMarkvNoneOfTheAbove, 1265 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 150685616, 480 },
      { 255302575, 388 },
      { 414620710, 600 },
      { 557400685, 624 },
      { 575205902, 480 },
      { 618761615, 924 },
      { 646282397, 2524 },
      { 686024761, 960 },
      { 740921498, 525 },
      { 921246433, 976 },
      { 1057578789, 960 },
      { 1162127370, 576 },
      { 1329499601, 976 },
      { 1352628475, 6208 },
      { 1502028603, 380 },
      { 1519723107, 380 },
      { 1543798545, 2112 },
      { 1545450160, 2112 },
      { 1570165302, 731 },
      { 1600392975, 798 },
      { 1641415225, 950 },
      { 2204920111, 527 },
      { 2257971049, 1038 },
      { 2276405827, 768 },
      { 2339018837, 700 },
      { 2340670452, 700 },
      { 2517964682, 6367 },
      { 2532518896, 1824 },
      { 2674090849, 592 },
      { 2754074729, 390 },
      { 2804281092, 768 },
      { 2816338013, 440 },
      { 2841008029, 380 },
      { 3234673086, 720 },
      { 3249261197, 388 },
      { 3619787319, 14623 },
      { 3627739127, 644 },
      { 3669223677, 468 },
      { 3787567939, 468 },
      { 3898287302, 540 },
      { 4142016703, 388 },
      { 4237092412, 1008 },
      { 4285779501, 484 },
      { kMarkvNoneOfTheAbove, 68903 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2855506940, 1 },
      { 3266548732, 1 },
      { 3732640764, 33 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3334207724, 54536 },
      { kMarkvNoneOfTheAbove, 545 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 27177503, 493 },
      { 30663912, 260 },
      { 151672195, 440 },
      { 162608772, 1644 },
      { 180913835, 240 },
      { 371621315, 232 },
      { 414444763, 247 },
      { 421602934, 440 },
      { 443347828, 280 },
      { 458937500, 264 },
      { 587888644, 284 },
      { 601656217, 320 },
      { 665789406, 320 },
      { 712168842, 228 },
      { 730943059, 640 },
      { 750870327, 168 },
      { 875212982, 1644 },
      { 899320334, 493 },
      { 973908139, 696 },
      { 989813600, 284 },
      { 1057606514, 180 },
      { 1171541710, 264 },
      { 1243764146, 232 },
      { 1310404265, 1680 },
      { 1366337101, 260 },
      { 1443547269, 640 },
      { 1472185378, 440 },
      { 1473799048, 320 },
      { 1543935193, 362 },
      { 1572834111, 960 },
      { 1623013158, 168 },
      { 1686512349, 696 },
      { 1705716306, 724 },
      { 1747355813, 1644 },
      { 1755165354, 240 },
      { 1781864804, 312 },
      { 1916983087, 440 },
      { 1941403425, 312 },
      { 2023008475, 1120 },
      { 2043684541, 264 },
      { 2274226560, 960 },
      { 2285438321, 960 },
      { 2315690100, 640 },
      { 2344328209, 440 },
      { 2414725163, 1280 },
      { 2493146691, 700 },
      { 2495155989, 168 },
      { 2558655180, 696 },
      { 2577859137, 724 },
      { 2857814560, 267 },
      { 2895151306, 1120 },
      { 2986830770, 960 },
      { 3006548167, 984 },
      { 3127329373, 1280 },
      { 3157581152, 960 },
      { 3216471040, 1600 },
      { 3296722158, 1356 },
      { 3367298820, 168 },
      { 3376009661, 1600 },
      { 3450001968, 2428 },
      { 3526837441, 168 },
      { 3609540589, 2435 },
      { 3743398113, 440 },
      { 3858973601, 960 },
      { 3953984401, 232 },
      { 3999472204, 1280 },
      { 4088613871, 320 },
      { 4184019303, 320 },
      { 4258229445, 1368 },
      { kMarkvNoneOfTheAbove, 7392 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 883854656, 28800 },
      { 1962971231, 240 },
      { 2036361232, 24628 },
      { 2366506734, 280 },
      { 3829682756, 588 },
      { kMarkvNoneOfTheAbove, 545 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 178571546, 260 },
      { 223310468, 984 },
      { 388034151, 228 },
      { 449954059, 1356 },
      { 694743357, 2880 },
      { 797415788, 440 },
      { 835638766, 440 },
      { 1002144380, 2880 },
      { 1221183390, 696 },
      { 1570165302, 4287 },
      { 1663234329, 792 },
      { 1750829822, 264 },
      { 1894133125, 416 },
      { 1967643923, 440 },
      { 1980341560, 240 },
      { 2278706468, 960 },
      { 2326990117, 260 },
      { 2464905186, 1368 },
      { 2511346984, 3840 },
      { 2517964682, 12018 },
      { 2616085763, 392 },
      { 2710583246, 840 },
      { 2745872368, 700 },
      { 2924263085, 267 },
      { 3027500544, 440 },
      { 3044723416, 2088 },
      { 3202324433, 284 },
      { 3289213933, 284 },
      { 3323682385, 4932 },
      { 3366848728, 487 },
      { 3417583519, 180 },
      { 3732916270, 624 },
      { 3787909072, 1920 },
      { 3877813395, 280 },
      { 4028028350, 280 },
      { 4178218543, 208 },
      { kMarkvNoneOfTheAbove, 5826 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2855506940, 54155 },
      { kMarkvNoneOfTheAbove, 926 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3533637837, 280 },
      { 3535289452, 1444 },
      { 3536941067, 1444 },
      { 3538592682, 4828 },
      { 3540244297, 9177 },
      { 3541895912, 9177 },
      { kMarkvNoneOfTheAbove, 263 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 122074 },
      { 1951208733, 94662 },
      { 2320303498, 101281 },
      { 3334207724, 24220 },
      { kMarkvNoneOfTheAbove, 3422 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 328661377, 1453 },
      { 464259778, 1222 },
      { 920941800, 1596 },
      { 969500141, 1368 },
      { 1449907751, 1280 },
      { 1451831482, 1096 },
      { 1543798545, 2112 },
      { 1545450160, 2112 },
      { 1626224034, 1248 },
      { 1669930486, 1536 },
      { 1770165905, 1280 },
      { 2278571792, 1938 },
      { 2432827426, 1228 },
      { 2656211099, 1536 },
      { 2736844435, 1248 },
      { 2870852215, 1596 },
      { 2919626325, 2286 },
      { 2923708820, 1570 },
      { 3325419312, 2490 },
      { 3678875745, 1416 },
      { 4182141402, 1644 },
      { 4241374559, 1368 },
      { kMarkvNoneOfTheAbove, 311036 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 135920445, 1096 },
      { 176166202, 1228 },
      { 294390719, 1536 },
      { 296981500, 5847 },
      { 743407979, 2112 },
      { 810488476, 2837 },
      { 837715723, 1252 },
      { 885020215, 1570 },
      { 922996215, 1248 },
      { 959681532, 6597 },
      { 963902061, 1644 },
      { 1136775085, 1050 },
      { 1189681639, 2800 },
      { 1203545131, 2762 },
      { 1297294717, 1056 },
      { 1317058015, 1056 },
      { 1352397672, 1416 },
      { 1367301635, 1455 },
      { 1412908157, 1400 },
      { 1570165302, 6797 },
      { 1763758554, 1222 },
      { 1791427568, 1368 },
      { 1992893964, 2499 },
      { 2013867381, 1280 },
      { 2096388952, 14194 },
      { 2219733501, 1521 },
      { 2383939514, 2112 },
      { 2517964682, 2244 },
      { 2555315060, 1368 },
      { 2572638469, 1536 },
      { 2762094724, 1938 },
      { 2770161927, 1440 },
      { 2855506940, 1060 },
      { 3044188332, 1204 },
      { 3187066832, 1692 },
      { 3319278167, 1440 },
      { 3653838348, 4273 },
      { 3675926744, 1056 },
      { 3701632935, 1289 },
      { 3712946115, 1056 },
      { 3732709413, 1056 },
      { 3743748793, 1924 },
      { 3783543823, 3316 },
      { 3930727258, 1280 },
      { kMarkvNoneOfTheAbove, 245532 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 126463145, 1056 },
      { 220008971, 1344 },
      { 246375791, 1152 },
      { 503145996, 2894 },
      { 628331516, 1944 },
      { 643418617, 4224 },
      { 743407979, 2112 },
      { 837715723, 2050 },
      { 858902117, 1152 },
      { 870594305, 1439 },
      { 939671928, 1056 },
      { 959681532, 10798 },
      { 1051471757, 1344 },
      { 1092948665, 3890 },
      { 1097775533, 2112 },
      { 1136775085, 1750 },
      { 1140367371, 1056 },
      { 1332643570, 1632 },
      { 1367301635, 1461 },
      { 1558001705, 1228 },
      { 1684282922, 1031 },
      { 2096388952, 3109 },
      { 2183547611, 2112 },
      { 2219733501, 1920 },
      { 2358141757, 2112 },
      { 2359973133, 1958 },
      { 2383939514, 2112 },
      { 2444465148, 2912 },
      { 2517964682, 4861 },
      { 2567901801, 2832 },
      { 2598189097, 1228 },
      { 2655147757, 1368 },
      { 2683080096, 4085 },
      { 2705434194, 1280 },
      { 2738307068, 1056 },
      { 2780898906, 1056 },
      { 3030911670, 1248 },
      { 3032677281, 1596 },
      { 3063300848, 2493 },
      { 3277199633, 1344 },
      { 3289969989, 1453 },
      { 3401762422, 2490 },
      { 3436143898, 1801 },
      { 3560552546, 1596 },
      { 3656163446, 1056 },
      { 3675926744, 1056 },
      { 3701632935, 1156 },
      { 3743748793, 1925 },
      { 3752211294, 1056 },
      { 3794803132, 1056 },
      { 4241374559, 1368 },
      { kMarkvNoneOfTheAbove, 242239 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 47621 },
      { 1951208733, 82301 },
      { 2320303498, 11961 },
      { 3334207724, 19136 },
      { kMarkvNoneOfTheAbove, 1610 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50385656, 792 },
      { 117250846, 696 },
      { 171494987, 800 },
      { 195244192, 600 },
      { 210754155, 580 },
      { 265392489, 984 },
      { 333855951, 782 },
      { 416853049, 1440 },
      { 529068443, 1416 },
      { 533021259, 800 },
      { 615982737, 2376 },
      { 660038281, 800 },
      { 663341511, 800 },
      { 669812542, 700 },
      { 716890919, 960 },
      { 1081536219, 576 },
      { 1119744229, 984 },
      { 1123617794, 528 },
      { 1139547465, 3792 },
      { 1162789888, 1600 },
      { 1178317551, 2112 },
      { 1190147516, 800 },
      { 1193734351, 800 },
      { 1215030156, 576 },
      { 1220749418, 1440 },
      { 1318479490, 1536 },
      { 1461398554, 932 },
      { 1486207619, 800 },
      { 1551372768, 1600 },
      { 1763758554, 1222 },
      { 1797960910, 960 },
      { 1850331254, 800 },
      { 1894417995, 720 },
      { 1964254745, 1368 },
      { 1965902997, 800 },
      { 1989327599, 800 },
      { 2095027856, 720 },
      { 2123683379, 720 },
      { 2124837447, 1596 },
      { 2137526937, 640 },
      { 2269114589, 931 },
      { 2269130237, 768 },
      { 2330636993, 2924 },
      { 2481746922, 528 },
      { 2503770904, 672 },
      { 2589449658, 2136 },
      { 2603020391, 768 },
      { 2604576561, 1222 },
      { 2795773560, 800 },
      { 2835131395, 1368 },
      { 2852854788, 1248 },
      { 2890638791, 1597 },
      { 2895413148, 672 },
      { 2950446516, 960 },
      { 2963744582, 864 },
      { 3079287749, 1056 },
      { 3088785099, 528 },
      { 3280064277, 720 },
      { 3335250889, 600 },
      { 3510242586, 1222 },
      { 3517169445, 700 },
      { 3518703473, 700 },
      { 3536471583, 769 },
      { 3579593979, 1644 },
      { 3591222197, 1280 },
      { 3673811979, 768 },
      { 3727034815, 768 },
      { 3730093054, 2112 },
      { 3898287302, 540 },
      { 3944781937, 2832 },
      { 3950980241, 1222 },
      { 4033586023, 800 },
      { 4041974454, 800 },
      { 4052965752, 800 },
      { 4083161638, 700 },
      { 4167600590, 644 },
      { 4185661467, 928 },
      { 4237092412, 1008 },
      { 4244540017, 597 },
      { kMarkvNoneOfTheAbove, 78485 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50998433, 1280 },
      { 171494987, 800 },
      { 249378857, 2450 },
      { 296981500, 41800 },
      { 508007510, 1600 },
      { 610429940, 696 },
      { 660038281, 800 },
      { 663341511, 800 },
      { 836581417, 800 },
      { 1027242654, 1228 },
      { 1167160774, 724 },
      { 1191015885, 701 },
      { 1200870684, 800 },
      { 1203545131, 5388 },
      { 1265796414, 2832 },
      { 1319785741, 2403 },
      { 1669959736, 720 },
      { 1684282922, 1217 },
      { 1752686878, 576 },
      { 1850331254, 1600 },
      { 1901166356, 705 },
      { 1906988301, 933 },
      { 2055836767, 1161 },
      { 2095027856, 580 },
      { 2096388952, 1170 },
      { 2144962711, 960 },
      { 2217833278, 1222 },
      { 2500819054, 800 },
      { 2525173102, 1008 },
      { 2575525651, 868 },
      { 2660843182, 545 },
      { 2855506940, 8667 },
      { 2918750759, 700 },
      { 2919787747, 1080 },
      { 3091876332, 2930 },
      { 3187066832, 4170 },
      { 3244209297, 1063 },
      { 3423702268, 800 },
      { 3508792859, 2120 },
      { 3548535223, 1080 },
      { 3619787319, 2919 },
      { 3653838348, 9160 },
      { 3692647551, 773 },
      { 3713290482, 800 },
      { 3753486980, 595 },
      { 3783756895, 600 },
      { 3797961332, 1597 },
      { 3836822275, 640 },
      { 4043078107, 540 },
      { 4052965752, 800 },
      { 4091394002, 960 },
      { kMarkvNoneOfTheAbove, 41468 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 49456560, 960 },
      { 170690025, 1598 },
      { 243178923, 984 },
      { 295017943, 760 },
      { 296981500, 5608 },
      { 330249537, 784 },
      { 435256475, 1222 },
      { 443558693, 2376 },
      { 456043370, 800 },
      { 470277359, 768 },
      { 592180731, 1222 },
      { 663258455, 1222 },
      { 706238670, 672 },
      { 810488476, 592 },
      { 870594305, 775 },
      { 877895868, 792 },
      { 900522183, 1056 },
      { 1077859090, 1280 },
      { 1082941229, 800 },
      { 1104362365, 1416 },
      { 1132589448, 1248 },
      { 1173092699, 1368 },
      { 1203545131, 2617 },
      { 1265796414, 8288 },
      { 1278818058, 672 },
      { 1285705317, 1368 },
      { 1319785741, 2629 },
      { 1382106590, 1008 },
      { 1461897718, 700 },
      { 1474506522, 800 },
      { 1530183840, 644 },
      { 1558001705, 2400 },
      { 1558990974, 576 },
      { 1616846013, 576 },
      { 1633850097, 782 },
      { 1684282922, 594 },
      { 1725011064, 700 },
      { 1767704813, 768 },
      { 1923453688, 696 },
      { 1941148668, 1430 },
      { 1955104493, 1644 },
      { 2022961611, 960 },
      { 2162274327, 768 },
      { 2212501241, 800 },
      { 2219733501, 1066 },
      { 2234361374, 2136 },
      { 2272221101, 528 },
      { 2305269460, 640 },
      { 2488410748, 800 },
      { 2566666743, 1222 },
      { 2598189097, 2400 },
      { 2775815164, 1947 },
      { 2793529873, 800 },
      { 2844616706, 720 },
      { 2970183398, 1025 },
      { 3103302036, 1536 },
      { 3110479131, 800 },
      { 3115038057, 800 },
      { 3116932970, 864 },
      { 3152745753, 800 },
      { 3187066832, 641 },
      { 3244209297, 4163 },
      { 3383007207, 984 },
      { 3392887901, 1056 },
      { 3508792859, 720 },
      { 3737376990, 800 },
      { 3753486980, 5112 },
      { 3765247327, 931 },
      { 3817149113, 984 },
      { 3839047923, 960 },
      { 3886529747, 2112 },
      { 4044928561, 800 },
      { 4061558677, 720 },
      { 4069720347, 540 },
      { 4069810315, 2832 },
      { 4128942283, 800 },
      { 4164704452, 2112 },
      { 4273793488, 2112 },
      { kMarkvNoneOfTheAbove, 56443 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 156333 },
      { 1951208733, 175477 },
      { 2320303498, 44808 },
      { 3334207724, 36168 },
      { kMarkvNoneOfTheAbove, 4127 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 342197850, 1358 },
      { 885020215, 1570 },
      { 963902061, 1644 },
      { 1041368449, 1416 },
      { 1352397672, 1416 },
      { 1791427568, 1368 },
      { 2013867381, 1280 },
      { 2513230733, 2634 },
      { 2555315060, 1368 },
      { 2562485583, 1248 },
      { 2567901801, 2832 },
      { 2655147757, 1368 },
      { 2680283743, 1596 },
      { 2752766693, 1248 },
      { 2806716850, 1280 },
      { 3030911670, 1248 },
      { 3401762422, 2490 },
      { 3697738938, 1537 },
      { 4164704452, 2112 },
      { 4273793488, 2112 },
      { kMarkvNoneOfTheAbove, 383788 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 126463145, 2568 },
      { 129135650, 4032 },
      { 200922300, 1920 },
      { 328661377, 1453 },
      { 354479447, 2920 },
      { 360730278, 1248 },
      { 451264926, 1284 },
      { 529068443, 1440 },
      { 593829839, 1488 },
      { 742917749, 1368 },
      { 761731755, 1920 },
      { 810488476, 7688 },
      { 870594305, 1576 },
      { 894529125, 2112 },
      { 959681532, 3003 },
      { 1054461787, 4359 },
      { 1077859090, 1280 },
      { 1086964761, 1440 },
      { 1158929937, 4359 },
      { 1168927492, 1920 },
      { 1196280518, 1570 },
      { 1203545131, 5232 },
      { 1367301635, 1456 },
      { 1508550646, 1248 },
      { 1618544981, 1456 },
      { 1661163736, 1240 },
      { 1684282922, 2688 },
      { 1766994680, 2568 },
      { 1830851200, 1920 },
      { 1901166356, 3139 },
      { 1955104493, 1644 },
      { 2055836767, 1372 },
      { 2096388952, 8913 },
      { 2100052708, 1368 },
      { 2161102232, 2491 },
      { 2197904616, 1392 },
      { 2262137600, 2112 },
      { 2278571792, 1938 },
      { 2281956980, 1716 },
      { 2438466459, 1320 },
      { 2443959748, 1499 },
      { 2517964682, 3409 },
      { 2557754096, 1416 },
      { 2622612602, 1389 },
      { 2660843182, 1523 },
      { 2736844435, 1248 },
      { 2780898906, 1712 },
      { 3044188332, 2539 },
      { 3059119137, 1344 },
      { 3194725903, 1248 },
      { 3270430997, 2919 },
      { 3337532056, 2112 },
      { 3407526215, 1712 },
      { 3496407048, 1460 },
      { 3504158761, 3223 },
      { 3534518722, 1344 },
      { 3570411982, 1309 },
      { 3701632935, 1934 },
      { 3929248764, 4458 },
      { 3944781937, 2832 },
      { 3970432934, 1368 },
      { 4008405264, 4992 },
      { 4245257809, 2666 },
      { 4253051659, 1248 },
      { kMarkvNoneOfTheAbove, 269848 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 13319433, 1815 },
      { 15502752, 4084 },
      { 162608772, 1479 },
      { 171307615, 5440 },
      { 296981500, 3512 },
      { 354479447, 2136 },
      { 413918748, 1358 },
      { 443490822, 1248 },
      { 487719832, 1400 },
      { 593829839, 7056 },
      { 615982737, 1428 },
      { 703543228, 1280 },
      { 810488476, 2995 },
      { 870594305, 1250 },
      { 875212982, 1479 },
      { 959681532, 5471 },
      { 1019457583, 3542 },
      { 1203545131, 2352 },
      { 1278448636, 1445 },
      { 1325348861, 1944 },
      { 1368383673, 2112 },
      { 1400019344, 1463 },
      { 1646147798, 1944 },
      { 1679946323, 1944 },
      { 1684282922, 7779 },
      { 1747355813, 1479 },
      { 1755648697, 2917 },
      { 1793544760, 2400 },
      { 1811839150, 5440 },
      { 1901166356, 1390 },
      { 1947620272, 1248 },
      { 1992893964, 2526 },
      { 2042001863, 2832 },
      { 2096388952, 1711 },
      { 2123388694, 1944 },
      { 2128251367, 1400 },
      { 2130747644, 2764 },
      { 2135340676, 2904 },
      { 2161102232, 2635 },
      { 2443959748, 1447 },
      { 2513230733, 2634 },
      { 2557754096, 1416 },
      { 2580096524, 2902 },
      { 2589449658, 2112 },
      { 2614879967, 3111 },
      { 2698156268, 1944 },
      { 2970183398, 2975 },
      { 3002890475, 1804 },
      { 3133016299, 1944 },
      { 3142155593, 2120 },
      { 3187066832, 1669 },
      { 3266548732, 3910 },
      { 3287039847, 2666 },
      { 3357301402, 2634 },
      { 3413713311, 1682 },
      { 3434076295, 1248 },
      { 3496407048, 1528 },
      { 3504158761, 1652 },
      { 3882634684, 1944 },
      { 3929248764, 6232 },
      { 3987079331, 1632 },
      { 4076840151, 2112 },
      { 4243119782, 1944 },
      { kMarkvNoneOfTheAbove, 260105 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 9073 },
      { 1951208733, 21138 },
      { 2320303498, 8436 },
      { 3334207724, 2988 },
      { kMarkvNoneOfTheAbove, 416 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 10142671, 162 },
      { 27865391, 880 },
      { 29517006, 880 },
      { 41739659, 360 },
      { 97231530, 144 },
      { 171334650, 223 },
      { 200553094, 240 },
      { 257136089, 1222 },
      { 294390719, 768 },
      { 375530199, 360 },
      { 380957745, 440 },
      { 388034151, 228 },
      { 455591063, 180 },
      { 462664429, 528 },
      { 491456522, 440 },
      { 502863753, 192 },
      { 626480004, 284 },
      { 643418617, 2112 },
      { 651464351, 235 },
      { 701281393, 240 },
      { 744817486, 264 },
      { 783918780, 768 },
      { 862784766, 223 },
      { 930804377, 168 },
      { 952536201, 300 },
      { 955476870, 168 },
      { 1043738701, 168 },
      { 1047011733, 164 },
      { 1080545747, 200 },
      { 1137442027, 360 },
      { 1235468610, 168 },
      { 1412908157, 700 },
      { 1431749301, 800 },
      { 1434223270, 300 },
      { 1440646342, 236 },
      { 1508570930, 174 },
      { 1510422521, 350 },
      { 1548121999, 700 },
      { 1582841441, 440 },
      { 1612225949, 360 },
      { 1665981878, 144 },
      { 1680746207, 192 },
      { 1696076631, 696 },
      { 1702168830, 440 },
      { 1761469971, 214 },
      { 1799299383, 243 },
      { 1910240213, 284 },
      { 1917451875, 180 },
      { 1945006185, 174 },
      { 1998444837, 400 },
      { 2045285083, 564 },
      { 2217966239, 140 },
      { 2279273489, 136 },
      { 2289803479, 384 },
      { 2348676810, 168 },
      { 2353194283, 180 },
      { 2403632109, 360 },
      { 2409539315, 140 },
      { 2414984922, 440 },
      { 2477389837, 140 },
      { 2524531022, 168 },
      { 2573160348, 564 },
      { 2639720559, 960 },
      { 2773229577, 180 },
      { 2796513469, 140 },
      { 2881225774, 440 },
      { 2890570341, 150 },
      { 2952850186, 960 },
      { 3023287679, 471 },
      { 3118548424, 284 },
      { 3877813395, 280 },
      { 3931288033, 232 },
      { 3972309363, 350 },
      { 4117704995, 136 },
      { 4140081844, 2112 },
      { 4258414038, 192 },
      { kMarkvNoneOfTheAbove, 11684 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 5908395, 440 },
      { 139011596, 264 },
      { 296981500, 1095 },
      { 342615870, 350 },
      { 370232173, 223 },
      { 492958971, 289 },
      { 528662843, 200 },
      { 551924251, 264 },
      { 604894932, 150 },
      { 610429940, 696 },
      { 780957373, 360 },
      { 810488476, 526 },
      { 872544165, 168 },
      { 878733439, 440 },
      { 918849409, 140 },
      { 959681532, 2910 },
      { 1013756921, 284 },
      { 1038982109, 140 },
      { 1081611718, 180 },
      { 1125913837, 140 },
      { 1209418480, 583 },
      { 1318081294, 168 },
      { 1367301635, 461 },
      { 1417425499, 180 },
      { 1625742020, 2286 },
      { 1684282922, 524 },
      { 1746004874, 168 },
      { 1758287856, 1164 },
      { 1777640493, 140 },
      { 2066323109, 360 },
      { 2094550054, 400 },
      { 2096388952, 621 },
      { 2144962711, 960 },
      { 2434845539, 180 },
      { 2480811229, 700 },
      { 2552825357, 360 },
      { 2636946065, 223 },
      { 2651956495, 228 },
      { 2669086217, 144 },
      { 2680819379, 350 },
      { 2709694527, 768 },
      { 2715304020, 360 },
      { 2790648021, 144 },
      { 2802261839, 440 },
      { 2806296851, 236 },
      { 2864543087, 168 },
      { 2952260510, 232 },
      { 2963184673, 168 },
      { 3091876332, 1184 },
      { 3098991995, 440 },
      { 3131890669, 168 },
      { 3138977758, 136 },
      { 3198541202, 768 },
      { 3260579369, 300 },
      { 3263841912, 300 },
      { 3335250889, 240 },
      { 3345856521, 174 },
      { 3381478137, 360 },
      { 3489269251, 180 },
      { 3510242586, 1222 },
      { 3820814597, 284 },
      { 3900859293, 700 },
      { 4041974454, 800 },
      { 4244540017, 192 },
      { 4265894873, 2112 },
      { kMarkvNoneOfTheAbove, 10516 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 116093251, 240 },
      { 149720480, 800 },
      { 183103444, 243 },
      { 251209228, 1944 },
      { 296981500, 2409 },
      { 357505993, 223 },
      { 394654115, 440 },
      { 410274915, 280 },
      { 452208841, 1128 },
      { 788046331, 350 },
      { 797934924, 1760 },
      { 810488476, 988 },
      { 1144188012, 150 },
      { 1220127364, 800 },
      { 1321616112, 1944 },
      { 1324351672, 1128 },
      { 1348149915, 180 },
      { 1459457331, 192 },
      { 1465623797, 192 },
      { 1531216990, 214 },
      { 1543672828, 950 },
      { 1578775276, 4224 },
      { 1738815671, 180 },
      { 1904128160, 136 },
      { 2071351379, 768 },
      { 2119793999, 384 },
      { 2274779301, 240 },
      { 2291766425, 2112 },
      { 2357410109, 284 },
      { 2438466459, 1760 },
      { 2496463830, 350 },
      { 2630220147, 440 },
      { 2682510803, 200 },
      { 3047649911, 782 },
      { 3085703811, 180 },
      { 3235459678, 180 },
      { 3261703164, 125 },
      { 3331487616, 136 },
      { 3462674048, 164 },
      { 3570219049, 2428 },
      { 3585315836, 473 },
      { 3602108619, 2112 },
      { 3724004880, 360 },
      { 3931641900, 1550 },
      { 3955205564, 235 },
      { 4073492988, 192 },
      { 4127308103, 223 },
      { kMarkvNoneOfTheAbove, 5278 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 107274 },
      { 2320303498, 29600 },
      { 3334207724, 20217 },
      { kMarkvNoneOfTheAbove, 1570 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 14113753, 700 },
      { 102358168, 624 },
      { 179458548, 624 },
      { 330388453, 696 },
      { 386525753, 758 },
      { 470277359, 833 },
      { 497658126, 768 },
      { 508007510, 800 },
      { 815034111, 1367 },
      { 826214242, 624 },
      { 849867303, 768 },
      { 885645401, 624 },
      { 939415664, 504 },
      { 968885186, 700 },
      { 1105835505, 504 },
      { 1159301677, 514 },
      { 1461897718, 700 },
      { 1482251215, 624 },
      { 1486206763, 504 },
      { 1527762373, 960 },
      { 1558990974, 576 },
      { 1618754372, 528 },
      { 1669959736, 720 },
      { 1752686878, 576 },
      { 2004567202, 644 },
      { 2055637638, 960 },
      { 2113506324, 700 },
      { 2154320787, 480 },
      { 2162274327, 768 },
      { 2306141594, 748 },
      { 2345566651, 504 },
      { 2457690657, 504 },
      { 2473053808, 530 },
      { 2500422644, 960 },
      { 2504802016, 1596 },
      { 2506771164, 700 },
      { 2793529873, 800 },
      { 2801333547, 768 },
      { 2879050471, 624 },
      { 3032677281, 1596 },
      { 3045470312, 624 },
      { 3181546731, 504 },
      { 3240977890, 504 },
      { 3262572726, 491 },
      { 3307100165, 491 },
      { 3425841570, 624 },
      { 3560552546, 1596 },
      { 3641833815, 830 },
      { 3652695478, 696 },
      { 3782362128, 700 },
      { 3797961332, 1605 },
      { 3837583704, 504 },
      { 3886529747, 2112 },
      { 3907920335, 700 },
      { 4043078107, 540 },
      { 4044928561, 800 },
      { 4069720347, 540 },
      { 4180570743, 696 },
      { 4245743275, 600 },
      { 4285201458, 624 },
      { kMarkvNoneOfTheAbove, 113402 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 100979271, 520 },
      { 269576093, 720 },
      { 314809953, 491 },
      { 354479447, 2136 },
      { 497658126, 768 },
      { 882718761, 800 },
      { 968885186, 700 },
      { 973908139, 696 },
      { 1019457583, 782 },
      { 1191015885, 701 },
      { 1266262705, 700 },
      { 1310404265, 480 },
      { 1325348861, 1944 },
      { 1367301635, 2214 },
      { 1368383673, 2112 },
      { 1570165302, 1008 },
      { 1618544981, 768 },
      { 1646147798, 1944 },
      { 1674464100, 820 },
      { 1679946323, 1944 },
      { 1686512349, 696 },
      { 1766401548, 2112 },
      { 1774052499, 700 },
      { 1788301425, 540 },
      { 2023008475, 480 },
      { 2055836767, 1296 },
      { 2096388952, 8757 },
      { 2123388694, 1944 },
      { 2129301998, 480 },
      { 2212501241, 800 },
      { 2274226560, 960 },
      { 2362972044, 2112 },
      { 2378763734, 575 },
      { 2506771164, 700 },
      { 2558655180, 696 },
      { 2622612602, 1133 },
      { 2660843182, 1751 },
      { 2698156268, 1944 },
      { 2801333547, 768 },
      { 2850246066, 720 },
      { 2895151306, 480 },
      { 2970183398, 2605 },
      { 2986830770, 960 },
      { 3001444829, 480 },
      { 3133016299, 1944 },
      { 3152745753, 800 },
      { 3187066832, 8690 },
      { 3261122899, 540 },
      { 3496407048, 725 },
      { 3513669836, 640 },
      { 3536390697, 1064 },
      { 3570411982, 895 },
      { 3653838348, 840 },
      { 3713290482, 829 },
      { 3858973601, 960 },
      { 3873587660, 480 },
      { 3877583949, 700 },
      { 3882634684, 1944 },
      { 3907920335, 700 },
      { 3997432565, 540 },
      { 4169226615, 768 },
      { 4219766939, 700 },
      { 4243119782, 1944 },
      { kMarkvNoneOfTheAbove, 77021 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 11698369, 700 },
      { 146392076, 700 },
      { 151810803, 576 },
      { 223800276, 624 },
      { 227103506, 624 },
      { 253329281, 936 },
      { 346929928, 720 },
      { 461040879, 936 },
      { 629859130, 540 },
      { 680157484, 720 },
      { 783918780, 768 },
      { 810488476, 3010 },
      { 824323032, 491 },
      { 870594305, 1011 },
      { 959681532, 3067 },
      { 975807626, 700 },
      { 1081642571, 1344 },
      { 1084574846, 1464 },
      { 1094817798, 720 },
      { 1141965917, 744 },
      { 1164137269, 792 },
      { 1166917451, 768 },
      { 1204787336, 768 },
      { 1232501371, 4032 },
      { 1318479490, 768 },
      { 1369818198, 960 },
      { 1372785527, 1729 },
      { 1526654696, 2136 },
      { 1543672828, 1200 },
      { 1548121999, 700 },
      { 1635292159, 592 },
      { 1641070431, 672 },
      { 1684282922, 3198 },
      { 1767704813, 744 },
      { 1781765116, 884 },
      { 1838763297, 504 },
      { 1901166356, 1680 },
      { 1904846533, 700 },
      { 2011183308, 768 },
      { 2032069771, 768 },
      { 2071351379, 576 },
      { 2087004702, 1489 },
      { 2244928358, 1729 },
      { 2314864456, 2880 },
      { 2374216296, 758 },
      { 2394332122, 744 },
      { 2443610186, 1040 },
      { 2524697596, 2112 },
      { 2526961521, 528 },
      { 2568098594, 4992 },
      { 2807907995, 540 },
      { 3103302036, 960 },
      { 3117071189, 1732 },
      { 3188115516, 1920 },
      { 3417584874, 672 },
      { 3554463148, 2112 },
      { 3561482820, 768 },
      { 3691770462, 2088 },
      { 3729929345, 960 },
      { 3733675151, 504 },
      { 3831290364, 939 },
      { 3866493821, 540 },
      { 3929248764, 15114 },
      { 4060703604, 1920 },
      { 4092487128, 1920 },
      { 4167600590, 644 },
      { 4214779116, 576 },
      { 4248015868, 1042 },
      { kMarkvNoneOfTheAbove, 62104 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 151035 },
      { kMarkvNoneOfTheAbove, 1510 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 78001013, 720 },
      { 170690025, 1599 },
      { 206688607, 768 },
      { 443490822, 1248 },
      { 461476226, 739 },
      { 537830163, 960 },
      { 669982125, 2376 },
      { 790502615, 1019 },
      { 805072272, 528 },
      { 1173092699, 1368 },
      { 1220643281, 1248 },
      { 1448448666, 864 },
      { 1466804584, 480 },
      { 1473411044, 480 },
      { 1515695460, 700 },
      { 1587730355, 864 },
      { 1625742020, 2286 },
      { 2071351379, 2690 },
      { 2250055803, 1240 },
      { 2291766425, 2112 },
      { 2416108131, 921 },
      { 2427834344, 864 },
      { 2436009347, 700 },
      { 2455417440, 1416 },
      { 2480811229, 700 },
      { 2654325647, 1464 },
      { 2919796598, 1356 },
      { 3047649911, 782 },
      { 3088511797, 960 },
      { 3104643263, 2215 },
      { 3198541202, 768 },
      { 3204986803, 864 },
      { 3272233597, 700 },
      { 3383007207, 984 },
      { 3602108619, 2112 },
      { 3622349409, 800 },
      { 3714664910, 1019 },
      { 3717942504, 468 },
      { 3732000233, 480 },
      { 3759072440, 1248 },
      { 3765247327, 941 },
      { 3805423332, 700 },
      { 3829325073, 1248 },
      { 3866493821, 540 },
      { 4058280485, 1008 },
      { 4061558677, 720 },
      { 4148979936, 640 },
      { 4155586396, 640 },
      { kMarkvNoneOfTheAbove, 100998 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50385656, 1574 },
      { 181902171, 1356 },
      { 560078433, 702 },
      { 615982737, 2376 },
      { 674428451, 760 },
      { 837715723, 1599 },
      { 886972033, 814 },
      { 900101778, 760 },
      { 983299427, 1063 },
      { 1237148906, 2784 },
      { 1364157225, 1740 },
      { 1367301635, 688 },
      { 1380160211, 1056 },
      { 1451831482, 984 },
      { 1499923635, 1638 },
      { 1570165302, 959 },
      { 1735295265, 526 },
      { 1766401548, 5006 },
      { 1796311149, 1047 },
      { 1826456251, 480 },
      { 1839669171, 640 },
      { 2012838864, 6120 },
      { 2024071551, 1040 },
      { 2096388952, 3896 },
      { 2161102232, 1550 },
      { 2197874825, 6432 },
      { 2279700640, 3200 },
      { 2289183712, 468 },
      { 2351620600, 702 },
      { 2362972044, 4248 },
      { 2472176885, 660 },
      { 2477434291, 834 },
      { 2530899578, 480 },
      { 2531826164, 702 },
      { 2558133383, 702 },
      { 2589449658, 4248 },
      { 2621255555, 852 },
      { 2622612602, 879 },
      { 2872580757, 1266 },
      { 2881302403, 3451 },
      { 2891091137, 720 },
      { 2923708820, 1368 },
      { 2936040203, 3552 },
      { 2970183398, 4006 },
      { 3187066832, 4804 },
      { 3224952074, 488 },
      { 3244383472, 521 },
      { 3261122899, 720 },
      { 3362830643, 1560 },
      { 3538158875, 2400 },
      { 3635542517, 4272 },
      { 3682213068, 468 },
      { 3721902098, 660 },
      { 3826846522, 496 },
      { 3877583949, 760 },
      { 3997432565, 2480 },
      { 4093615095, 480 },
      { 4106828015, 640 },
      { kMarkvNoneOfTheAbove, 51868 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 27177503, 1471 },
      { 50385656, 784 },
      { 129748122, 518 },
      { 139011596, 1897 },
      { 162608772, 492 },
      { 181902171, 800 },
      { 225200779, 960 },
      { 342159236, 1600 },
      { 386293029, 468 },
      { 429023543, 1440 },
      { 443558693, 1416 },
      { 504514034, 832 },
      { 615982737, 1488 },
      { 669812542, 700 },
      { 674428451, 640 },
      { 837715723, 1599 },
      { 861753115, 480 },
      { 875212982, 492 },
      { 876867882, 1600 },
      { 899320334, 1471 },
      { 900101778, 640 },
      { 938517572, 700 },
      { 1347339159, 464 },
      { 1356063462, 1600 },
      { 1373856501, 960 },
      { 1376656865, 831 },
      { 1451831482, 984 },
      { 1522979646, 464 },
      { 1548491889, 528 },
      { 1570165302, 941 },
      { 1735295265, 494 },
      { 1747355813, 492 },
      { 1766401548, 5024 },
      { 1871105284, 518 },
      { 1918742169, 1416 },
      { 1922045399, 1416 },
      { 1978689945, 862 },
      { 2024071551, 520 },
      { 2059975069, 1056 },
      { 2076833303, 960 },
      { 2096388952, 3621 },
      { 2181030375, 480 },
      { 2197874825, 3624 },
      { 2362972044, 2136 },
      { 2414725163, 960 },
      { 2517964682, 17246 },
      { 2564745684, 795 },
      { 2577387676, 768 },
      { 2589449658, 2136 },
      { 2604242419, 528 },
      { 2683080096, 3200 },
      { 2696349144, 960 },
      { 2763960513, 1463 },
      { 2817823941, 672 },
      { 2852854788, 1248 },
      { 2891091137, 720 },
      { 2919626325, 2286 },
      { 2923708820, 1368 },
      { 2936040203, 2328 },
      { 2963744582, 864 },
      { 2970183398, 3307 },
      { 2984459037, 528 },
      { 2996594997, 1600 },
      { 3015046341, 1463 },
      { 3055195668, 832 },
      { 3127329373, 960 },
      { 3187066832, 2702 },
      { 3193597927, 518 },
      { 3200890815, 960 },
      { 3224258475, 480 },
      { 3224480461, 960 },
      { 3261122899, 540 },
      { 3609540589, 1471 },
      { 3619404941, 480 },
      { 3619626927, 960 },
      { 3727034815, 768 },
      { 3742724777, 480 },
      { 3742946763, 960 },
      { 3836179806, 811 },
      { 3913885196, 5826 },
      { 3927338499, 832 },
      { 3927466635, 960 },
      { 3997432565, 1940 },
      { 3999472204, 960 },
      { 4010499223, 1600 },
      { 4032662899, 480 },
      { 4110915453, 461 },
      { 4145966869, 720 },
      { 4228303141, 1392 },
      { kMarkvNoneOfTheAbove, 27173 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1036475267, 129408 },
      { kMarkvNoneOfTheAbove, 1294 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLabel, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1036475267, 2130 },
      { kMarkvNoneOfTheAbove, 21 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpBranch, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 57149555, 528 },
      { 139011596, 160 },
      { 255835594, 414 },
      { 330249537, 720 },
      { 388686774, 768 },
      { 508217552, 534 },
      { 550831114, 180 },
      { 559246409, 320 },
      { 599185303, 1596 },
      { 649208064, 414 },
      { 679061455, 397 },
      { 810488476, 1601 },
      { 951841533, 277 },
      { 1008886329, 800 },
      { 1022544883, 291 },
      { 1215030156, 576 },
      { 1305703280, 134 },
      { 1367301635, 162 },
      { 1453447304, 528 },
      { 1487177499, 136 },
      { 1603937321, 1596 },
      { 1617826947, 336 },
      { 1643868273, 240 },
      { 1672607981, 274 },
      { 1681941034, 140 },
      { 1755165354, 240 },
      { 1781864804, 312 },
      { 1795715718, 180 },
      { 1977038330, 149 },
      { 2096388952, 5721 },
      { 2204920111, 567 },
      { 2244470522, 1596 },
      { 2330636993, 2396 },
      { 2400601988, 140 },
      { 2424848261, 336 },
      { 2603020391, 768 },
      { 2622612602, 767 },
      { 2645135839, 336 },
      { 2660843182, 1057 },
      { 2708915136, 1149 },
      { 2724166585, 1596 },
      { 2728667725, 528 },
      { 2890638791, 1596 },
      { 2901034693, 133 },
      { 2941648648, 280 },
      { 2970183398, 455 },
      { 2998120306, 223 },
      { 3123244280, 879 },
      { 3187066832, 165 },
      { 3209399506, 491 },
      { 3230260738, 349 },
      { 3344189994, 211 },
      { 3345707173, 337 },
      { 3367298820, 168 },
      { 3397078357, 249 },
      { 3569736966, 237 },
      { 3816961131, 828 },
      { 4091670162, 349 },
      { 4237497041, 330 },
      { kMarkvNoneOfTheAbove, 3020 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpReturnValue, 0), std::move(codec));
  }

  return codecs;
}
