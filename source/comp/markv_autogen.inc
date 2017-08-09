
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
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 3424 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 3176 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 2604 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4168 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 17307 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 26006 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 2416 },
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
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1528 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 6481 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 6143 },
      { kMarkvNoneOfTheAbove, 7813 },
    }));

    codecs.emplace(SpvOpFMul, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 10216 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 3868 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1217 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 11769 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 227 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 13619 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 8167 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 47493 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 34792 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 33797 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 18592 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 236 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2647 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 9214 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 25159 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 9578 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 12024 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 40033 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 6167 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3486 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 12880 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 4639 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 6240 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 3770 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 8662 },
      { kMarkvNoneOfTheAbove, 17166 },
    }));

    codecs.emplace(SpvOpFAdd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 181653 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 3592 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 22370 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 16194 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 6 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 6164 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1208 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 667 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2865 },
      { kMarkvNoneOfTheAbove, 9780 },
    }));

    codecs.emplace(SpvOpTypePointer, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 481 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4640 },
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
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4657 },
      { kMarkvNoneOfTheAbove, 2213 },
    }));

    codecs.emplace(SpvOpFSub, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 82380 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 18335 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 29082 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 12085 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleDrefExplicitLod, 7), 26146 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 411888 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 13835 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 84211 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 12449 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 6102 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 26522 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 28805 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 38172 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 5709 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 328 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 61873 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 2535 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 35580 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 13374 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 10221 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 16008 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 3192 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 5449 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 1056 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 2408 },
      { kMarkvNoneOfTheAbove, 47135 },
    }));

    codecs.emplace(SpvOpCompositeExtract, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 3262 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 6663 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 2960 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 104343 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 7124 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 14352 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 391556 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 7796 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 19985 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 9101 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 13382 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 211 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 829 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 276 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1080 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 6988 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 123 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 53605 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 488 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 11515 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1 },
      { kMarkvNoneOfTheAbove, 24893 },
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
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 14 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 51 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 789 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 348 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3599 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 5861 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 1956 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 2455 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 42272 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 33922 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 7179 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 517 },
      { CombineOpcodeAndNumOperands(SpvOpPhi, 6), 9677 },
      { CombineOpcodeAndNumOperands(SpvOpKill, 0), 4020 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1684 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 170 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1826 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 7 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 7087 },
      { kMarkvNoneOfTheAbove, 7127 },
    }));

    codecs.emplace(SpvOpLabel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1019 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2423 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 209221 },
      { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 21998 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2203 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 23178 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 65 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 1743 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 104 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 15046 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 166986 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 12581 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 469 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 29667 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 5 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 56176 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 39174 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1832 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 463 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 895 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 431 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 7 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 498 },
      { kMarkvNoneOfTheAbove, 24843 },
    }));

    codecs.emplace(SpvOpStore, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 7), 2 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 6), 2200 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 82716 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 13661 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 67125 },
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
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 45388 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 5861 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 153858 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 56687 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 48604 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 34729 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1238 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 32528 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 19573 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 6217 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 7136 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 25162 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 38811 },
      { kMarkvNoneOfTheAbove, 68383 },
    }));

    codecs.emplace(SpvOpLoad, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 684 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 14110 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 4748 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 11467 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 9868 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 11377 },
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
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 44524 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 6693 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 16 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 12655 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 436 },
      { kMarkvNoneOfTheAbove, 11983 },
    }));

    codecs.emplace(SpvOpCompositeConstruct, std::move(codec));
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
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 16288 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 3643 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 2527 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 5349 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2897 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 11010 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 9559 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 95125 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 10888 },
      { kMarkvNoneOfTheAbove, 2065 },
    }));

    codecs.emplace(SpvOpConstantComposite, std::move(codec));
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
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 10483 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 64856 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 41330 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 195059 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 9601 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 46831 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 35273 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 28243 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 13079 },
      { kMarkvNoneOfTheAbove, 12937 },
    }));

    codecs.emplace(SpvOpConstant, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 15 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 4682 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4436 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 11942 },
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
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 6574 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 3723 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 8319 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 13259 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 27004 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 21730 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 625 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 7029 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 58196 },
      { kMarkvNoneOfTheAbove, 6425 },
    }));

    codecs.emplace(SpvOpExtInst, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 381 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 15 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 2766 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 7119 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 35761 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 20869 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 6423 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 14927 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 423 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 36 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 12084 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 14222 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 709 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 8268 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 10398 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 5984 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2465 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 3592 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 3835 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 5116 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 4 },
      { kMarkvNoneOfTheAbove, 3263 },
    }));

    codecs.emplace(SpvOpVectorTimesScalar, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4420 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 26630 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 439 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 4976 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 23378 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 39955 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 41569 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 18558 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 85740 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 78211 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 9875 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 73737 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 20004 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 51881 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 11783 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 20395 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1077 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 14950 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1761 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 15548 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 6067 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 7422 },
      { kMarkvNoneOfTheAbove, 18902 },
    }));

    codecs.emplace(SpvOpVectorShuffle, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
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
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 4 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 440 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 27 },
      { kMarkvNoneOfTheAbove, 1653 },
    }));

    codecs.emplace(SpvOpImageSampleImplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
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
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 28 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 14574 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 11873 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2905 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 3764 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 8488 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 2217 },
      { kMarkvNoneOfTheAbove, 1778 },
    }));

    codecs.emplace(SpvOpDot, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 61 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 195 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 62 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 6), 10572 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 31114 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 4107 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 8103 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 2732 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 8 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 3153 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeInsert, 5), 8979 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 40252 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 2335 },
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
      { kMarkvNoneOfTheAbove, 3779 },
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
      { 26, 21184 },
      { 31, 10071 },
      { 40, 79897 },
      { 43, 78206 },
      { 46, 76057 },
      { 49, 4019 },
      { 68, 8273 },
      { 69, 66141 },
      { 71, 6379 },
      { kMarkvNoneOfTheAbove, 20178 },
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
      { kMarkvNoneOfTheAbove, 223 },
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
      { 10, 4972 },
      { 981668463, 4888 },
      { 1055437881, 9474 },
      { 1056964608, 17499 },
      { 1065353216, 18701 },
      { 1073741824, 14579 },
      { 3212836864, 8554 },
      { 3332128768, 5774 },
      { kMarkvNoneOfTheAbove, 238801 },
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
      { 30, 154654 },
      { 33, 81053 },
      { 34, 203509 },
      { kMarkvNoneOfTheAbove, 6574 },
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
      { 15, 6334 },
      { 16, 19387 },
      { 64, 15692 },
      { kMarkvNoneOfTheAbove, 25582 },
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
      { 37, 47007 },
      { kMarkvNoneOfTheAbove, 69148 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4, 94035 },
      { 7, 94035 },
      { 35, 1843200 },
      { kMarkvNoneOfTheAbove, 26203 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 84284 },
      { 16, 141303 },
      { 28, 26463 },
      { 32, 47213 },
      { 40, 26386 },
      { 44, 25824 },
      { 48, 47176 },
      { 60, 31332 },
      { 64, 62843 },
      { 76, 36076 },
      { 80, 47098 },
      { 92, 21711 },
      { 96, 47113 },
      { 108, 20406 },
      { 112, 47047 },
      { 120, 20406 },
      { 128, 62732 },
      { 140, 37375 },
      { 144, 46957 },
      { 148, 21705 },
      { 152, 26521 },
      { 156, 36072 },
      { 160, 46971 },
      { 176, 46943 },
      { 192, 62657 },
      { 204, 21721 },
      { 208, 31268 },
      { 224, 31280 },
      { 256, 39822 },
      { kMarkvNoneOfTheAbove, 716932 },
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
      { 5, 6397 },
      { 6, 1503 },
      { kMarkvNoneOfTheAbove, 1667 },
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
      { 8, 13948 },
      { 9, 13648 },
      { 10, 10496 },
      { 12, 12176 },
      { 16, 11384 },
      { 23, 32128 },
      { 24, 12712 },
      { kMarkvNoneOfTheAbove, 135192 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 31904 },
      { 1, 20033 },
      { 2, 20948 },
      { 3, 24408 },
      { kMarkvNoneOfTheAbove, 27969 },
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
      { kMarkvNoneOfTheAbove, 2251 },
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
      { 2161102232, 4811 },
      { kMarkvNoneOfTheAbove, 386290 },
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
      { 139011596, 10009 },
      { 810488476, 4591 },
      { 870594305, 4102 },
      { 1742737136, 4886 },
      { 2096388952, 6878 },
      { 2855506940, 20158 },
      { 3044188332, 4811 },
      { 3487022798, 4171 },
      { 3701632935, 4025 },
      { kMarkvNoneOfTheAbove, 327470 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 139011596, 4521 },
      { 296981500, 7796 },
      { 1367301635, 3776 },
      { 2855506940, 72723 },
      { 3233393284, 3128 },
      { 3251128023, 12499 },
      { 3582002820, 2832 },
      { kMarkvNoneOfTheAbove, 175910 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 68240 },
      { 508217552, 2063 },
      { 2683080096, 1951 },
      { 3547456240, 1820 },
      { 3753486980, 3123 },
      { kMarkvNoneOfTheAbove, 82667 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 135486769, 4099 },
      { 440421571, 4980 },
      { 450406196, 2596 },
      { 503094540, 4320 },
      { 543621065, 7788 },
      { 827698488, 6438 },
      { 907126242, 4841 },
      { 908777857, 4841 },
      { 910429472, 3182 },
      { 1294403159, 6309 },
      { 1296054774, 6050 },
      { 1297706389, 5621 },
      { 1322549027, 6084 },
      { 1784441183, 3052 },
      { 2080953106, 4274 },
      { 2194691858, 10303 },
      { 2448331885, 3858 },
      { 2468230023, 2957 },
      { 2547657777, 15339 },
      { 2549309392, 13490 },
      { 2550961007, 12727 },
      { 2934934694, 2989 },
      { 2936586309, 2639 },
      { 2938237924, 2725 },
      { 3094180193, 4500 },
      { 3094857332, 3973 },
      { 3095831808, 7788 },
      { 3183924418, 10249 },
      { 3561562003, 18879 },
      { 3563213618, 5294 },
      { 3564865233, 4594 },
      { 4028622909, 13556 },
      { kMarkvNoneOfTheAbove, 38665 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 119981689, 3052 },
      { 162255877, 7788 },
      { 679771963, 32314 },
      { 1154919607, 13556 },
      { 1343794461, 4500 },
      { 1674803691, 4099 },
      { 1951208733, 22622 },
      { 2263349224, 4498 },
      { 2320303498, 28849 },
      { 2924146124, 10249 },
      { 2984325996, 7788 },
      { 3334207724, 48357 },
      { 3410158390, 4320 },
      { 3489360962, 2816 },
      { 3866587616, 2596 },
      { 3868239231, 6084 },
      { 3869890846, 10303 },
      { kMarkvNoneOfTheAbove, 35209 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 2), std::move(codec));
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
      { 142465290, 10625 },
      { 144116905, 9543 },
      { 210116709, 9474 },
      { 296981500, 18701 },
      { 529742207, 8164 },
      { 959681532, 17499 },
      { 1156369516, 10624 },
      { 1158021131, 8511 },
      { 1543646433, 6153 },
      { 1782996825, 10684 },
      { 1784648440, 9177 },
      { 2170273742, 9233 },
      { 2321729979, 4888 },
      { 2444465148, 8554 },
      { 2524697596, 5774 },
      { 2557550659, 4972 },
      { 2796901051, 10748 },
      { 2798552666, 18329 },
      { 2855506940, 18523 },
      { 3184177968, 7338 },
      { 3810805277, 9648 },
      { 3929248764, 14579 },
      { kMarkvNoneOfTheAbove, 238933 },
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
      { 139011596, 13776 },
      { 249378857, 3769 },
      { 251209228, 1945 },
      { 503145996, 2894 },
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
      { 3133016299, 3888 },
      { 3251128023, 9474 },
      { 3504158761, 10890 },
      { 3538592682, 4828 },
      { 3540244297, 4831 },
      { 3541895912, 4831 },
      { 3570219049, 2428 },
      { 3653838348, 7071 },
      { 3764205609, 8382 },
      { 3882634684, 3888 },
      { 3913885196, 7043 },
      { 4243119782, 3888 },
      { kMarkvNoneOfTheAbove, 19474 },
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
      { 910398460, 1951 },
      { 959681532, 3725 },
      { 1039111164, 2430 },
      { 1087394637, 3888 },
      { 1156369516, 4828 },
      { 2444465148, 5689 },
      { 2732195517, 3888 },
      { 2796901051, 4831 },
      { 2855506940, 41703 },
      { 3202349435, 3888 },
      { 3362723943, 3888 },
      { 3712763835, 3888 },
      { 3929248764, 8615 },
      { 4016096296, 3888 },
      { 4248015868, 2007 },
      { kMarkvNoneOfTheAbove, 19531 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 52882140, 3888 },
      { 210116709, 9474 },
      { 296981500, 23181 },
      { 385229009, 3888 },
      { 910398460, 1947 },
      { 959681532, 3668 },
      { 1031290113, 3888 },
      { 1039111164, 2430 },
      { 1172110445, 3888 },
      { 1622381564, 1650 },
      { 1782996825, 17658 },
      { 1971252067, 3888 },
      { 2444465148, 5664 },
      { 2490492987, 3888 },
      { 2678954464, 3888 },
      { 2855506940, 41580 },
      { 3912967080, 3888 },
      { 3929248764, 8710 },
      { 4248015868, 2011 },
      { kMarkvNoneOfTheAbove, 14905 },
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
      { kMarkvNoneOfTheAbove, 3197 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 4864 },
      { 615748604, 1200 },
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
      { kMarkvNoneOfTheAbove, 610 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543621065, 50382 },
      { 827698488, 31911 },
      { 1294403159, 40166 },
      { 1296054774, 73442 },
      { 1297706389, 23479 },
      { 1322549027, 7499 },
      { 2194691858, 48598 },
      { 2468230023, 51648 },
      { 2547657777, 59039 },
      { 2549309392, 12073 },
      { 2550961007, 19702 },
      { 2934934694, 53926 },
      { 2936586309, 17687 },
      { 2938237924, 14790 },
      { 3095831808, 43266 },
      { 3183924418, 10218 },
      { 3561562003, 51481 },
      { 4028622909, 13531 },
      { kMarkvNoneOfTheAbove, 82087 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 48598 },
      { 137840602, 17687 },
      { 565334834, 7499 },
      { 625975427, 10218 },
      { 630964591, 19702 },
      { 680016782, 13531 },
      { 1009983433, 14790 },
      { 1572088444, 51648 },
      { 1584774136, 50382 },
      { 1918481917, 40166 },
      { 2790624748, 73442 },
      { 3181646225, 59039 },
      { 3560665067, 53926 },
      { 3662767579, 23479 },
      { 4053789056, 12073 },
      { 4064212479, 43266 },
      { 4192247221, 51481 },
      { 4224872590, 31911 },
      { kMarkvNoneOfTheAbove, 82087 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 162255877, 63938 },
      { 679771963, 209209 },
      { 1951208733, 375067 },
      { 2320303498, 97959 },
      { 2984325996, 56390 },
      { 3334207724, 326179 },
      { 3869890846, 61816 },
      { kMarkvNoneOfTheAbove, 95073 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543558236, 56110 },
      { 810488476, 37985 },
      { 870594305, 18683 },
      { 883854656, 60901 },
      { 1570165302, 21190 },
      { 1684282922, 26076 },
      { 1901166356, 15318 },
      { 1949759310, 63658 },
      { 2087004702, 19741 },
      { 2096388952, 88899 },
      { 2517964682, 37601 },
      { 2622612602, 62157 },
      { 2970183398, 24585 },
      { 3091876332, 14937 },
      { 3187066832, 46898 },
      { 3496407048, 52811 },
      { 3570411982, 68029 },
      { 3692647551, 12734 },
      { kMarkvNoneOfTheAbove, 557318 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 60901 },
      { 137840602, 24585 },
      { 522971108, 46898 },
      { 630964591, 37601 },
      { 1572088444, 26076 },
      { 1584774136, 63658 },
      { 1918481917, 62157 },
      { 1957218950, 15318 },
      { 2313593054, 14937 },
      { 2790624748, 88899 },
      { 2838165089, 12734 },
      { 3181646225, 68029 },
      { 3364388739, 18683 },
      { 3560665067, 52811 },
      { 3662767579, 21190 },
      { 4064212479, 56110 },
      { 4224872590, 37985 },
      { 4239834800, 19741 },
      { kMarkvNoneOfTheAbove, 557318 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 137840602, 23401 },
      { 1009983433, 14792 },
      { 1572088444, 54630 },
      { 1918481917, 58464 },
      { 2790624748, 110466 },
      { 3560665067, 61574 },
      { 3662767579, 27441 },
      { 4192247221, 52730 },
      { 4224872590, 50029 },
      { kMarkvNoneOfTheAbove, 157504 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 177111659, 6083 },
      { 296981500, 6088 },
      { 408465899, 8381 },
      { 2055836767, 8540 },
      { 2087004702, 16395 },
      { 2096388952, 13614 },
      { 2622612602, 8920 },
      { 2855506940, 9672 },
      { 2959147533, 8501 },
      { 3187066832, 14439 },
      { 3570411982, 7666 },
      { 3831290364, 8378 },
      { kMarkvNoneOfTheAbove, 494354 },
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
      { 3561562003, 16461 },
      { kMarkvNoneOfTheAbove, 23549 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1079999262, 10176 },
      { 2311941439, 15646 },
      { 2313593054, 17658 },
      { 2838165089, 15812 },
      { 2839816704, 13281 },
      { 2841468319, 10107 },
      { 3364388739, 23228 },
      { 3366040354, 10563 },
      { 3367691969, 9286 },
      { 3369343584, 8554 },
      { 4239834800, 20420 },
      { 4241486415, 9510 },
      { 4243138030, 10095 },
      { 4244789645, 9695 },
      { kMarkvNoneOfTheAbove, 414872 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 112745085, 6022 },
      { 116376005, 11317 },
      { 400248103, 8928 },
      { 406044930, 14594 },
      { 522971108, 9186 },
      { 625975427, 28360 },
      { 680016782, 27906 },
      { 1009983433, 7074 },
      { 1062250709, 9534 },
      { 1410849099, 10560 },
      { 1918481917, 51631 },
      { 2190437442, 23694 },
      { 2790624748, 39200 },
      { 2879917723, 13248 },
      { 2882994691, 24268 },
      { 3181646225, 15686 },
      { 3263901372, 11745 },
      { 3390051757, 24308 },
      { 3560665067, 49720 },
      { 3662767579, 33304 },
      { 3717523241, 8504 },
      { 3945795573, 6897 },
      { 4101009465, 5940 },
      { 4290024976, 38424 },
      { kMarkvNoneOfTheAbove, 118853 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 29886 },
      { 144116905, 65121 },
      { 529742207, 14081 },
      { 1156369516, 25119 },
      { 1158021131, 48703 },
      { 1543646433, 13130 },
      { 1782996825, 98487 },
      { 1784648440, 63380 },
      { 1930923350, 5938 },
      { 2170273742, 24423 },
      { 2557550659, 7744 },
      { 2705477184, 14450 },
      { 2796901051, 24244 },
      { 2798552666, 47648 },
      { 3184177968, 14454 },
      { 3810805277, 19142 },
      { 4198082194, 10573 },
      { kMarkvNoneOfTheAbove, 72380 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 5160 },
      { 144116905, 5552 },
      { 198967948, 917 },
      { 529742207, 1024 },
      { 825595257, 1017 },
      { 1156369516, 3980 },
      { 1158021131, 15046 },
      { 1452222566, 1073 },
      { 1782996825, 8937 },
      { 1784648440, 8065 },
      { 1839499483, 1017 },
      { 2170273742, 1188 },
      { 2466126792, 1004 },
      { 2796901051, 4614 },
      { 2798552666, 5678 },
      { 3184177968, 1004 },
      { 3480031018, 1017 },
      { 3810805277, 1299 },
      { 4106658327, 1073 },
      { 4198082194, 896 },
      { kMarkvNoneOfTheAbove, 20852 },
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
      { 837715723, 8081 },
      { 1203545131, 27250 },
      { 1352628475, 7058 },
      { 1367301635, 20971 },
      { 2055836767, 16365 },
      { 2204920111, 7357 },
      { 3619787319, 21822 },
      { 3701632935, 9269 },
      { kMarkvNoneOfTheAbove, 451837 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 488500848, 6483 },
      { 495107308, 12067 },
      { 2096388952, 10459 },
      { 2157103435, 8640 },
      { 2622612602, 45615 },
      { 3496407048, 45072 },
      { 3570411982, 68988 },
      { 3713290482, 5842 },
      { kMarkvNoneOfTheAbove, 374113 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 495107308, 11673 },
      { 2096388952, 10619 },
      { 2622612602, 29950 },
      { 3496407048, 38031 },
      { 3570411982, 68437 },
      { kMarkvNoneOfTheAbove, 418569 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 199758 },
      { 2320303498, 76406 },
      { 3334207724, 144141 },
      { kMarkvNoneOfTheAbove, 12767 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1319785741, 6170 },
      { 3753486980, 9693 },
      { kMarkvNoneOfTheAbove, 417209 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 17937 },
      { 959681532, 5276 },
      { 1141965917, 7559 },
      { 2855506940, 10484 },
      { 3091876332, 5830 },
      { kMarkvNoneOfTheAbove, 385986 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 269823086, 7477 },
      { 296981500, 17955 },
      { 959681532, 12976 },
      { 2219733501, 5692 },
      { 2855506940, 21902 },
      { kMarkvNoneOfTheAbove, 367070 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 8290 },
      { 959681532, 4669 },
      { 1227221002, 3663 },
      { 2855506940, 12748 },
      { 3692647551, 7146 },
      { kMarkvNoneOfTheAbove, 317765 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 18776483, 1600 },
      { 296981500, 23843 },
      { 959681532, 1962 },
      { 2855506940, 19013 },
      { 3874089391, 1738 },
      { kMarkvNoneOfTheAbove, 98807 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 60816 },
      { 1951208733, 780809 },
      { 2320303498, 31784 },
      { 3334207724, 101284 },
      { kMarkvNoneOfTheAbove, 20187 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { kMarkvNoneOfTheAbove, 994880 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 495107308, 12488 },
      { 850497536, 70744 },
      { 1584369690, 32340 },
      { 1890300748, 18320 },
      { 2043873558, 54992 },
      { 2338272340, 50624 },
      { 3312467582, 37632 },
      { 3504158761, 10272 },
      { kMarkvNoneOfTheAbove, 707468 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 11808 },
      { 630592085, 1880 },
      { 679771963, 8738 },
      { 899570100, 17230 },
      { 929101967, 5712 },
      { 1100599986, 30526 },
      { 1103903216, 3240 },
      { 1369578001, 14610 },
      { 2320303498, 11309 },
      { 2926633629, 2112 },
      { 3334207724, 16899 },
      { 3486057732, 2272 },
      { 3705139860, 3360 },
      { 3800912395, 15609 },
      { 3822983876, 7390 },
      { 4141567741, 4872 },
      { 4292991777, 2200 },
      { kMarkvNoneOfTheAbove, 9073 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { kMarkvNoneOfTheAbove, 168840 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 1941 },
      { 2517964682, 5432 },
      { 2855506940, 10084 },
      { 3044188332, 3012 },
      { 3570411982, 16600 },
      { 3764205609, 4800 },
      { kMarkvNoneOfTheAbove, 126971 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 371428004, 2272 },
      { 1543280290, 2136 },
      { 2162986400, 2112 },
      { 3808408202, 2544 },
      { kMarkvNoneOfTheAbove, 159776 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeInsert, 3), std::move(codec));
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
      { 488500848, 6208 },
      { 495107308, 14623 },
      { 1858116930, 2524 },
      { 2231688008, 1824 },
      { 2693892518, 2112 },
      { 3566035349, 2112 },
      { kMarkvNoneOfTheAbove, 92022 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 883854656, 59847 },
      { 2036361232, 49574 },
      { 2356768706, 6015 },
      { 2637132451, 6012 },
      { 3237903670, 2064 },
      { 3829682756, 2340 },
      { kMarkvNoneOfTheAbove, 1940 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 646282397, 2524 },
      { 1352628475, 6208 },
      { 1543798545, 2112 },
      { 1545450160, 2112 },
      { 2517964682, 6367 },
      { 2532518896, 1824 },
      { 3619787319, 14623 },
      { kMarkvNoneOfTheAbove, 92022 },
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
      { kMarkvNoneOfTheAbove, 345659 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 5847 },
      { 959681532, 6597 },
      { 1570165302, 6797 },
      { 2096388952, 14194 },
      { 3653838348, 4273 },
      { kMarkvNoneOfTheAbove, 307951 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 643418617, 4224 },
      { 959681532, 10798 },
      { 1092948665, 3890 },
      { 2517964682, 4861 },
      { 2683080096, 4085 },
      { kMarkvNoneOfTheAbove, 317801 },
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
      { 615982737, 2376 },
      { 1139547465, 3792 },
      { 1178317551, 2112 },
      { 2330636993, 2924 },
      { 2589449658, 2136 },
      { 3579593979, 1644 },
      { 3730093054, 2112 },
      { 3944781937, 2832 },
      { kMarkvNoneOfTheAbove, 142701 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 249378857, 2450 },
      { 296981500, 41800 },
      { 1203545131, 5388 },
      { 1265796414, 2832 },
      { 1319785741, 2403 },
      { 2855506940, 8667 },
      { 3091876332, 2930 },
      { 3187066832, 4170 },
      { 3508792859, 2120 },
      { 3619787319, 2919 },
      { 3653838348, 9160 },
      { kMarkvNoneOfTheAbove, 77790 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 5608 },
      { 443558693, 2376 },
      { 1203545131, 2617 },
      { 1265796414, 8288 },
      { 1319785741, 2629 },
      { 1558001705, 2400 },
      { 1955104493, 1644 },
      { 2234361374, 2136 },
      { 2598189097, 2400 },
      { 2775815164, 1947 },
      { 3244209297, 4163 },
      { 3753486980, 5112 },
      { 3886529747, 2112 },
      { 4069810315, 2832 },
      { 4164704452, 2112 },
      { 4273793488, 2112 },
      { kMarkvNoneOfTheAbove, 112141 },
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
      { kMarkvNoneOfTheAbove, 416913 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 810488476, 7688 },
      { 1054461787, 4359 },
      { 1158929937, 4359 },
      { 1203545131, 5232 },
      { 2096388952, 8913 },
      { 3929248764, 4458 },
      { 4008405264, 4992 },
      { kMarkvNoneOfTheAbove, 376912 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 171307615, 5440 },
      { 593829839, 7056 },
      { 959681532, 5471 },
      { 1684282922, 7779 },
      { 1811839150, 5440 },
      { 3929248764, 6232 },
      { kMarkvNoneOfTheAbove, 379495 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 3), std::move(codec));
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
      { 2504802016, 1596 },
      { 3032677281, 1596 },
      { 3560552546, 1596 },
      { 3797961332, 1605 },
      { 3886529747, 2112 },
      { kMarkvNoneOfTheAbove, 150156 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 354479447, 2136 },
      { 1325348861, 1944 },
      { 1367301635, 2214 },
      { 1368383673, 2112 },
      { 1646147798, 1944 },
      { 1679946323, 1944 },
      { 1766401548, 2112 },
      { 2096388952, 8757 },
      { 2123388694, 1944 },
      { 2362972044, 2112 },
      { 2660843182, 1751 },
      { 2698156268, 1944 },
      { 2970183398, 2605 },
      { 3133016299, 1944 },
      { 3187066832, 8690 },
      { 3882634684, 1944 },
      { 4243119782, 1944 },
      { kMarkvNoneOfTheAbove, 110620 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 810488476, 3010 },
      { 959681532, 3067 },
      { 1232501371, 4032 },
      { 1372785527, 1729 },
      { 1526654696, 2136 },
      { 1684282922, 3198 },
      { 1901166356, 1680 },
      { 2244928358, 1729 },
      { 2314864456, 2880 },
      { 2524697596, 2112 },
      { 2568098594, 4992 },
      { 3117071189, 1732 },
      { 3188115516, 1920 },
      { 3554463148, 2112 },
      { 3691770462, 2088 },
      { 3929248764, 15114 },
      { 4060703604, 1920 },
      { 4092487128, 1920 },
      { kMarkvNoneOfTheAbove, 101290 },
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
      { 170690025, 1599 },
      { 669982125, 2376 },
      { 1625742020, 2286 },
      { 2071351379, 2690 },
      { 2291766425, 2112 },
      { 3104643263, 2215 },
      { 3602108619, 2112 },
      { kMarkvNoneOfTheAbove, 137155 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50385656, 1574 },
      { 615982737, 2376 },
      { 837715723, 1599 },
      { 1237148906, 2784 },
      { 1364157225, 1740 },
      { 1499923635, 1638 },
      { 1766401548, 5006 },
      { 2012838864, 6120 },
      { 2096388952, 3896 },
      { 2161102232, 1550 },
      { 2197874825, 6432 },
      { 2279700640, 3200 },
      { 2362972044, 4248 },
      { 2589449658, 4248 },
      { 2881302403, 3451 },
      { 2936040203, 3552 },
      { 2970183398, 4006 },
      { 3187066832, 4804 },
      { 3362830643, 1560 },
      { 3538158875, 2400 },
      { 3635542517, 4272 },
      { 3997432565, 2480 },
      { kMarkvNoneOfTheAbove, 79609 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 139011596, 1897 },
      { 342159236, 1600 },
      { 837715723, 1599 },
      { 876867882, 1600 },
      { 1356063462, 1600 },
      { 1766401548, 5024 },
      { 2096388952, 3621 },
      { 2197874825, 3624 },
      { 2362972044, 2136 },
      { 2517964682, 17246 },
      { 2589449658, 2136 },
      { 2683080096, 3200 },
      { 2919626325, 2286 },
      { 2936040203, 2328 },
      { 2970183398, 3307 },
      { 2996594997, 1600 },
      { 3187066832, 2702 },
      { 3913885196, 5826 },
      { 3997432565, 1940 },
      { 4010499223, 1600 },
      { kMarkvNoneOfTheAbove, 85673 },
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

  return codecs;
}
