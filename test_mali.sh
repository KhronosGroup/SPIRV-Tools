time sh build_host.sh

opt() {
  FILE=$1
  FILE_OPT="${FILE%.*}_opt.spv"

  echo "Optimizing $FILE"
  install-x86-64/bin/spirv-opt \
      --remove-clip-cull-dist --fix-mali-spec-constant-composite --mali-optimization-barrier \
      $FILE -o $FILE_OPT
  spirv-dis $FILE > $FILE_OPT.txt
  spirv-dis $FILE_OPT > $FILE_OPT.txt
  spirv-val $FILE_OPT
  echo ""
}

opt gta_frag.spv
opt gta_vertex.spv
opt shader.spv
opt opselect.spv
opt optbarrier.spv
opt failing_shader.spv
