time sh build_host.sh; install-x86-64/bin/spirv-opt --fix-mali-spec-constant-composite opselect.spv -o opselect_opt.spv
spirv-dis opselect.spv > opselect.spv.txt
spirv-dis opselect_opt.spv > opselect_opt.spv.txt
spirv-val opselect_opt.spv
