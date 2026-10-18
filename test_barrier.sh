time sh build_host.sh
install-x86-64/bin/spirv-opt --mali-optimization-barrier optbarrier.spv -o optbarrier_opt.spv
spirv-dis optbarrier.spv > optbarrier.spv.txt
spirv-dis optbarrier_opt.spv > optbarrier_opt.spv.txt
spirv-val optbarrier_opt.spv
